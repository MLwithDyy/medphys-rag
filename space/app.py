import spaces  # must be imported before torch on ZeroGPU Spaces

@spaces.GPU(duration=1)
def _zerogpu_startup_check():
    # Never called. ZeroGPU refuses to start a Space with no @spaces.GPU function;
    # this app is CPU-only, so the function only exists to pass that check.
    return None

import os, re, json, time, datetime, collections
import numpy as np, faiss, gradio as gr
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GATE, K, CAP = 0.78, 8, 3            # topic gate, chunks given to the LLM, max chunks per document
PER_IP_HOUR, GLOBAL_DAY = 10, 300    # protect the free API quota

# ---------- data and indexes ----------
chunks = [json.loads(l) for l in open("chunks_filtered.jsonl", encoding="utf-8")]
E = np.load("embeddings_e5small.npy").astype("float32")
assert len(E) == len(chunks), "embeddings and chunks do not match"
index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
emb = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
tok = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok(c["title"] + " " + c["section"] + " " + c["text"]) for c in chunks])

def rrf(*rankings, c=60):
    sc = {}
    for r in rankings:
        for pos, j in enumerate(r): sc[j] = sc.get(j, 0) + 1 / (c + pos + 1)
    return [j for j, _ in sorted(sc.items(), key=lambda x: -x[1])]

def cap(ranking, per_doc):
    cnt, kept = {}, []
    for j in ranking:
        p = chunks[j]["pmcid"]; cnt[p] = cnt.get(p, 0) + 1
        if cnt[p] <= per_doc: kept.append(j)
    return kept

def retrieve(q):
    qv = emb.encode(["query: " + q], normalize_embeddings=True).astype("float32")
    s, i = index.search(qv, 100)
    d = [int(j) for j in i[0]]
    b = [int(j) for j in np.argsort(-bm25.get_scores(tok(q)))[:100]]
    return cap(rrf(d, b), CAP)[:K], float(s[0][0])

SYS = """You answer questions about medical physics using ONLY the numbered context passages.
Rules:
- If the passages do not contain the answer, set "answerable" to false and "answer" to a short note saying the documents do not cover it. Do not use outside knowledge.
- Otherwise answer concisely (1-3 sentences) and copy numbers exactly as written in the passages.
- Cite the passages you used by number in "citations".
- Answer in the same language as the question.
Reply with JSON only: {"answerable": true|false, "answer": "...", "citations": [1, 2]}"""

# ---------- usage limits (shared key only) ----------
hits, day = collections.defaultdict(list), {"d": None, "n": 0}
def allowed(ip):
    now, today = time.time(), datetime.date.today()
    hits[ip] = [t for t in hits[ip] if now - t < 3600]
    if day["d"] != today: day.update(d=today, n=0)
    if len(hits[ip]) >= PER_IP_HOUR or day["n"] >= GLOBAL_DAY: return False
    hits[ip].append(now); day["n"] += 1
    return True

def sources_md(ids, cited):
    lines = []
    for n in cited:
        c = chunks[ids[n - 1]]
        url = f"https://europepmc.org/article/PMC/{c['pmcid']}"
        lines.append(f"**[{n}] {c['title']}**  \n{c['section']} · [Europe PMC]({url}) · CC BY\n\n> {c['text'][:350]}...")
    return "\n\n".join(lines)

def ask(question, user_key, request: gr.Request):
    q = (question or "").strip()
    if len(q) < 5: return "Tulis pertanyaan yang lebih lengkap.", ""
    if len(q) > 500: return "Pertanyaan maksimal 500 karakter.", ""
    ids, top = retrieve(q)
    if top < GATE:
        return "Pertanyaan ini tampaknya di luar topik fisika medis yang dicakup dokumen, jadi tidak dijawab.", ""
    key = (user_key or "").strip()
    if not key:
        ip = "unknown"
        if request is not None:
            fwd = request.headers.get("x-forwarded-for") if request.headers else None
            ip = (fwd or getattr(request.client, "host", "unknown")).split(",")[0].strip()
        if not allowed(ip):
            return "Kuota demo gratis sedang habis. Coba lagi nanti, atau isi kunci API Gemini milikmu di kolom di bawah.", ""
    ctx = "\n\n".join(f"[{n+1}] ({chunks[j]['title'][:90]} | {chunks[j]['section'][:50]})\n{chunks[j]['text']}"
                      for n, j in enumerate(ids))
    try:
        client = OpenAI(api_key=key or os.environ["GEMINI_API_KEY"], base_url=BASE_URL)
        r = client.chat.completions.create(model=MODEL, max_tokens=3000, temperature=0, messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}])
        d = json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))
    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg.lower():
            return "Kuota API sedang penuh. Coba lagi beberapa menit lagi.", ""
        return "Terjadi kesalahan saat membuat jawaban. Coba lagi.", ""
    cited = [n for n in d.get("citations", []) if isinstance(n, int) and 1 <= n <= len(ids)]
    if not d.get("answerable"):
        return "Dokumen yang tersedia tidak memuat jawaban untuk pertanyaan ini.", ""
    return d.get("answer", ""), sources_md(ids, cited)

INTRO = """# MedPhys RAG
Tanya jawab fisika medis dengan sitasi, dari 33 artikel open access (CC BY) di Europe PMC. Boleh bertanya dalam bahasa Indonesia atau Inggris.

**Alat edukasi, bukan panduan klinis.** Jawaban hanya berasal dari dokumen yang diindeks, dan model menolak menjawab jika dokumen tidak memuatnya. Jangan memasukkan data pasien."""

with gr.Blocks(title="MedPhys RAG") as demo:
    gr.Markdown(INTRO)
    q = gr.Textbox(label="Pertanyaan", lines=2, placeholder="Contoh: What machine learning algorithm was used to predict daily MLC positional deviations?")
    btn = gr.Button("Tanya", variant="primary")
    ans = gr.Markdown(label="Jawaban")
    src = gr.Markdown(label="Sumber")
    with gr.Accordion("Opsional: pakai kunci API Gemini sendiri", open=False):
        key = gr.Textbox(label="Kunci API (tidak disimpan)", type="password")
    gr.Examples([["What machine learning algorithm was used to predict daily MLC positional deviations?"],
                 ["Apa keuntungan adaptive radiotherapy pada kanker prostat?"],
                 ["Berapa batas dosis lensa mata untuk pekerja radiasi?"]], inputs=[q])
    btn.click(ask, [q, key], [ans, src])
    q.submit(ask, [q, key], [ans, src])

demo.queue(max_size=20).launch(ssr_mode=False)
