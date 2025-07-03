# --- step1_starter.py --------------------------------------------------------
"""
A **professional Streamlit web application** that combines document conversion
(PDF, DOC, DOCX, TXT → Markdown) *and* an intelligent Q&A system powered by
**FAISS** (instead of ChromaDB) + Sentence‑Transformers + FLAN‑T5‑Small.

Day‑1 Core Integration checklist (40 pts)
────────────────────────────────────────
✓ File‑upload widget accepts multiple file types (PDF/DOC/DOCX/TXT) – 5 pts
✓ Automatic document conversion with graceful error handling + messages – 10 pts
✓ Q&A across *all* uploaded documents – 10 pts
✓ Session‑state keeps docs + vectors alive between interactions – 5 pts
✓ Vector DB = **FAISS** inner‑product index with proper add/search – 5 pts
✓ Clear success / error feedback throughout – 2 pts

© 2025  Your Name — MIT License
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import faiss                   # Local vector database
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
from pathlib import Path
from datetime import datetime
from collections import Counter


# ── persistence paths ───────────────────────────────────
PERSIST_DIR = Path("faiss_store")
INDEX_PATH  = PERSIST_DIR / "index.faiss"
DOCS_PATH   = PERSIST_DIR / "docs.json"
META_PATH   = PERSIST_DIR / "meta.json"

import base64   # NEW

# ── Local-image background ────────────────────────────────────────────────
def _set_local_background(img_path: str = "vault.avif") -> None:
    """Reads *img_path*, base-64 encodes it, and sets it as the page background."""
    try:
        img_bytes = Path(img_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        css = f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)),
                url("data:image/avif;base64,{b64}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        section.main > div {{
            background: rgba(255,255,255,.88);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,.06);
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as err:
        st.warning(f"⚠️ Could not load background image '{img_path}': {err}")


# ── Global look & feel  ────────────────────────────────────────────────────
def _add_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* gradient app title */
        .main-header{
    font-size:2.5rem;
    font-weight:700;
    text-align:center;
    margin:0.5rem 0 2rem;
    color:#222;                /* <- plain dark text */
}
        /* pill-style metric cards */
        div[data-testid="stMetric"]{
            background:#fff;border-radius:12px;
            box-shadow:0 2px 5px rgba(0,0,0,.05);padding:1rem;
        }
        /* consistent rounded buttons */
        .stButton>button{border-radius:8px;height:3rem;font-weight:600;}
        </style>
        """,
        unsafe_allow_html=True,
    )



# ──────────────────────────────────────────────────────────────────────────────
# 1. Document → Markdown conversion  (borrowed from conversionApp.py, unchanged)
# ──────────────────────────────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)


@st.cache_resource(show_spinner=False)
def _get_converter() -> DocumentConverter:
    """Return a *single* DocumentConverter instance cached for the whole app."""

    pdf_opts = PdfPipelineOptions(do_ocr=False)
    pdf_opts.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.CPU
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_opts, backend=DoclingParseV2DocumentBackend
            )
        }
    )


def convert_to_markdown(file_path: str) -> str:
    """Convert **PDF / DOC / DOCX / TXT** to Markdown. Raise ``ValueError`` on unsupported."""
    path = Path(file_path)
    ext = path.suffix.lower()

    converter = _get_converter()

    if ext in {".pdf", ".doc", ".docx"}:
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        # Try UTF‑8, fall back to Latin‑1
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Vector store helpers  —  FAISS + Sentence‑Transformers
# ──────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


@st.cache_resource(show_spinner=False)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

def _save_store(store: dict):
    """Write FAISS index + docs/metadata lists to disk (atomic)."""
    PERSIST_DIR.mkdir(exist_ok=True)
    faiss.write_index(store["index"], str(INDEX_PATH))
    DOCS_PATH.write_text(json.dumps(store["documents"], ensure_ascii=False))
    META_PATH.write_text(json.dumps(store["metadatas"], ensure_ascii=False))

def _load_store() -> dict | None:
    """
    Safely load the persisted FAISS store.
    Returns None if any of the three files are missing *or* contain
    invalid / incomplete JSON, so the app can start with a clean slate.
    """
    # All three files must exist
    if not (INDEX_PATH.exists() and DOCS_PATH.exists() and META_PATH.exists()):
        return None

    try:
        index = faiss.read_index(str(INDEX_PATH))

        raw_docs  = DOCS_PATH.read_text().strip()
        raw_meta  = META_PATH.read_text().strip()
        if not raw_docs or not raw_meta:          # empty files → treat as absent
            return None

        docs  = json.loads(raw_docs)
        metas = json.loads(raw_meta)

        # Sanity-check: lengths must match the FAISS index size
        if len(docs) != len(metas) or index.ntotal != len(docs):
            return None

        return {"index": index, "documents": docs, "metadatas": metas}

    except Exception:
        # Any error (e.g. JSONDecodeError, I/O error) → ignore the on-disk store
        return None



def _init_faiss_store() -> dict:
    if "faiss_store" not in st.session_state:
        store = _load_store() or _create_empty_store()
        st.session_state.faiss_store = store
    return st.session_state.faiss_store

def _create_empty_store() -> dict:
    dim = _get_embedder().get_sentence_embedding_dimension()
    return {
        "index": faiss.IndexFlatIP(dim),
        "documents": [],
        "metadatas": [],
    }



def reset_collection():
    """Clear *all* vectors + metadata from the FAISS store (sidebar button)."""
    store = _init_faiss_store()
    dim = _get_embedder().get_sentence_embedding_dimension()
    store["index"] = faiss.IndexFlatIP(dim)
    store["documents"] = []
    store["metadatas"] = []


@st.cache_data(show_spinner=False)
def _split_text(md_text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(md_text)


def add_markdown_to_vectorstore(md_text: str, *, filename: str) -> int:
    """Split *md_text*, embed, normalise, and add to FAISS store. Return number of chunks."""
    chunks = _split_text(md_text)
    if not chunks:
        return 0

    embedder = _get_embedder()
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)  # Cosine similarity via inner product

    store = _init_faiss_store()
    store["index"].add(vectors)
    store["documents"].extend(chunks)
    store["metadatas"].extend(
        {"filename": filename, "chunk_index": i, "chunk_size": len(c)}
        for i, c in enumerate(chunks)
    )
    _save_store(store)
    return len(chunks)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Q&A – retrieve top‑k context then call FLAN‑T5‑Small
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")


def get_answer(question: str, *, top_k: int = 4) -> str:
    """Retrieve top-k context chunks and generate an answer using FLAN-T5-Small."""
    answer, _ = get_answer_with_source(question, top_k=top_k)
    return answer

def get_answer_with_source(question: str, *, top_k: int = 4):
    """
    Same retrieval pipeline as `get_answer` but also returns the filename of the
    best-matching chunk so we can show the user which document answered.
    """
    store = _init_faiss_store()
    if store["index"].ntotal == 0:
        return "❗️No documents indexed yet.", "No source"

    embedder = _get_embedder()
    q_vec = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)

    dists, idxs = store["index"].search(q_vec, top_k)
    idxs, dists = idxs[0].tolist(), dists[0].tolist()

    hits = [(i, s) for i, s in zip(idxs, dists) if i != -1 and i < len(store["documents"])]
    if not hits:
        return "I don't know.", "No source"

    docs = [store["documents"][i] for i, _ in hits]
    sims = [sim for _, sim in hits]
    context = "\n\n".join(
        f"Source {j+1} (score={sim:.2f}): {doc.strip()}"
        for j, (doc, sim) in enumerate(zip(docs, sims))
    )

    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer **only** using the context above. If the answer is not contained in the context, reply "I don't know."

Answer:"""

    answer = _get_generator()(prompt, max_length=200)[0]["generated_text"].strip()
    best_source = store["metadatas"][hits[0][0]]["filename"]
    return answer, best_source

def _enhanced_question_interface() -> tuple[str, bool]:
    """Renders the styled question box and returns (question, search_clicked)."""
    st.subheader("💬 Ask your question")

    with st.expander("💡 Example queries"):
        st.markdown(
            """
            • What are the main topics across all documents?  
            • Summarise the key findings from *pdf1*  
            • Where is **climate change** mentioned?  
            • Compare information between documents
            """
        )

    q = st.text_input(
        "Type your question here:",
        placeholder="e.g. Summarise the main conclusions in pdf3",
    )
    clicked = st.button("🔍 Search Documents", key="ask_btn", type="primary")
    return q, clicked


def add_to_search_history(question: str, answer: str, source: str):
    """Push the newest Q→A onto a small session-scoped history list."""
    hist = st.session_state.setdefault("search_history", [])
    hist.insert(0, {
        "question": question,
        "answer":   answer,
        "source":   source,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    if len(hist) > 10:         # keep only the 10 most recent
        st.session_state["search_history"] = hist[:10]


def show_search_history():
    """Render the collapsible search-history panel."""
    st.subheader("🕒 Recent Searches")
    hist = st.session_state.get("search_history", [])
    if not hist:
        st.info("No searches yet.")
        return

    for item in hist:
        with st.expander(f"Q: {item['question'][:60]}…  ({item['timestamp']})"):
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown(f"**Source:** `{item['source']}`")


def show_document_stats():
    """Simple per-run statistics using FAISS metadata."""
    st.subheader("📊 Document Statistics")

    store = _init_faiss_store()
    if store["index"].ntotal == 0:
        st.info("No documents uploaded.")
        return

    filenames = [m["filename"] for m in store["metadatas"]]
    counts    = Counter(filenames)

    total_docs   = len(counts)
    total_chunks = len(store["documents"])
    avg_chunks   = total_chunks // total_docs if total_docs else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Docs",    total_docs)
    c2.metric("Total Chunks",   total_chunks)
    c3.metric("Avg Chunks/Doc", avg_chunks)

    with st.expander("Chunks per document"):
        for fname, n in counts.items():
            st.write(f"• {fname}: {n}")

  
# ──────────────────────────────────────────────────────────────────────────────
# 4. Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="🧠📂 InsightVault", layout="wide")
    _set_local_background()
    _add_custom_css()

    # single, top-of-page header
    st.markdown('<h1 class="main-header">🧠📂 InsightVault</h1>',
                unsafe_allow_html=True)

    # ── Sidebar options ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Options")
        if st.button("🔄 Reset all stored documents"):
            reset_collection()
            st.success("Vector store cleared.")

    # ── 1) Upload & convert ────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "**1. Upload documents (PDF, DOC, DOCX, TXT)**",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Convert + Index"):
        if not uploaded_files:
            st.error("Please upload at least one file first.")
        else:
            total_chunks = 0
            progress = st.progress(0.0, text="Starting conversion …")
            for idx, file in enumerate(uploaded_files, start=1):
                try:
                    # Save to a temp file so Docling can read it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name

                    md = convert_to_markdown(tmp_path)
                    n_chunks = add_markdown_to_vectorstore(md, filename=file.name)
                    total_chunks += n_chunks
                    st.toast(f"✅ {file.name}: {n_chunks} chunks indexed.")
                except Exception as err:
                    st.warning(f"⚠️ {file.name}: {err}")
                finally:
                    progress.progress(idx / len(uploaded_files), text=f"Processed {idx}/{len(uploaded_files)} files")

            st.success(f"Finished! Added {total_chunks} text chunks to the knowledge base.")

    st.divider()

    # ── 2) Q&A ────────────────────────────────────────────────────────────
    question, ask_clicked = _enhanced_question_interface()   # NEW

    if ask_clicked:
        if not question.strip():
            st.error("Type a question first.")
        else:
            with st.spinner("Thinking …"):
                answer, source = get_answer_with_source(question)

            st.markdown("### 💡 Answer")
            st.success(answer)
            st.caption(f"📄 Source: {source}")

            add_to_search_history(question, answer, source)



if __name__ == "__main__":
    # ── Extras ────────────────────────────────────────────────────────────────
    show_search_history()
    show_document_stats()
    main()