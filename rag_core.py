import re
from io import BytesIO
from typing import List, Tuple

from pypdf import PdfReader


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.replace("\x00", " ")
        pages.append(t)
    return "\n\n".join(pages)


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    overlap = max(0, min(overlap, max_chars - 1))

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if end == n:
            break
    return chunks


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]{2,}", s.lower())


def score_chunk(query: str, chunk: str) -> int:
    q = set(tokenize(query))
    c = tokenize(chunk)
    if not q or not c:
        return 0
    hits = sum(1 for w in c if w in q)
    uniq = len(set(c) & q)
    return hits + 2 * uniq


def top_k_chunks(query: str, chunks: List[dict], k: int = 4) -> List[Tuple[int, dict]]:
    scored = [(score_chunk(query, item["text"]), item) for item in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, item) for s, item in scored[:k] if s > 0]


def build_context(question: str, chunks: List[dict], k: int) -> Tuple[str, List[Tuple[int, dict]]]:
    selected = top_k_chunks(question, chunks, k=k)
    if selected:
        ctx = "\n\n".join(
            [
                f"[CHUNK score={s} doc={item['doc_name']} idx={item['chunk_id']}]\n{item['text']}"
                for s, item in selected
            ]
        )
        return ctx, selected

    fallback_items = chunks[:2]
    fallback_ctx = "\n\n".join(
        [
            f"[CHUNK score=0 doc={item['doc_name']} idx={item['chunk_id']}]\n{item['text']}"
            for item in fallback_items
        ]
    )
    return fallback_ctx, []
