from typing import Dict, List, Optional


def chunk_document(
    text: str,
    source_name: str,
    min_length: int = 50,
) -> List[Dict[str, str]]:
    """
    Chunk document by splitting on double newlines (\n\n),
    ignoring separator lines ('==') and paragraphs below min_length.
    Preserves exact concept from Naive RAG.ipynb.
    """
    paragraphs = text.strip().split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if len(para) < min_length:
            continue
        if para.startswith("=="):
            continue

        chunks.append({
            "text": para,
            "Source": source_name,
        })

    return chunks


def chunk_all_documents(
    documents: List[Dict[str, any]],
    min_length: int = 50,
) -> List[Dict[str, str]]:
    """
    Chunk a list of documents returned by the document loader.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(
            text=doc["text"],
            source_name=doc["source_name"],
            min_length=min_length,
        )
        all_chunks.extend(doc_chunks)

    return all_chunks

