from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.loader import COMPANY_DOCS_DIR, resolve_docs_dir, load_documents_from_dir
from src.chunker import chunk_all_documents
from src.vectorstore import VectorStore
from src.generator import RAGGenerator, DEFAULT_GROQ_MODEL, FALLBACK_GROQ_MODEL
from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, DEFAULT_TOP_K


class NaiveRAGPipeline:
    """
    End-to-end Naive RAG system for internal company documents.
    Works completely out-of-the-box without requiring a .env file.
    """

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        chroma_persist_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        groq_api_key: Optional[str] = None,
        groq_model: str = DEFAULT_GROQ_MODEL,
        in_memory: bool = False,
    ):
        self.docs_dir = resolve_docs_dir(docs_dir)
        self.vector_store = VectorStore(
            persist_dir=chroma_persist_dir or CHROMA_PERSIST_DIR,
            collection_name=collection_name,
            in_memory=in_memory,
        )
        self.generator = RAGGenerator(
            api_key=groq_api_key,
            model=groq_model,
        )
        self.indexed_documents: List[Dict[str, any]] = []

    def ingest(self, force_reindex: bool = False) -> int:
        """
        Load text documents from the company documents path, chunk them, and index into ChromaDB.
        Returns total number of chunks indexed.
        """
        self.indexed_documents = load_documents_from_dir(self.docs_dir)
        all_chunks = chunk_all_documents(self.indexed_documents)
        count = self.vector_store.index_chunks(all_chunks, force_reindex=force_reindex)
        return count

    def retrieve(
        self,
        question: str,
        n_results: int = DEFAULT_TOP_K,
    ) -> Tuple[List[str], List[Dict[str, any]], List[float]]:
        """Retrieve top relevant chunks and their source metadata."""
        return self.vector_store.retrieve(question, n_results=n_results)

    def ask(
        self,
        question: str,
        n_results: int = DEFAULT_TOP_K,
        verbose: bool = True,
    ) -> Dict[str, any]:
        """
        Execute full RAG pipeline: Retrieve relevant context and generate answer.
        """
        chunks, metadatas, distances = self.retrieve(question, n_results=n_results)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Question: {question}")
            print(f"{'-' * 60}")
            print(f"Retrieved {len(chunks)} chunks:")
            for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
                source = meta.get("source", "Unknown")
                preview = chunk.replace("\n", " ")[:80]
                print(f"  [{source}]  {preview}...")
            print(f"{'-' * 60}")

        try:
            answer = self.generator.generate(
                question=question,
                chunks=chunks,
                metadatas=metadatas,
            )
        except Exception as e:
            if "GROQ_API_KEY" in str(e):
                answer = "[!] Notice: GROQ_API_KEY is not set. Please enter it in the Streamlit UI sidebar or provide it via environment variable to generate answers."
            else:
                raise e

        if verbose:
            print(f"Answer: {answer}")
            print(f"{'=' * 60}\n")

        return {
            "question": question,
            "answer": answer,
            "chunks": chunks,
            "sources": metadatas,
            "distances": distances,
        }
