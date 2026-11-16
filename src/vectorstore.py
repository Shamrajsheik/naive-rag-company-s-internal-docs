from pathlib import Path
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


class VectorStore:
    """
    Manages vector storage and similarity retrieval using ChromaDB.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = "company_docs",
        in_memory: bool = False,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.in_memory = in_memory

        if in_memory or persist_dir is None:
            self.client: ClientAPI = chromadb.Client()
        else:
            persist_path = Path(persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)
            self.client: ClientAPI = chromadb.PersistentClient(path=str(persist_path))

        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def count(self) -> int:
        """Return the number of items in the collection."""
        return self.collection.count()

    def index_chunks(
        self,
        chunks: List[Dict[str, str]],
        force_reindex: bool = False,
    ) -> int:
        """
        Index a list of chunk dictionaries into ChromaDB.
        Each chunk must have 'text' and 'Source' keys.
        """
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reindex:
            return existing_count

        if force_reindex and existing_count > 0:
            # Delete existing collection and recreate
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

        documents = []
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk["text"])
            ids.append(f"chunk_{i}")
            metadatas.append({"source": chunk["Source"]})

        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

        return len(documents)

    def retrieve(
        self,
        question: str,
        n_results: int = 3,
    ) -> Tuple[List[str], List[Dict[str, any]], List[float]]:
        """
        Query ChromaDB for top-n most relevant chunks.
        Returns: (chunks, metadatas, distances)
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results,
        )

        chunks = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        return chunks, metadatas, distances

