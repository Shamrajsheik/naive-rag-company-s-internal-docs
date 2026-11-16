import unittest
from pathlib import Path
import sys

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import DOCS_DIR
from src.loader import load_documents_from_dir, DEFAULT_SOURCE_MAPPING
from src.chunker import chunk_document, chunk_all_documents
from src.generator import RAGGenerator
from src.config import STRICT_SYSTEM_PROMPT


class TestNaiveRAG(unittest.TestCase):

    def test_document_loader(self):
        """Verify that all 5 company documents are correctly loaded."""
        docs = load_documents_from_dir(DOCS_DIR)
        self.assertEqual(len(docs), 5, f"Expected 5 documents, found {len(docs)}")

        source_names = {d["source_name"] for d in docs}
        expected_sources = {
            "HR Policy",
            "Engineering Standard",
            "Onboarding Guide",
            "Product Knowledge Base",
            "Security Policy",
        }
        self.assertEqual(source_names, expected_sources)

        for doc in docs:
            self.assertGreater(doc["char_count"], 1000, f"Doc {doc['filename']} is unexpectedly short")
            self.assertGreater(doc["word_count"], 200, f"Doc {doc['filename']} word count is too low")

    def test_chunker(self):
        """Verify that chunking logic reproduces the 120 chunks from the original notebook."""
        docs = load_documents_from_dir(DOCS_DIR)
        all_chunks = chunk_all_documents(docs)

        # In Naive RAG.ipynb:
        # HR: 25, Engineering: 18, Onboarding: 22, Product: 33, Security: 22 -> Total: 120
        self.assertEqual(len(all_chunks), 120, f"Expected exactly 120 chunks, got {len(all_chunks)}")

        for chunk in all_chunks:
            self.assertIn("text", chunk)
            self.assertIn("Source", chunk)
            self.assertGreaterEqual(len(chunk["text"]), 50)
            self.assertFalse(chunk["text"].startswith("=="))

    def test_prompt_formatting(self):
        """Verify prompt construction adheres to strict grounding requirements."""
        generator = RAGGenerator(api_key="mock_key")
        sample_chunks = ["Employees may work from home up to 2 days per week."]
        question = "What is the WFH policy?"

        messages = generator.build_prompt_messages(question, sample_chunks)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], STRICT_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Question: What is the WFH policy?", messages[1]["content"])
        self.assertIn("Context: Employees may work from home up to 2 days per week.", messages[1]["content"])

    def test_empty_chunks_refusal(self):
        """Verify that empty retrieved chunks immediately refuse without hallucinating."""
        generator = RAGGenerator(api_key="mock_key")
        result = generator.generate("What is company policy?", chunks=[])
        self.assertEqual(result, "I don't have enough context or information to answer this question.")

    def test_vectorstore_indexing_and_retrieval(self):
        """Verify ChromaDB indexing and top-k retrieval functionality."""
        from src.vectorstore import VectorStore
        docs = load_documents_from_dir(DOCS_DIR)
        chunks = chunk_all_documents(docs)

        store = VectorStore(in_memory=True, collection_name="test_company_docs")
        count = store.index_chunks(chunks)
        self.assertEqual(count, 120)

        retrieved_chunks, metadatas, distances = store.retrieve(
            "What is work from home policy?",
            n_results=3,
        )
        self.assertEqual(len(retrieved_chunks), 3)
        self.assertEqual(len(metadatas), 3)
        # Verify that HR Policy is among the top retrieved sources
        sources = [m["source"] for m in metadatas]
        self.assertIn("HR Policy", sources)



if __name__ == "__main__":
    unittest.main()
