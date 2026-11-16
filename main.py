import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

from src.pipeline import NaiveRAGPipeline
from src.loader import COMPANY_DOCS_DIR
from src.generator import DEFAULT_GROQ_MODEL
from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, DEFAULT_TOP_K


def print_banner():
    banner = """
============================================================
       🏢 NovaTech Solutions — Internal Docs RAG
             Naive RAG System (Groq + ChromaDB)
============================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="Naive RAG System for Internal Company Documents"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Ask a single question against the company documents",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive Q&A session in the terminal",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of all documents in the knowledge base",
    )
    parser.add_argument(
        "--n_results", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve (default: 3)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_GROQ_MODEL,
        help=f"Groq model name (default: {DEFAULT_GROQ_MODEL})",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional Groq API Key (or enter it in the Streamlit UI / environment)",
    )

    args = parser.parse_args()

    pipeline = NaiveRAGPipeline(
        docs_dir=COMPANY_DOCS_DIR,
        chroma_persist_dir=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        groq_api_key=args.api_key,
        groq_model=args.model,
    )

    # Ingest documents if collection is empty or re-indexing requested
    if pipeline.vector_store.count() == 0 or args.reindex:
        print(f"[*] Indexing documents from '{COMPANY_DOCS_DIR}'...")
        count = pipeline.ingest(force_reindex=args.reindex)
        print(f"[✓] Successfully indexed {count} chunks into collection '{COLLECTION_NAME}'\n")

    if args.query:
        pipeline.ask(args.query, n_results=args.n_results, verbose=True)
        return

    if args.interactive or len(sys.argv) == 1:
        print_banner()
        print("Type your question and press Enter. Type 'exit' or 'quit' to end.\n")

        while True:
            try:
                question = input("Question > ").strip()
                if not question:
                    continue
                if question.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                pipeline.ask(question, n_results=args.n_results, verbose=True)
            except KeyboardInterrupt:
                print("\nSession ended.")
                break
            except Exception as e:
                print(f"[!] Error: {e}\n")


if __name__ == "__main__":
    main()
