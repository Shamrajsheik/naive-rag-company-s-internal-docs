from pathlib import Path
from typing import Dict, List, Optional
import os

# Root directory of the repository (resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Default in-code path to the company documents folder
# (Does not rely on .env, works immediately upon cloning or deployment)
COMPANY_DOCS_DIR = BASE_DIR / "company Info"

# Mapping for standard company document names to clean display names
DEFAULT_SOURCE_MAPPING = {
    "company_hr_policy.txt": "HR Policy",
    "engineering_standards.txt": "Engineering Standard",
    "onboarding_guide.txt": "Onboarding Guide",
    "product_knowledge_base.txt": "Product Knowledge Base",
    "security_policy.txt": "Security Policy",
}


def get_source_name(filename: str) -> str:
    """Derive a friendly source name from filename."""
    if filename in DEFAULT_SOURCE_MAPPING:
        return DEFAULT_SOURCE_MAPPING[filename]
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title()


def resolve_docs_dir(docs_dir: Optional[Path] = None) -> Path:
    """
    Automatically resolve the directory containing company .txt files.
    Searches standard relative locations so the application works seamlessly
    in local development, Git clones, Streamlit Cloud, Docker, or Google Colab
    WITHOUT requiring a .env file.
    """
    if docs_dir is not None:
        target = Path(docs_dir)
        if target.exists() and any(target.glob("*.txt")):
            return target.resolve()

    search_candidates = [
        COMPANY_DOCS_DIR,
        Path("./company Info"),
        Path("../company Info"),
        BASE_DIR / "company Info",
        Path("/content/company Info"),
        Path("/content"),
    ]

    for candidate in search_candidates:
        if candidate.exists() and any(candidate.glob("*.txt")):
            return candidate.resolve()

    # Fallback to the default path
    return COMPANY_DOCS_DIR.resolve()


def load_documents_from_dir(docs_dir: Optional[Path] = None) -> List[Dict[str, any]]:
    """
    Load all text documents from the resolved company documents directory.
    If docs_dir is omitted, it automatically resolves to 'company Info/'.
    """
    resolved_dir = resolve_docs_dir(docs_dir)

    if not resolved_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found at '{resolved_dir}'. "
            f"Please ensure the 'company Info' folder exists."
        )

    documents = []
    txt_files = sorted(resolved_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{resolved_dir}'.")

    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        filename = file_path.name
        source_name = get_source_name(filename)
        words = text.split()

        documents.append({
            "filename": filename,
            "source_name": source_name,
            "path": str(file_path),
            "text": text,
            "char_count": len(text),
            "word_count": len(words),
        })

    return documents


def load_named_documents(docs_dir: Optional[Path] = None) -> Dict[str, str]:
    """Convenience helper returning a dictionary keyed by document source name."""
    docs = load_documents_from_dir(docs_dir)
    return {doc["source_name"]: doc["text"] for doc in docs}
