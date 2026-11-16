import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load local .env file quietly in the background
load_dotenv()

from src.pipeline import NaiveRAGPipeline
from src.loader import load_documents_from_dir, COMPANY_DOCS_DIR
from src.generator import DEFAULT_GROQ_MODEL
from src.config import DEFAULT_TOP_K

# Page configuration
st.set_page_config(
    page_title="NovaTech Internal Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F172A;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chunk-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
    }
    .chunk-source {
        font-weight: 600;
        color: #1D4ED8;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .chunk-text {
        font-size: 0.9rem;
        color: #334155;
        white-space: pre-wrap;
    }
    .suggestion-box {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper to get API key quietly from .env or environment
def get_api_key() -> str:
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Initializing ChromaDB Knowledge Base...")
def get_rag_pipeline(api_key: str):
    """Initialize and cache the RAG pipeline."""
    pipeline = NaiveRAGPipeline(
        docs_dir=COMPANY_DOCS_DIR,
        groq_api_key=api_key,
        groq_model=DEFAULT_GROQ_MODEL,
    )
    # Ensure documents are indexed
    if pipeline.vector_store.count() == 0:
        pipeline.ingest()
    return pipeline


# Sidebar: Knowledge Base information only
with st.sidebar:
    st.header("🏢 Knowledge Base")
    st.caption(f"Backend Model: **{DEFAULT_GROQ_MODEL}**")

    # Document Indexing Status
    try:
        docs = load_documents_from_dir(COMPANY_DOCS_DIR)
        st.success(f"✓ {len(docs)} documents loaded & indexed")
        with st.expander("📄 Indexed Company Policies", expanded=True):
            for doc in docs:
                st.markdown(f"• **{doc['source_name']}** ({doc['word_count']} words)")
    except Exception as e:
        st.error(f"Error loading docs: {e}")

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=6,
        value=DEFAULT_TOP_K,
        help="Number of context paragraphs retrieved for each question",
    )

    if st.button("🔄 Re-index Knowledge Base"):
        with st.spinner("Re-indexing chunks into ChromaDB..."):
            pipeline = get_rag_pipeline(get_api_key())
            count = pipeline.ingest(force_reindex=True)
            st.success(f"Indexed {count} chunks successfully!")
            st.rerun()

    st.divider()
    st.caption("NovaTech Solutions — Internal Knowledge Retrieval System")


# Main Header
st.markdown('<div class="main-title">🏢 NovaTech Policy Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Semantic Search with ChromaDB + Grounded Generation via Groq</div>',
    unsafe_allow_html=True,
)

# Suggested Sample Questions across all 5 Company Documents
st.markdown('<div class="section-header">💡 Suggested Questions to Explore:</div>', unsafe_allow_html=True)

selected_sample = None

tab_hr, tab_eng, tab_onboard, tab_product, tab_sec = st.tabs([
    "🌴 HR & Leaves",
    "💻 Engineering Standards",
    "🚀 Onboarding",
    "💳 Product & Billing",
    "🔒 Security & IT",
])

with tab_hr:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 How many days of annual leave do employees get?", use_container_width=True):
            selected_sample = "How many days of annual leave do employees get?"
        if st.button("🏠 What is the work from home (WFH) policy?", use_container_width=True):
            selected_sample = "What is the work from home policy?"
    with col2:
        if st.button("⏱️ What happens during the probation period?", use_container_width=True):
            selected_sample = "What happens during the probation period?"
        if st.button("💰 What are the meal and travel expense reimbursement limits?", use_container_width=True):
            selected_sample = "What are the meal and travel expense reimbursement limits?"

with tab_eng:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔀 What are the pull request rules and approval requirements?", use_container_width=True):
            selected_sample = "What are the pull request rules and approval requirements?"
        if st.button("🌿 What is the Git branching strategy and naming convention?", use_container_width=True):
            selected_sample = "What is the Git branching strategy and naming convention?"
    with col2:
        if st.button("🧪 What is the minimum required test coverage for new code?", use_container_width=True):
            selected_sample = "What is the minimum required test coverage for new code?"
        if st.button("🗄️ What are the database schema and migration rules?", use_container_width=True):
            selected_sample = "What are the database schema and migration rules?"

with tab_onboard:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 What tasks must a new joiner complete on Day 1?", use_container_width=True):
            selected_sample = "What tasks must a new joiner complete on Day 1?"
    with col2:
        if st.button("🗺️ What is the 30-60-90 day onboarding roadmap?", use_container_width=True):
            selected_sample = "What is the 30-60-90 day onboarding roadmap?"

with tab_product:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("❌ How do I cancel my CloudDesk Pro subscription?", use_container_width=True):
            selected_sample = "How do I cancel my subscription?"
        if st.button("💵 What is the refund policy for subscriptions?", use_container_width=True):
            selected_sample = "What is the refund policy?"
    with col2:
        if st.button("🚫 What are the pricing tiers for CloudDesk Pro? (Negative Test)", use_container_width=True):
            selected_sample = "What are the pricing plan for CloudDesk Pro?"

with tab_sec:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 What are the password complexity and expiry rules?", use_container_width=True):
            selected_sample = "What are the password requirements and expiry rules according to security policy?"
    with col2:
        if st.button("🚨 How should an employee report a suspected security incident?", use_container_width=True):
            selected_sample = "How should an employee report a suspected security incident?"

st.markdown("---")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message and message["chunks"]:
            with st.expander("📚 Retrieved Context & Sources"):
                for i, (chunk, meta) in enumerate(zip(message["chunks"], message["sources"])):
                    source = meta.get("source", "Internal Policy")
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <div class="chunk-source">Chunk {i+1} — {source}</div>
                            <div class="chunk-text">{chunk}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Input for new question
user_query = st.chat_input("Ask a question about internal company policies...")
if selected_sample:
    user_query = selected_sample

if user_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        api_key = get_api_key()
        if not api_key:
            warning_msg = (
                "⚠️ **GROQ_API_KEY Not Found:** Please ensure your `GROQ_API_KEY` is set in your `.env` file."
            )
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        else:
            try:
                pipeline = get_rag_pipeline(api_key)
                with st.spinner("Retrieving relevant context and generating answer..."):
                    result = pipeline.ask(
                        question=user_query,
                        n_results=top_k,
                        verbose=False,
                    )

                answer = result["answer"]
                st.markdown(answer)

                if result["chunks"]:
                    with st.expander("📚 Retrieved Context & Sources", expanded=True):
                        for i, (chunk, meta) in enumerate(zip(result["chunks"], result["sources"])):
                            source = meta.get("source", "Internal Policy")
                            st.markdown(
                                f"""
                                <div class="chunk-card">
                                    <div class="chunk-source">Chunk {i+1} — {source}</div>
                                    <div class="chunk-text">{chunk}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": result["chunks"],
                    "sources": result["sources"],
                })

            except Exception as e:
                err_msg = f"❌ **Error running RAG pipeline:** {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
