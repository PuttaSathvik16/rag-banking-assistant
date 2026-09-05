"""Streamlit UI for RAG Banking Assistant."""
import streamlit as st
from streamlit_extras.colored_header import colored_header
import os
from dotenv import load_dotenv
from src.pipeline import RAGPipeline
from src.schemas import AnswerResponse
import json

load_dotenv()

st.set_page_config(
    page_title="Banking Q&A Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #e8f4f8;
        border-left: 3px solid #1f77b4;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .abstained-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .retrieved-doc {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline once and cache it."""
    return RAGPipeline()

def confidence_color(confidence: float) -> str:
    """Return CSS class based on confidence score."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_answer(response: AnswerResponse):
    """Display the answer with citations and metadata."""
    col1, col2 = st.columns([3, 1])

    with col1:
        colored_header(
            label="Answer",
            description="",
            color_name="blue"
        )

    with col2:
        if response.abstained:
            st.info("⚠️ Abstained - Insufficient confidence")
        else:
            conf_class = confidence_color(response.confidence)
            st.markdown(f'<p class="{conf_class}">Confidence: {response.confidence:.1%}</p>',
                       unsafe_allow_html=True)

    # Display answer content
    if response.abstained:
        st.markdown("""
        <div class='abstained-box'>
        <strong>Unable to provide a confident answer</strong><br>
        The corpus does not contain sufficient information to answer this question with confidence.
        Please consult with a banking representative for assistance.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"**{response.answer}**", unsafe_allow_html=True)

        # Display citations
        if response.citations:
            st.markdown("### 📎 Sources")
            for i, citation in enumerate(response.citations, 1):
                st.markdown(f"""
                <div class='citation-badge'>
                <strong>Source {i}:</strong> {citation.source}<br>
                <em>{citation.text[:150]}...</em>
                </div>
                """, unsafe_allow_html=True)

def display_metadata(response: AnswerResponse, sub_queries: list, retrieved_docs: list, reranked_docs: list):
    """Display retrieval and generation metadata."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div class='metric-card'>
        <h4>Retrieved Docs</h4>
        <p style='font-size: 2rem; margin: 0;'>{}</p>
        </div>""".format(len(retrieved_docs)), unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class='metric-card'>
        <h4>Reranked (Top-K)</h4>
        <p style='font-size: 2rem; margin: 0;'>{}</p>
        </div>""".format(len(reranked_docs)), unsafe_allow_html=True)

    with col3:
        st.markdown("""<div class='metric-card'>
        <h4>Sub-queries</h4>
        <p style='font-size: 2rem; margin: 0;'>{}</p>
        </div>""".format(len(sub_queries)), unsafe_allow_html=True)

    # Display sub-queries
    if len(sub_queries) > 1:
        with st.expander("📝 Query Decomposition"):
            for i, sq in enumerate(sub_queries, 1):
                st.markdown(f"**Query {i}:** {sq}")

    # Display retrieved documents
    with st.expander("📄 Retrieved Documents (Before Reranking)"):
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs[:5], 1):  # Show top 5
                with st.container():
                    st.markdown(f"""
                    <div class='retrieved-doc'>
                    <strong>📌 Document {i}</strong><br>
                    <strong>Source:</strong> {doc.get('source', 'Unknown')}<br>
                    <strong>Relevance Score:</strong> {doc.get('score', 'N/A')}<br>
                    <em>{doc.get('text', '')[:200]}...</em>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No documents retrieved")

    # Display reranked documents
    with st.expander("⭐ Reranked Documents (Top-K)"):
        if reranked_docs:
            for i, doc in enumerate(reranked_docs[:3], 1):  # Show top 3
                with st.container():
                    st.markdown(f"""
                    <div class='retrieved-doc'>
                    <strong>✓ Rank {i}</strong><br>
                    <strong>Source:</strong> {doc.get('source', 'Unknown')}<br>
                    <strong>Text:</strong> {doc.get('text', '')[:250]}...
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No documents reranked")

def main():
    st.markdown('<p class="main-header">🏦 Banking Q&A Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about banking products, fees, and eligibility policies</p>',
                unsafe_allow_html=True)

    # Initialize pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {str(e)}")
        st.info("Make sure your `.env` file contains `GOOGLE_API_KEY` and run `python cli.py setup` first.")
        return

    # Sidebar with example questions
    with st.sidebar:
        st.header("📚 Example Questions")
        example_questions = [
            "What is the minimum balance for Premium Savings?",
            "What are the fees for the Rewards Visa card?",
            "Am I eligible for a personal loan?",
            "What documents do I need for a home improvement loan?",
            "How much does a wire transfer cost?",
        ]

        selected_example = st.selectbox(
            "Try an example:",
            [""] + example_questions,
            index=0
        )

    # Question input
    question = st.text_area(
        "Ask a question about banking products and policies:",
        value=selected_example,
        height=100,
        placeholder="e.g., What is the annual percentage rate for a personal loan?"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("🔍 Search", use_container_width=True, type="primary")

    # Process question
    if submit_button and question.strip():
        with st.spinner("Searching for answer..."):
            try:
                response: AnswerResponse = pipeline.answer(question)

                # For demonstration, we'll create mock metadata
                # In production, these would come from the pipeline
                sub_queries = [question]  # Simplified for demo
                retrieved_docs = response.context if hasattr(response, 'context') else []
                reranked_docs = response.context if hasattr(response, 'context') else []

                # Display results
                st.success("✅ Answer generated successfully!")
                st.divider()

                # Display answer
                display_answer(response)

                st.divider()

                # Display metadata
                colored_header(
                    label="Retrieval & Generation Details",
                    description="",
                    color_name="gray"
                )
                display_metadata(response, sub_queries, retrieved_docs, reranked_docs)

                # Display raw JSON for debugging
                with st.expander("🔧 Raw Response (JSON)"):
                    st.json(response.model_dump())

            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                st.info("Please check your API key and corpus setup.")
    else:
        # Show placeholder content
        if question.strip() == "" and not submit_button:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("💡 **Tip:** Use example questions from the sidebar or type your own")
            with col2:
                st.info("🎯 **Best for:** Product details, fees, and eligibility questions")
            with col3:
                st.info("⚠️ **Note:** The system will abstain if it lacks sufficient information")

if __name__ == "__main__":
    main()
