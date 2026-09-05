"""Streamlit UI for RAG Banking Assistant."""
import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Banking Q&A Assistant",
    page_icon="🏦",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .main-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .citation {
        background-color: #e8f4f8;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
        border-radius: 4px;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .abstained {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-title">🏦 Banking Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown("*Ask questions about banking products, fees, and eligibility policies*")
    st.divider()

    # Check for required setup
    check_setup = st.empty()

    try:
        # Import pipeline
        from src.pipeline import RAGPipeline

        # Load pipeline
        if 'pipeline' not in st.session_state:
            with st.spinner("Loading RAG pipeline..."):
                st.session_state.pipeline = RAGPipeline()

        pipeline = st.session_state.pipeline

        # Sidebar - Examples
        with st.sidebar:
            st.header("📚 Examples")
            examples = [
                "What is the minimum balance for Premium Savings?",
                "What are the fees for the Rewards Visa card?",
                "What is the APR for a personal loan?",
                "Am I eligible for a home improvement loan?",
                "How much does a wire transfer cost?",
            ]
            selected = st.radio("Try an example:", ["Custom"] + examples)

            if selected == "Custom":
                selected = ""

        # Question input
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_area(
                "Your question:",
                value=selected if selected != "Custom" else "",
                height=80,
                placeholder="Ask about banking products, fees, eligibility, etc."
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.button("🔍 Search", use_container_width=True, type="primary")

        # Process question
        if search_button and question.strip():
            try:
                with st.spinner("🔄 Searching for answer..."):
                    response = pipeline.answer(question)

                st.success("✅ Answer generated!")
                st.divider()

                # Display answer
                if response.abstained:
                    st.markdown("""
                    <div class="abstained">
                    <strong>⚠️ Unable to provide a confident answer</strong><br>
                    The corpus does not contain sufficient information to answer this question confidently.
                    Please consult with a banking representative.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Confidence color
                    if response.confidence >= 0.8:
                        conf_class = "confidence-high"
                    elif response.confidence >= 0.6:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"

                    st.markdown(f'<p class="{conf_class}">✓ Confidence: {response.confidence:.0%}</p>',
                               unsafe_allow_html=True)

                    # Answer
                    st.markdown(f'<div class="answer-box">{response.answer}</div>', unsafe_allow_html=True)

                    # Citations
                    if response.citations:
                        st.subheader("📎 Sources")
                        for i, citation in enumerate(response.citations, 1):
                            with st.container():
                                st.markdown(f"""
                                <div class="citation">
                                <strong>Source {i}:</strong> {citation.source}<br>
                                <small>{citation.text[:200]}...</small>
                                </div>
                                """, unsafe_allow_html=True)

                # Show raw response
                with st.expander("🔧 Raw Response (JSON)"):
                    st.json(response.model_dump())

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Try asking a simpler question or check your API key.")

        else:
            if not search_button:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info("💡 Select an example or type your own question")
                with col2:
                    st.info("🎯 Ask about products, fees, and eligibility")
                with col3:
                    st.info("⚠️ System abstains if uncertain")

    except ImportError as e:
        st.error("❌ Missing dependencies")
        st.warning("""
        Please install dependencies first:
        ```bash
        pip install -r requirements.txt
        ```
        """)
        st.stop()

    except Exception as e:
        st.error(f"❌ Setup Error: {str(e)}")
        st.warning("""
        Make sure you have:
        1. Installed dependencies: `pip install -r requirements.txt`
        2. Created `.env` file with `GOOGLE_API_KEY`
        3. Run setup: `python cli.py setup`
        """)
        st.stop()

if __name__ == "__main__":
    main()
