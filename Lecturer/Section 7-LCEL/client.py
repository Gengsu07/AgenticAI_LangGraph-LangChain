"""
Streamlit Client for LangServe API
====================================
Demonstrates calling multiple chain endpoints with proper error handling.

Run with:
    streamlit run client.py
"""

import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
API_BASE_URL = "http://127.0.0.1:8000"

# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────
def invoke_chain(endpoint: str, payload: dict) -> dict:
    """
    Invoke a LangServe chain endpoint with proper error handling.

    Args:
        endpoint: The chain path (e.g., "/translate")
        payload: The input dict matching the chain's expected schema

    Returns:
        The response JSON dict
    """
    url = f"{API_BASE_URL}{endpoint}/invoke"
    try:
        response = requests.post(
            url,
            json={"input": payload, "config": {}, "kwargs": {}},
            timeout=30,  # 30s timeout to handle slow LLM responses
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Cannot connect to API server. Is serve.py running?"}
    except requests.exceptions.Timeout:
        return {"error": "⏱️ Request timed out. The server might be overloaded."}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return {"error": "🚫 Rate limit exceeded. Please wait a moment."}
        return {"error": f"HTTP Error {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def check_server_health() -> bool:
    """Check if the API server is healthy."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM Application — LCEL", page_icon="🤖", layout="wide")
st.title("🤖 LLM Application Using LCEL")
st.caption("Powered by LangServe + FastAPI + Groq")

# Server status indicator
if check_server_health():
    st.success("✅ API Server is online")
else:
    st.error("❌ API Server is offline — run `python serve.py` first")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Tabs for different chains
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌍 Translator", "💬 Assistant", "📋 Summarizer"])

# --- Tab 1: Translation ---
with tab1:
    st.subheader("Translate Text")
    col1, col2 = st.columns([3, 1])
    with col1:
        translate_text = st.text_area(
            "Enter text to translate:",
            height=100,
            key="translate_input",
        )
    with col2:
        target_lang = st.selectbox(
            "Target Language:",
            ["French", "Spanish", "German", "Japanese", "Indonesian", "Korean",
             "Chinese", "Arabic", "Portuguese", "Italian"],
        )

    if st.button("🔄 Translate", key="btn_translate"):
        if translate_text.strip():
            with st.spinner("Translating..."):
                result = invoke_chain("/translate", {
                    "language": target_lang,
                    "text": translate_text,
                })
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Translation:")
                st.write(result.get("output", result))
        else:
            st.warning("Please enter some text to translate.")

# --- Tab 2: Assistant ---
with tab2:
    st.subheader("Ask the AI Assistant")
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., How can I improve my small business marketing strategy?",
        key="assistant_input",
    )

    if st.button("💡 Ask", key="btn_ask"):
        if question.strip():
            with st.spinner("Thinking..."):
                result = invoke_chain("/assistant", {"question": question})
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Answer:")
                st.write(result.get("output", result))
        else:
            st.warning("Please enter a question.")

# --- Tab 3: Summarizer ---
with tab3:
    st.subheader("Summarize Text")
    summarize_text = st.text_area(
        "Enter text to summarize:",
        height=150,
        placeholder="Paste a long article, report, or document here...",
        key="summarize_input",
    )

    if st.button("📋 Summarize", key="btn_summarize"):
        if summarize_text.strip():
            with st.spinner("Summarizing..."):
                result = invoke_chain("/summarize", {"text": summarize_text})
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Summary:")
                st.write(result.get("output", result))
        else:
            st.warning("Please enter text to summarize.")

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📖 API Docs: [Swagger UI](http://127.0.0.1:8000/docs) | "
    "[ReDoc](http://127.0.0.1:8000/redoc) | "
    "Built with LangChain + LangServe + FastAPI"
)