
"""
AuroraStay QnA — Staff-facing chat UI (Streamlit)

The app talks to the Flask RAG backend. Configure the backend URL via:
  - the sidebar text box, or
  - the BACKEND_URL environment variable (default http://localhost:5000)
"""

import os
import requests
import streamlit as st

st.set_page_config(page_title="AuroraStay QnA", page_icon="🏨", layout="centered")

# ---------------- Configuration ----------------
DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://localhost:5000")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND)
    top_k = st.slider("Context chunks (top_k)", min_value=1, max_value=8, value=4)
    st.divider()
    st.caption("AuroraStay Hotels — internal POC")
    st.caption("Answers are grounded in policy documents only.")

# ---------------- Header ----------------
st.title("🏨 AuroraStay QnA")
st.write("Ask a guest-services question — answers are grounded in AuroraStay policy documents, with sources shown.")

# ---------------- Health check ----------------
try:
    health = requests.get(f"{backend_url.rstrip('/')}/health", timeout=5).json()
    st.success(f"Backend connected — {health.get('chunks', '?')} policy chunks loaded")
except Exception:
    st.error(f"Cannot reach backend at {backend_url}. Start the Flask API or check the URL in the sidebar.")
    st.stop()

# ---------------- Chat history ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 View supporting sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['document']}** — page {s['page']} "
                        f"(relevance {s['score']:.3f})"
                    )
                    st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))
                    st.divider()

# ---------------- Input ----------------
if prompt := st.chat_input("e.g. What is the pet fee at ASH-MIA Brickell?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents…"):
            try:
                resp = requests.post(
                    f"{backend_url.rstrip('/')}/query",
                    json={"question": prompt, "top_k": top_k},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                data = {"answer": f"Backend error: {e}", "sources": []}

        st.markdown(data["answer"])

        if data.get("sources"):
            with st.expander("📄 View supporting sources"):
                for s in data["sources"]:
                    st.markdown(
                        f"**{s['document']}** — page {s['page']} "
                        f"(relevance {s['score']:.3f})"
                    )
                    st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "sources": data.get("sources", [])}
    )
