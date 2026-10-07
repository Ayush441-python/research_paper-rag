import os
import requests
import streamlit as st

# Configure API URL from environment variable or default to local FastAPI server
API_URL = os.getenv("API_URL", "http://localhost:8000/api").rstrip("/")

st.set_page_config(page_title="Research RAG", page_icon="📚", layout="centered")

st.title("🔬 Research Paper Assistant")
st.caption("Upload a PDF research paper and ask questions about its content.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")

    # Backend status badge
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        if res.ok:
            st.success("Backend: Connected ✅")
        else:
            st.warning("Backend: Error ⚠️")
    except Exception:
        st.error("Backend: Offline ❌")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file and st.button("Upload & Index", use_container_width=True):
        with st.spinner("Processing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                if res.ok:
                    st.success("Document indexed! 🎉")
                else:
                    st.error(f"Upload failed ({res.status_code})")
            except Exception as e:
                st.error(f"Connection error: {e}")

    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User prompt input
if prompt := st.chat_input("Ask a question about your paper..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt},
                    timeout=120
                )
                if res.ok:
                    data = res.json()
                    answer = data.get("answer", data.get("response", "No answer returned."))
                else:
                    answer = f"Backend error: {res.status_code}"
            except Exception as e:
                answer = f"Error connecting to server: {e}"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})