import requests
import streamlit as st


st.set_page_config(
    page_title="Research RAG",
    page_icon="📚",
    layout="wide"
)

API_URL = "http://localhost:8000/api"


if "messages" not in st.session_state:
    st.session_state.messages = []


def api_request(method, endpoint, **kwargs):
    try:
        response = requests.request(
            method,
            f"{API_URL}{endpoint}",
            **kwargs
        )

        if response.ok:
            return response.json()

        return None

    except requests.RequestException:
        return None


st.sidebar.title("📚 Research RAG")

backend = api_request("GET", "/health", timeout=5)

if backend:
    st.sidebar.success("FastAPI Connected")
else:
    st.sidebar.error("FastAPI Offline")


st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload Research Paper",
    type=["pdf"]
)

if uploaded_file:

    if st.sidebar.button(
        "Upload & Ingest",
        use_container_width=True
    ):

        with st.spinner("Uploading..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            upload = api_request(
                "POST",
                "/upload",
                files=files,
                timeout=60
            )

        if upload:

            with st.spinner("Indexing document..."):

                ingestion = api_request(
                    "POST",
                    "/ingestion",
                    json={
                        "filename": upload.get(
                            "filename",
                            uploaded_file.name
                        )
                    },
                    timeout=120
                )

            if ingestion:
                st.sidebar.success("Document indexed ✅")
            else:
                st.sidebar.error("Ingestion failed")

        else:
            st.sidebar.error("Upload failed")


if st.sidebar.button(
    "Clear Chat",
    use_container_width=True
):
    st.session_state.messages = []
    st.rerun()


st.title("🔬 Research Paper Assistant")

st.caption(
    "Ask questions about your uploaded research paper."
)

st.divider()


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about your research paper..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = api_request(
                "POST",
                "/chat",
                json={"question": question},
                timeout=120
            )

        if result:

            answer = result.get(
                "answer",
                result.get("response", str(result))
            )

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        else:

            error = "Unable to generate an answer."

            st.error(error)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error
            })