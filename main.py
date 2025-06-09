import os

# ── Fix OpenMP conflict (libomp vs libiomp) ─────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"



import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama  # updated imports
import whisper
import json
from htmltemplate import css, bot_template, user_template

VECTORSTORE_DIR = "vectorstore_index"


@st.cache_data
def load_json_chunks_with_metadata(json_files):
    chunks_with_metadata = []
    for json_file in json_files:
        data = json.load(json_file)
        json_file.seek(0)
        text = json.dumps(data, indent=2)
        metadata = {"source": json_file.name}
        chunks_with_metadata.append((text, metadata))
    return chunks_with_metadata


from langchain.text_splitter import CharacterTextSplitter

def extract_text_from_json(obj):
    """Recursively extract all text values from a nested JSON object."""
    texts = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                texts.extend(extract_text_from_json(value))
            else:
                texts.append(f"{key}: {value}")
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text_from_json(item))
    else:
        texts.append(str(obj))

    return texts


@st.cache_data
def split_chunks_with_metadata(chunks_with_metadata):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=50,
        length_function=len
    )

    texts, metadatas = [], []

    for json_obj, metadata in chunks_with_metadata:
        try:
            # If input is raw JSON string, parse it first
            if isinstance(json_obj, str):
                json_obj = json.loads(json_obj)

            # Extract and flatten JSON fields
            extracted_texts = extract_text_from_json(json_obj)
            joined_text = "\n".join(extracted_texts)

            for chunk in splitter.split_text(joined_text):
                texts.append(chunk)
                metadatas.append(metadata)

        except Exception as e:
            print(f"Error processing chunk: {e}")

    return texts, metadatas



@st.cache_resource
def get_vectorstore(text_chunks, metadatas):
    embeddings = OllamaEmbeddings(model="all-minilm")
    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadatas
    )
    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore


@st.cache_resource
def get_conversation_chain(_vectorstore):
    llm = ChatOllama(model="llama3", temperature=0.5)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False,
        output_key="answer"
    )


def transcribe(audio_path: str) -> str:
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except FileNotFoundError:
        st.error(
            "🚨 FFmpeg not found! Whisper uses FFmpeg under the hood.\n"
            "Please install FFmpeg and add its `bin/` folder to your PATH."
        )
        raise


def handle_userinput(user_question: str):
    # use .invoke() instead of deprecated __call__
    result = st.session_state.conversation.invoke({"question": user_question})
    answer = result["answer"]

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    st.session_state.qa_history.append((user_question, answer))

    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

    docs = st.session_state.conversation.retriever.get_relevant_documents(user_question)
    for doc in docs:
        meta = doc.metadata
        st.markdown(
            f"<div style='font-size: small; color: gray;'>"
            f"Source: {meta.get('source','Unknown')}"
            f"</div>",
            unsafe_allow_html=True
        )


def main():
    load_dotenv()
    st.set_page_config(page_title="Metro_ChatBot")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.title("IIT Kanpur – IT Helpdesk Assistant")
    st.markdown("Chat with uploaded JSON files. You can type or upload an audio file.")

    # — Text input & confirm button —
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("Enter your question:", key="user_input")
    with col2:
        text_confirm = st.button("Confirm Text Query")

    if text_confirm and user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # — Audio uploader & confirm button —
    uploaded_audio = st.file_uploader(
        "Upload an audio file (wav, mp3, m4a, aac, ogg):",
        type=["wav", "mp3", "m4a", "aac", "ogg"]
    )
    if uploaded_audio:
        st.audio(uploaded_audio)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_audio.getbuffer())
            temp_audio_path = f.name

        try:
            transcription = transcribe(temp_audio_path)
            st.success(f"Recognized: {transcription}")

            if st.button("Confirm Audio Query"):
                if st.session_state.conversation:
                    handle_userinput(transcription)
        except Exception as e:
            st.error(f"Error during transcription: {e}")

    # — Sidebar for JSON processing —
    with st.sidebar:
        st.header("Upload JSON Data Files")
        json_docs = st.file_uploader(
            "Upload one or more JSON files",
            accept_multiple_files=True,
            type="json"
        )

        if st.button("Process") and json_docs:
            with st.spinner("Processing JSON files..."):
                chunks_meta = load_json_chunks_with_metadata(json_docs)
                texts, metas = split_chunks_with_metadata(chunks_meta)
                vectorstore = get_vectorstore(texts, metas)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.qa_history = []
            st.success("Files processed successfully. You can now chat.")

        if "qa_history" in st.session_state and st.session_state.qa_history:
            st.divider()
            st.subheader("Chat History")
            for i, (q, a) in enumerate(reversed(st.session_state.qa_history), 1):
                with st.expander(f"Q{i}: {q}"):
                    st.markdown(a)


if __name__ == "__main__":
    main()
