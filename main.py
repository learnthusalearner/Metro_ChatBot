import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # ✅ Community FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings, ChatOllama  # ✅ Ollama embedding and LLM
import whisper  # ✅ Local Whisper
import json
import re

# ✅ Local translation fallback or just pass-through
def translate_to_english(text: str) -> str:
    return text  # You can add local translation if needed

from htmltemplate import css, bot_template, user_template, source_template
from img import display_connection_image, get_station_codes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VECTORSTORE_DIR = "vectorstore_index"

@st.cache_data
def split_chunks_with_metadata(chunks_with_metadata):
    texts, metadatas = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    for text, metadata in chunks_with_metadata:
        sections = text.split("\n\n")
        for section in sections:
            for chunk in splitter.split_text(section):
                if chunk.strip():
                    texts.append(chunk.strip())
                    metadatas.append(metadata)
    return texts, metadatas


@st.cache_resource
def get_vectorstore(text_chunks, metadatas):
    embeddings = OllamaEmbeddings(model="all-minilm")  # ✅ Local embeddings
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
    llm = ChatOllama(model="llama3", temperature=0.5)  # ✅ Local Ollama LLM
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )


def transcribe(audio_path: str) -> str:
    model = whisper.load_model("medium")  # ✅ Local Whisper
    result = model.transcribe(audio_path)
    return result["text"]


def handle_userinput(user_question: str):
    if user_question.lower().endswith((".mp3", ".wav", ".m4a")):
        try:
            user_question = transcribe(user_question)
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return
    result = st.session_state.conversation.invoke({"question": user_question})
    answer = result["answer"]

    st.session_state.qa_history.append((user_question, answer))
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

    if "source_documents" in result:
        seen = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get('source', 'Unknown')
            if source not in seen:
                seen.add(source)
                st.markdown(source_template.replace("{{SOURCE}}", source), unsafe_allow_html=True)


def read_file_contents(file):
    try:
        content = file.read().decode("utf-8", errors="ignore")
        file.seek(0)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return "", {"source": file.name}

    return content, {"source": file.name}


def main():
    load_dotenv()
    st.set_page_config(page_title="Metro_ChatBot")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    st.title("Kanpur Metro ChatBot (Offline)")
    st.markdown("Chat with uploaded Metro files (text/audio).")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("Enter your question:")
    with col2:
        if st.button("Confirm Text Query") and user_question:
            handle_userinput(user_question)

    uploaded_audio = st.file_uploader("Upload audio:", type=["mp3", "wav", "m4a"])
    if uploaded_audio:
        st.audio(uploaded_audio)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_audio.getbuffer())
            audio_path = f.name
        if st.button("Confirm Audio Query"):
            transcription = transcribe(audio_path)
            handle_userinput(transcription)

    with st.sidebar:
        st.header("Upload Files")
        uploaded_files = st.file_uploader("Upload TXT files", accept_multiple_files=True, type=["txt"])

        if st.button("Process Files") and uploaded_files:
            with st.spinner("Processing..."):
                chunks_with_metadata = []
                for file in uploaded_files:
                    text, metadata = read_file_contents(file)
                    chunks_with_metadata.append((text, metadata))

                texts, metas = split_chunks_with_metadata(chunks_with_metadata)
                vectorstore = get_vectorstore(texts, metas)
                st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Ready to chat!")

        if st.session_state.qa_history:
            st.divider()
            st.subheader("History")
            for i, (q, a) in enumerate(reversed(st.session_state.qa_history), 1):
                with st.expander(f"Q{i}: {q}"):
                    st.markdown(a)

        st.divider()
        st.subheader("Route Image Viewer")
        station_codes = get_station_codes()
        code1 = st.selectbox("First station", list(station_codes.keys()), format_func=lambda x: station_codes[x])
        code2 = st.selectbox("Second station", list(station_codes.keys()), format_func=lambda x: station_codes[x])
        if st.button("Show Route Image"):
            if code1 == code2:
                st.error("Select different stations.")
            else:
                display_connection_image(code1, code2)


if __name__ == "__main__":
    main()
