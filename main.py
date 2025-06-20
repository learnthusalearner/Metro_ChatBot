import os
import tempfile
import streamlit as st
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from googletrans import Translator
from htmltemplate import css, bot_template, user_template, source_template
from img import display_connection_image, get_station_codes
import json
import re

VECTORSTORE_DIR = "/tmp/vectorstore"

# Translation
def translate_to_english(text: str) -> str:
    try:
        return Translator().translate(text, src='auto', dest='en').text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Section patterns
SECTION_SPLITTERS = [
    r"\n\s*(Route summary:|FAQs::|NETWORK/LINE:|Token Fare:|Amenities include:|Station Gates:|More Routes from.*?)\s*\n",
    r"(?<=\n)(?=\d+\.\s)", r"\n\n+"
]

@st.cache_data
def split_chunks_with_metadata(chunks_with_metadata):
    texts, metadatas = [], []
    for text, metadata in chunks_with_metadata:
        try:
            joined_text = translate_to_english(text)
            chunks = split_by_sections(joined_text)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            for section in chunks:
                for chunk in splitter.split_text(section):
                    if chunk.strip():
                        texts.append(chunk.strip())
                        metadatas.append(metadata)
        except Exception as e:
            print(f"Error processing chunk: {e}")
    return texts, metadatas

def split_by_sections(text):
    for pattern in SECTION_SPLITTERS:
        sections = re.split(pattern, text, flags=re.IGNORECASE)
        if len(sections) > 1:
            return [s.strip() for s in sections if s.strip()]
    return [text.strip()]

@st.cache_resource
def get_vectorstore(text_chunks, metadatas):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    vectorstore = FAISS.from_texts(text_chunks, embeddings, metadatas)
    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore

@st.cache_resource
def get_conversation_chain(_vectorstore):
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=_vectorstore.as_retriever(), memory=memory, return_source_documents=True, output_key="answer")

def transcribe(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            result = openai.Audio.transcribe("whisper-1", f)
            return translate_to_english(result["text"])
    except Exception as e:
        st.error(f"Transcription error: {e}")
        raise

def handle_userinput(user_question: str):
    if user_question.lower().endswith((".mp3", ".wav", ".m4a")):
        try:
            user_question = transcribe(user_question)
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return
    else:
        user_question = translate_to_english(user_question)

    result = st.session_state.conversation.invoke({"question": user_question})
    answer = result["answer"]

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    st.session_state.qa_history.append((user_question, answer))

    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

    if "source_documents" in result:
        seen = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                seen.add(source)
                st.markdown(source_template.replace("{{SOURCE}}", source), unsafe_allow_html=True)

def read_file_contents(file):
    filename = file.name.lower()
    try:
        content = file.read().decode("utf-8", errors="ignore")
        file.seek(0)
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return "", {"source": file.name}

    text_blocks = []
    if filename.endswith(".txt"):
        current_block = ""
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if any(marker in line for marker in [
                "Route summary", "Token Fare", "NETWORK/LINE", "First Metro",
                "Last Metro", "Station Gates", "Station Layout", "Train Frequency",
                "Fare", "Time Limit", "Travel Time", "Total Stations", "FAQs::"
            ]):
                if current_block:
                    text_blocks.append(current_block.strip())
                current_block = line + "\n"
            else:
                current_block += line + "\n"
        if current_block:
            text_blocks.append(current_block.strip())
        grouped_text = "\n\n".join(text_blocks)
        return translate_to_english(grouped_text), {"source": file.name}

def main():
    st.set_page_config(page_title="Metro ChatBot")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.title("Kanpur Metro ChatBot")
    st.markdown("Chat with uploaded files or ask questions via voice.")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("Enter your question:", key="user_input")
    with col2:
        text_confirm = st.button("Confirm Text Query")

    if text_confirm and user_question and st.session_state.conversation:
        handle_userinput(user_question)

    uploaded_audio = st.file_uploader("Upload an audio file (wav, mp3, m4a, aac, ogg):", type=["wav", "mp3", "m4a", "aac", "ogg"])
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

    with st.sidebar:
        st.header("Upload Data Files")
        uploaded_files = st.file_uploader("Upload one or more files (TXT only)", accept_multiple_files=True, type=["txt"])
        if st.button("Process") and uploaded_files:
            with st.spinner("Processing files..."):
                chunks_with_metadata = []
                for file in uploaded_files:
                    text, metadata = read_file_contents(file)
                    chunks_with_metadata.append((text, metadata))
                texts, metas = split_chunks_with_metadata(chunks_with_metadata)
                vectorstore = get_vectorstore(texts, metas)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.qa_history = []
            st.success("Files processed. You can chat now.")

        if "qa_history" in st.session_state and st.session_state.qa_history:
            st.divider()
            st.subheader("Chat History")
            for i, (q, a) in enumerate(reversed(st.session_state.qa_history), 1):
                with st.expander(f"Q{i}: {q}"):
                    st.markdown(a)

        st.divider()
        st.subheader("Metro Route Image Viewer")
        station_codes = get_station_codes()
        code1 = st.selectbox("First station", list(station_codes.keys()), format_func=lambda x: station_codes[x])
        code2 = st.selectbox("Second station", list(station_codes.keys()), format_func=lambda x: station_codes[x])
        if st.button("Show Route Image", key="show_route_image"):
            if code1 == code2:
                st.error("Please select two different stations.")
            else:
                display_connection_image(code1, code2)

if __name__ == "__main__":
    main()
