import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings, ChatOllama
import whisper
import json
from deep_translator import GoogleTranslator
from htmltemplate import css, bot_template, user_template,source_template

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

from img import display_connection_image, get_station_codes

import numpy as np
import time
from scipy.io.wavfile import write as scipy_write

#net search
from langchain.retrievers.web_research import WebResearchRetriever
from langchain import LLMChain, PromptTemplate
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchRun

from duckduckgo_search import DDGS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

VECTORSTORE_DIR = "vectorstore_index"

def search_web(query: str, max_results: int = 1):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                return results[0]['body'], results[0]['href']
    except Exception as e:
        print(f"Web search error: {e}")
    return None, None


def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

SECTION_SPLITTERS = [
    r"\n\s*(Route summary:|FAQs::|NETWORK/LINE:|Token Fare:|Amenities include:|Station Gates:|More Routes from.*?)\s*\n",  # common headers
    r"(?<=\n)(?=\d+\.\s)",  # numbered FAQ or list
    r"\n\n+",  # fallback to paragraph breaks
]

@st.cache_data
def split_chunks_with_metadata(chunks_with_metadata):
    texts, metadatas = [], []

    for text, metadata in chunks_with_metadata:
        try:
            joined_text = translate_to_english(text)
             
            # Combine section-based and recursive splitting
            chunks = split_by_sections(joined_text)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                length_function=len
            )

            for section in chunks:
                for chunk in splitter.split_text(section):
                    if chunk.strip():
                        texts.append(chunk.strip())
                        metadatas.append(metadata)
                        
        except Exception as e:
            print(f"Error processing chunk: {e}")

    return texts, metadatas


def split_by_sections(text):
    """Split text using patterns like headings or paragraph breaks."""
    
    for pattern in SECTION_SPLITTERS:
        # Try to split the text using the current pattern
        sections = re.split(pattern, text, flags=re.IGNORECASE)
        
        # If the split worked (more than one part), clean and return it
        if len(sections) > 1:
            cleaned_sections = []
            for section in sections:
                section = section.strip()
                if section:  # only keep non-empty sections
                    cleaned_sections.append(section)
            return cleaned_sections

    # If no patterns worked, return the whole text as one section
    return [text.strip()]



@st.cache_resource
def get_vectorstore(text_chunks, metadatas):
    embeddings = OllamaEmbeddings(model="all-minilm")

    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
            #used when u are "You're loading untrusted files (e.g. from a user upload or public source)."
        )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadatas
    )

    # Save in a structured directory
    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore

@st.cache_resource
def get_conversation_chain(_vectorstore):
    llm = ChatOllama(model="llama3:8b", temperature=0.5)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        #Controls whether the chatbot returns just the answer i.e false or the answer + source documents. ie true
        output_key="answer"
    )


def transcribe(audio_path: str) -> str:
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path)
        return translate_to_english(result["text"])
    except FileNotFoundError:
        st.error("🚨")
        raise


def search_web(query: str, max_results: int = 1):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                return results[0]['body'], results[0]['href']
    except Exception as e:
        print(f"Web search error: {e}")
    return None, None

def handle_userinput(user_question: str):
    original_question = user_question  # Save original user input

    if user_question.lower().endswith((".mp3", ".wav", ".m4a")):
        try:
            user_question = transcribe(user_question)
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return
    else:
        user_question = translate_to_english(user_question)

    with st.spinner("⏳ Thinking..."):
        countdown_placeholder = st.empty()
        for i in range(3, 0, -1):
            countdown_placeholder.markdown(f"⏳ Generating answer in **{i}** seconds...")
            time.sleep(1)
        countdown_placeholder.empty()

    result = st.session_state.conversation.invoke({"question": user_question})
    answer_en = result["answer"]

    is_web_fallback = False

    # 🔁 Web fallback
    if any(phrase in answer_en.lower() for phrase in ["i apologize", "i'm not sure", "don't know", "unable to find"]):
        web_answer, web_url = search_web(user_question)
        if web_answer:
            answer_en = f"🌐 *This answer is fetched from the internet as it's not in my local data.*\n\n{web_answer}"
            is_web_fallback = True

    selected_lang = st.session_state.get("selected_language", "English")

    if selected_lang == "Hindi":
        try:
            answer_translated = translate_to_hindi(answer_en)
        except Exception as e:
            st.warning(f"Hindi translation failed: {e}")
            answer_translated = answer_en
    else:
        answer_translated = answer_en

    # ✅ Display both
    st.write(user_template.replace("{{MSG}}", original_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer_translated), unsafe_allow_html=True)

    # ✅ Save displayed form to history
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    st.session_state.qa_history.append((original_question, answer_translated))

    # 🔊 Speak answer
    try:
        from gtts import gTTS
        import base64

        lang_code = "hi" if selected_lang == "Hindi" else "en"
        tts = gTTS(answer_translated, lang=lang_code)
        tts.save("audio_Q/response.mp3")

        with open("response.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")

    # ✅ Show source if local, else say "Internet"
    if not is_web_fallback and "source_documents" in result:
        docs = result["source_documents"]
        seen = set()
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in seen:
                seen.add(source)
                st.markdown(source_template.replace("{{SOURCE}}", source), unsafe_allow_html=True)
    elif is_web_fallback:
        st.markdown(source_template.replace("{{SOURCE}}", "Internet"), unsafe_allow_html=True)


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
        grouped_text = translate_to_english(grouped_text)
        return grouped_text, {"source": file.name }

    # elif filename.endswith(".json"):
    #     try:
    #         data = json.load(file)
    #         file.seek(0)
    #         text = json.dumps(data, indent=2)
    #     except json.JSONDecodeError:
    #         text = content
    # else:
    #     text = content

    # text = translate_to_english(text)
    # return text, {"source": file.name}



def translate_to_hindi(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='hi').translate(text)
    except Exception as e:
        print(f"Hindi translation error: {e}")
        return text


def save_audio_to_audio_Q(data: np.ndarray, samplerate: int = 16000) -> str:
    os.makedirs("audio_Q", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("audio_Q", f"mic_query_{timestamp}.wav")
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    scipy_write(filepath, samplerate, data)
    return filepath

def get_web_fallback():
    search_tool = DuckDuckGoSearchRun()
    
    prompt = PromptTemplate.from_template("""
    Based on the web results below, answer the user question truthfully:
    
    {context}
    
    Question: {question}
    Answer:
    """)
    
    llm_chain = LLMChain(
        llm=ChatOllama(model="llama3:8b"),
        prompt=prompt
    )
    
    retriever = WebResearchRetriever.from_llm_and_tools(
        llm_chain=llm_chain,
        tools=[search_tool],
        num_search_results=2
    )
    return retriever


def main():
    load_dotenv()
    st.set_page_config(page_title="Metro_ChatBot")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    st.title("Kanpur Metro ChatBot")
    st.markdown("Chat with uploaded files. You can type, upload an audio file, or record with your mic.")

        # Set default language in session state
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "English"

    # Language selector dropdown (global)
    st.session_state.selected_language = st.selectbox(
        "🔀 Select language for answer:",
        ["English", "Hindi"],
        key="language_selector"
    )


    # --- Chat Input Section ---
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        user_question = st.text_input("Enter your question:", key="user_input")
    with col2:
        text_confirm = st.button("Confirm Text Query")
    with col3:
        mic_trigger = st.button("🎤 Record Mic")

    if text_confirm and user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # ✅🎙️ MIC RECORDING SECTION (AUTO 10 SEC, INTEGRATED WITH COL3 BUTTON)
    if mic_trigger:
        fs = 16000
        duration = 10  # seconds

        st.info("🎙️ Listening... Please speak clearly into the mic.")

        import sounddevice as sd
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        audio_path = save_audio_to_audio_Q(recording, fs)
        st.audio(audio_path)
        st.info("🔍 Transcribing your query...")

        try:
            mic_transcription = transcribe(audio_path)
            st.success(f"📝 Recognized: {mic_transcription}")

            if st.button("✅ Use This Mic Query"):
                if st.session_state.conversation:
                    handle_userinput(mic_transcription)
        except Exception as e:
            st.error(f"❌ Mic transcription error: {e}")

    # --- File-based Audio Upload Section ---
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

    # --- Sidebar: File Upload and Chat History ---
    with st.sidebar:
        st.header("Upload Data Files")
        uploaded_files = st.file_uploader(
            "Upload one or more files (JSON, TXT, etc.)",
            accept_multiple_files=True,
            type=None
        )

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

            st.success("Files processed successfully. You can now chat.")

        if "qa_history" in st.session_state and st.session_state.qa_history:
            st.divider()
            st.subheader("Chat History")
            for i, (q, a) in enumerate(reversed(st.session_state.qa_history), 1):
                with st.expander(f"Q{i}: {q}"):
                    st.markdown(a)
            chat_history_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.qa_history])
            st.download_button(
                label="Download as .txt",
                data=chat_history_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )
        # --- Sidebar: Metro Route Image Viewer ---
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


