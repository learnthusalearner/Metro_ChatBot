# Metro ChatBot

A Streamlit-based AI chatbot for querying metro information (e.g., Kanpur Metro) with support for:

* Conversational RAG pipeline using LangChain + FAISS vector store
* Multi-language support (English/Hindi) with translation via Deep Translator
* Audio transcription with Whisper
* Microphone recording (10-second fixed capture)
* Text-to-speech playback using gTTS
* Web fallback search via DuckDuckGo when LLM cannot answer

---

## 🚀 Features

* **File Upload & PDF/Text Ingestion**: Upload `.txt`, `.json`, etc., for the bot to index.
* **Conversational Retrieval**: Maintains chat history and uses FAISS for context retrieval.
* **Automatic Translation**: All user inputs get translated to English; answers can be displayed in Hindi.
* **Mic Input**: One-click 10-second recording that saves to `audio_Q/`.
* **Audio Uploader**: Upload audio files (`.wav`, `.mp3`, `.m4a`, etc.) for transcription.
* **TTS Playback**: Answers spoken via gTTS in selected language.
* **Web Fallback**: If the bot responds with uncertainty, it fetches answer from the internet and labels source.
* **Download Chat History**: Export conversation as a `.txt` file.
* **Metro Route Viewer**: Visualize metro connections via pre-generated images.

---

## 📁 Repository Structure

```
Metro_ChatBot/
├── audio_Q/                # Recorded mic queries (WAV files)
├── vectorstore_index/      # FAISS index directory
├── img.py                  # Helper to load and display metro route images
├── htmltemplate.py         # CSS & HTML templates for chat UI
├── main.py                 # Streamlit application entrypoint
├── requirements.txt        # Python dependencies
├── README.md               # This file
```

---

## 🛠️ Prerequisites

* Python 3.9 or newer
* `git` installed

---

## 📥 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/learnthusalearner/Metro_ChatBot.git
   cd Metro_ChatBot
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux / macOS
   venv\Scripts\activate.bat   # Windows
   ```

3. **Install dependencies** using `requirements.txt`:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Pull LLaMA model** (if required by Ollama):

   ````bash
   ollama pull llama3:8b
   ```** (if required by Ollama):

   ```bash
   ollama pull llama3:8b
   ````

---

## 🚀 Running the App

```bash
streamlit run main.py
```

* Navigate to `http://localhost:8501` in your browser.
* Use the top dropdown to select the answer language (English/Hindi).
* Upload files, type questions, or record/upload audio.

---

## 🔧 Configuration

* **Vector Store**: Stored in `vectorstore_index/`. Delete to re-index.
* **Audio Outputs**: Mic recordings saved in `audio_Q/`. TTS responses saved as `response.mp3`.
* **Cache**: Streamlit caches models and data; clear cache with `streamlit cache clear`.

---

## 📝 Important Notes

* Whisper transcription can be resource-intensive; ensure sufficient CPU/RAM.
* DuckDuckGo fallback may return partial answers; always check source link.
* Google TTS free tier has rate limits; monitor usage.

---

## 🎉 Contributing

Feel free to open issues or pull requests. For major changes, please open an issue first to discuss.

---

## 📜 License

MIT License. See `LICENSE` for details.
