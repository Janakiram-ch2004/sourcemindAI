# SourceMindAI

SourceMindAI is a chatbot I built as my mini project. You can ask it normal questions, upload your own documents and ask questions about them, or ask things that need the internet. The best part is that it shows the **sources** for every answer, so you can check where the information came from.

## What it can do

- Chat normally (greetings, casual questions)
- Answer questions from your uploaded PDF, DOCX and TXT files
- Search the web for latest or factual information
- Tell the current weather of a city
- Do math calculations
- Show sources: clickable links for web results, and file name with page number for documents
- Keep multiple chats with history in the sidebar

## How it works

When I ask a question, the app first decides what kind of question it is:

1. **Casual question** -> the model just answers directly.
2. **Factual / weather / math question** -> the model uses a tool (web search, weather or calculator) and then answers.
3. **Document question** -> the app searches my uploaded files. It then checks whether the retrieved text is actually useful. If it is, the answer comes from the documents. If not, it searches the web instead.

I used LangGraph to connect all these steps together.

## Tech used

- Python
- Streamlit (UI)
- LangGraph and LangChain
- Groq API (qwen/qwen3.8-27b model)
- ChromaDB (to store document chunks)
- FastEmbed (embeddings)
- DDGS (web search)
- OpenWeatherMap API (weather)

## Project files

- `source_citition_frontend.py` - the Streamlit app (chat UI, sidebar, file upload)
- `source_citition_backend.py` - the LangGraph workflow, tools and router
- `rag.py` - loads documents, splits them into chunks and stores them in ChromaDB
- `requirements.txt` - libraries needed

## How to run it on your computer

1. Clone the repo
```
   git clone https://github.com/Janakiram-ch2004/sourcemindAI.git
   cd sourcemindAI
```
2. Create a virtual environment and install the libraries
```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```
3. Create a `.env` file and add your keys
```
   GROQ_API_KEY=your_groq_key
   OPENWEATHER_API_KEY=your_openweather_key
```
4. Run the app
```
   streamlit run source_citition_frontend.py
```

## Things I learned

- How to build a chatbot with multiple steps using LangGraph
- How RAG works: chunking, embeddings and a vector database
- How to fall back to web
