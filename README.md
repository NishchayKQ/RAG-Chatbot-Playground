# Rag Chat

A minimal, local Retrieval-Augmented Generation (RAG) chat application built using LangChain and Streamlit. The application allows users to upload PDF files on-the-fly inside the chat interface, automatically structures and indexes the text, and answers context-specific questions using localized search data. 

Think of it as a bare-bones, lightweight alternative to NotebookLM: it parses, chunks, and does semantic search proximity

## Features

* **In-Chat Document Ingestion:** Upload PDF files directly into the Streamlit chat layout.
* **On-the-Fly Layout Parsing:** Uses modern `Unstructured` document partitioning to intelligently group sections by header titles before vectorization.
* **Dynamic Context Injection:** Leverages LangChain agent middleware to run vector store similarity lookups right before invoking the LLM, ensuring the system prompt contains the most up-to-date documentation relevant to the user query.
* **Real-time Streaming:** Token-by-token streaming UI response using Streamlit's native `st.write_stream`.

## Tech Stack

* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **Orchestration & Agents:** [LangChain Framework](https://github.com/langchain-ai/langchain)
* **Inference Pipeline:** [Groq Cloud Engine](https://groq.com/) (Running `llama-3.1-8b-instant`)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (Persistent local storage)
* **Embeddings:** HuggingFace `sentence-transformers/all-mpnet-base-v2`

## Example
below chat uses [this](https://engineering.purdue.edu/kak/compsec/NewLectures/Lecture4.pdf) source pdf
<img width="811" height="931" alt="Screenshot From 2026-06-14 12-11-48 (Edited)" src="https://github.com/user-attachments/assets/4dfcf52a-da44-4651-9a5e-a2a6d496bf23" />

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NishchayKQ/RAG-Chatbot-Playground
   ```

2. Set up a virtual environment:
   ```bash
   uv sync
   ```
3. export groq api keys
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```
4. run the app with:
   ```bash
   uv run streamlit run main.py
   ```

## Project Structure

```text
├── core/
│   ├── __init__.py
│   ├── model_config.py     # Groq initialization, agent middleware, and streaming generator
│   └── vector_store.py     # Chroma client setup and Unstructured PDF loader loop
├── data/
│   └── chroma_langchain_db # Persistent directory for local vector binaries
└──  main.py                 # Streamlit UI layout and session state management
```

## References
streamlit disable local network
> streamlit by default allows access to running applications over local network. see - https://github.com/Aider-AI/aider/issues/2153

run config references
> run config setup for PyCharm and other IDE's <br>
> https://discuss.streamlit.io/t/run-streamlit-from-pycharm/21624/3 <br>
> https://stackoverflow.com/questions/60172282/how-to-run-debug-a-streamlit-application-from-an-ide <br>
