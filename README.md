# Docsense AI – RAG-Based Conversational AI System

> Multi-source intelligent document QA system powered by Retrieval-Augmented Generation (RAG)

---

## 📌 Introduction
Docsense AI is an advanced conversational AI system that enables users to interact with multiple data sources—including PDFs, web content, and YouTube transcripts—through natural language queries.

Unlike basic document chat applications, Docsense AI integrates hybrid retrieval mechanisms, multi-modal ingestion, and context-aware response generation to deliver accurate and relevant answers.

---

## 🚀 Key Features
- **Multi-Source Ingestion**  
  Supports PDFs, web URLs, and YouTube transcripts  

- **Retrieval-Augmented Generation (RAG)**  
  Combines semantic search with LLM-based reasoning  

- **Hybrid Retrieval System**  
  Vector search (Pinecone) + fallback web search  

- **Conversational Memory**  
  Maintains multi-turn context  

- **Adaptive Responses**  
  Tone control: Formal / Casual / Explanatory / Concise  

- **Modular Architecture**  
  Easily extensible for new data sources  

---

## 🧠 System Architecture

```
User Query
    ↓
Retriever (Vector Search - Pinecone)
    ↓
If insufficient context → Web Search Fallback
    ↓
Relevant Context Chunks
    ↓
LLM (OpenAI)
    ↓
Final Response (Context-Aware)
```

### Pipeline Steps:
1. Data ingestion (PDFs, URLs, transcripts)  
2. Text chunking  
3. Embedding generation  
4. Vector indexing (Pinecone)  
5. Hybrid retrieval  
6. Conversational chain with memory  
7. Response generation  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Frontend:** Streamlit  
- **LLM & Orchestration:** OpenAI, LangChain, LlamaIndex  
- **Vector Database:** Pinecone  
- **Concepts:** RAG, Semantic Search, Hybrid Retrieval  

---

## ⚙️ Installation

```bash
git clone <your-repo-link>
cd docsense-ai
pip install -r requirements.txt
```

Create a `.env` file and add:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Steps:
1. Open the app in browser  
2. Upload PDFs or enter URLs  
3. Ask questions  
4. Get context-aware answers  

---

## 🔥 Improvements Over Basic PDF Chat
- Supports multiple data sources  
- Hybrid retrieval for better accuracy  
- Web fallback for missing context  
- Conversational memory  
- Modular and scalable design  

---

## ⚠️ Limitations
- Depends on quality of retrieved data  
- Requires API keys (OpenAI, Pinecone)  
- Web fallback may increase latency  
