# Patent Summariser

An ultra-fast, RAG-powered Streamlit web application designed to ingest complex patent documents (PDF/TXT) and generate structured, human-readable technical summaries in seconds. 

By leveraging **Google GenAI** for dense document embedding and **Groq's Llama 3.3** for lightning-fast text generation, this tool bypasses traditional API rate limits to deliver comprehensive patent analysis without the wait.

## Features

* **Intelligent Metadata Extraction:** Automatically parses the cover page to identify the Patent Title, Inventors, Assignee, Status, and Timeline.
* **Targeted RAG Pipeline:** Uses FAISS vector similarity search to accurately answer 6 highly specific queries about the patent:
  1. Simplified TL;DR
  2. Problem & Prior Art
  3. Core Invention Breakdown
  4. Independent Claims Analysis
  5. Key Technical Terms (Glossary)
  6. Novelty & Risk Flags
* **Dual-API Architecture:** Optimises cost and speed by using Google's Free Tier for heavy PDF embedding and Groq's Llama 3.3 for high-speed inference.
* **Built-in Resilience:** Includes a custom smart-retry wrapper to gracefully handle network blips and API rate limits.
* **One-Click Export:** Download the fully synthesised patent report as a clean `.txt` file.

## Tech Stack

* **Frontend & UI:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Store:** FAISS (In-memory)
* **Embeddings:** Google AI Studio (`models/gemini-embedding-001`)
* **LLM Generation:** Groq (`llama-3.3-70b-versatile`)
* **Document Parsing:** PyPDF

## Getting Started

Follow these steps to run the Patent Summariser locally on your machine.

### Prerequisites
You will need two free API keys to run this application:
1. **Google AI Studio API Key:** Get it [here](https://aistudio.google.com/).
2. **Groq API Key:** Get it [here](https://console.groq.com/).

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/innovasearch-ai.git](https://github.com/yourusername/innovasearch-ai.git)
   cd innovasearch-ai
