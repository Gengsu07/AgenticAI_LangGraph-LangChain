# Machine Learning & AI Agent Development Learning Resume

This document summarizes the key concepts and practical skills learned in AI Agent development, focusing on the RAG (Retrieval-Augmented Generation) pipeline using LangChain.

## Table of Contents
1. [Data Ingestion](#data-ingestion)
2. [Data Splitting](#data-splitting)
3. [Embeddings](#embeddings)
4. [Vector Stores](#vector-stores)
5. [Retrieval](#retrieval)

## 1. Data Ingestion
**Concept**: The first step in any RAG pipeline is loading data from various sources into a format the LLM application can understand.

**Tools Used**: `LangChain Document Loaders`

**Key Techniques**:
- **PDF Loading**: Using `PyMuPDFLoader` to extract text and metadata from PDF files.
- **Text Loading**: Using `TextLoader` for plain text files (`.txt`, `.md`).
- **CSV Loading**: Using `CSVLoader` to load structured data, treating each row as a document.

**Example Code**:
```python
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader

# Load PDF
pdf_loader = PyMuPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

# Load Text
text_loader = TextLoader("notes.txt")
text_docs = text_loader.load()
```

## 2. Data Splitting
**Concept**: Large documents need to be broken down into smaller chunks to fit within the LLM's context window and to improve retrieval accuracy.

**Tools Used**: `RecursiveCharacterTextSplitter`, `CharacterTextSplitter`

**Key Techniques**:
- **Recursive Splitting**: Tries to split on a list of characters (e.g., `\n\n`, `\n`, ` `, ``) to keep semantically related text together. This is the recommended splitter for generic text.
- **Chunk Size & Overlap**: 
    - `chunk_size`: Maximum number of characters in a chunk.
    - `chunk_overlap`: Number of characters shared between adjacent chunks to maintain context.

**Example Code**:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(docs)
```

## 3. Embeddings
**Concept**: Converting text into vector representations (lists of floating-point numbers) so that semantic similarity can be calculated.

**Tools Used**: `OpenAIEmbeddings`, `OllamaEmbeddings`

**Key Techniques**:
- **OpenAI Models**: High-performance embeddings (e.g., `text-embedding-3-small`, `text-embedding-3-large`). usage requires an API key.
- **Ollama Models**: Running open-source embedding models locally (e.g., `nomic-embed-text`).
- **Methods**:
    - `embed_query(text)`: For a single query string.
    - `embed_documents(list_of_texts)`: For a list of documents.

**Example Code**:
```python
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# Initialize Embedding Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# OR
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector = embeddings.embed_query("What is AI?")
```

## 4. Vector Stores
**Concept**: specialized databases designed to store and search vector embeddings efficiently.

**Tools Used**: `Chroma`, `FAISS` (Facebook AI Similarity Search)

**Key Techniques**:
- **Chroma**: An AI-native open-source vector database. easy to set up and persist to disk.
- **FAISS**: A library for efficient similarity search of dense vectors, great for in-memory search.
- **Persistence**: Saving the vector store to disk (`persist_directory`) so it can be reloaded later without re-embedding the data.

**Example Code**:
```python
from langchain_community.vectorstores import Chroma, FAISS

# Create Vector Store from Documents
db = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

# Load from Disk
db_new = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
```

## 5. Retrieval
**Concept**: Finding the most relevant documents related to a user's query from the vector store.

**Key Techniques**:
- **Similarity Search**: Finds documents with embeddings closest to the query embedding (usually using Cosine Similarity or L2 distance).
- **Similarity Search with Score**: Returns documents along with a similarity score (lower score is better for L2 distance in FAISS).
- **Retriever Interface**: Converting the vector store into a `Retriever` object to easily integrate into LangChain chains.

**Example Code**:
```python
# Similarity Search
docs = db.similarity_search("Tell me about the transformer architecture")

# Retrieval with Score
docs_and_scores = db.similarity_search_with_score("query")

# As a Retriever
retriever = db.as_retriever()
relevant_docs = retriever.invoke("What is the main topic?")
```

---
*Created based on the analysis of course materials in Learning and Lecturer folders.*
