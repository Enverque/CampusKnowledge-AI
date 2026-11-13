# CampusKnowledge-AI
 using RAG

## Project Overview

**CampusKnowledge-AI** is an AI-powered Retrieval-Augmented Generation (RAG) application that allows students to ask questions about college-related information through an interactive **Streamlit chatbot**.

The system retrieves relevant information from a collection of college documents and uses a Large Language Model to generate answers based only on the retrieved document context.

The project is designed to answer questions related to:

* Admissions
* Fees
* Placements
* Hostel rules
* Attendance
* Examinations
* NIRF rankings
* Student handbooks
* Rules and regulations

The application also displays the **source document and page number** used to generate the answer, improving transparency and traceability.

---

## Key Features

### 1. Retrieval-Augmented Generation

The application follows a complete RAG pipeline:

```text
College PDF Documents
        ↓
PDF Loading
        ↓
Metadata Extraction
        ↓
Text Chunking
        ↓
Chunk Cleaning
        ↓
Hugging Face Embeddings
        ↓
FAISS Vector Store
        ↓
Hybrid Retrieval
        ↓
Relevant Context
        ↓
DeepSeek-V3
        ↓
Grounded Answer
        ↓
Streamlit Chat Interface
```

### 2. Hybrid Retrieval

Instead of relying only on semantic vector search, the project combines two retrieval methods:

```text
Semantic Search
       +
Keyword Search
       ↓
Hybrid Retrieval
```

#### Semantic Search

FAISS is used to retrieve semantically similar document chunks using vector embeddings.

#### Keyword Search

The system performs exact keyword matching to improve retrieval for information such as:

* College names
* Company names
* Regions
* Years
* Placement information
* Percentages
* Rankings
* Tables and statistics

The final retrieval combines semantic and keyword-based results.

---

## Document Metadata

The project assigns metadata to each document and page, including:

* Source file
* Category
* Year
* Document type
* Page number

### Supported Categories

```text
Admission
Fees
Placement
NIRF Ranking
Hostel
Student Handbook
Attendance
Examination
Rules & Regulations
Other
```

This metadata is preserved through the chunking and retrieval process.

---

## Document Processing

PDF documents are loaded using `PyPDFLoader`.

The project processes every PDF in the document directory and extracts page-level content.

Example:

```python
loader = PyPDFLoader(str(pdf_file))
pdf_docs = loader.load()
```

The extracted documents are then enriched with metadata before chunking.

---

## Text Chunking

The extracted document text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

```python
chunk_size = 800
chunk_overlap = 120
```

The splitter uses multiple separators to preserve meaningful text boundaries.

Very small chunks containing less than 50 characters are removed during cleaning.

---

## Embeddings

The project uses the following Sentence Transformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings convert text chunks into numerical vector representations for semantic retrieval.

---

## FAISS Vector Database

The generated embeddings are stored in a **FAISS vector store**.

FAISS enables fast similarity search over the indexed document chunks.

The application loads the stored vector database using:

```python
FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

## Large Language Model

The project uses **DeepSeek-V3** through Hugging Face.

```text
Model:
deepseek-ai/DeepSeek-V3
```

The model is accessed using:

* HuggingFaceEndpoint
* ChatHuggingFace

The temperature is set to `0.2` to encourage more focused and consistent responses.

The Hugging Face API token is loaded securely from an environment variable:

```python
os.getenv("HF_TOKEN")
```

---

## Grounded Answer Generation

The system is instructed to answer **only using the retrieved college document context**.

The prompt includes rules such as:

* Do not use outside knowledge.
* Do not invent information.
* Preserve numerical values accurately.
* Pay attention to specific regions, years, companies, percentages, fees, rankings, etc.
* Do not replace one region with another.
* Mention the source and page whenever possible.
* Clearly state when the required information cannot be found.

When information is unavailable, the system responds:

```text
I could not find this information in the college documents.
```

This helps reduce unsupported answers.

---

## Streamlit Application

The project includes an interactive Streamlit web application.

### Application Features

* Chat-based question answering
* Persistent conversation history
* Clear conversation button
* Suggested questions
* Cached vector store
* Cached language model
* Retrieved source documents
* Page-level source references
* Knowledge-base chunk count

### Example Questions

```text
What is the placement percentage?

What are the admission requirements?

What are the hostel rules?

What is the NIRF ranking?

What are the college fees?
```

---

## Source Transparency

For every generated response, the application displays relevant document sources.

Example:

```text
Sources

📄 placement_report.pdf — Page 12
📄 placement_policy.pdf — Page 4
```

This allows users to trace the answer back to the original document.

---

## Technology Stack

| Technology              | Purpose                  |
| ----------------------- | ------------------------ |
| Python                  | Application development  |
| LangChain               | RAG pipeline             |
| PyPDFLoader             | PDF document loading     |
| Hugging Face Embeddings | Text embeddings          |
| FAISS                   | Vector similarity search |
| DeepSeek-V3             | Answer generation        |
| Streamlit               | Web application          |
| Regular Expressions     | Keyword extraction       |
| NumPy                   | Data processing          |

---

## Project Structure

```text
CampusKnowledge-AI/
│
├── app/
│   ├── app.py
│   └── rag_backend.py
│
├── notebooks/
│   └── college_rag_pipeline.ipynb
│
├── college_faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── documents/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/CampusKnowledge-AI.git
```

### 2. Navigate to the project

```bash
cd CampusKnowledge-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

The application requires a Hugging Face API token.

Set it as an environment variable instead of writing the token directly in the source code.

### Windows Command Prompt

```bash
set HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

### PowerShell

```powershell
$env:HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

Never commit API keys or tokens to GitHub.

---

## Running the Application

From the project root:

```bash
streamlit run app/app.py
```

The Streamlit application will open in your browser.

---

## Creating the FAISS Index

The RAG notebook performs the document-processing pipeline:

```text
PDF Loading
↓
Metadata Detection
↓
Chunking
↓
Chunk Cleaning
↓
Embeddings
↓
FAISS Index
```

The generated FAISS index can then be loaded by the Streamlit application.

---

## Example User Flow

```text
Student asks a question
        ↓
Hybrid retrieval searches the knowledge base
        ↓
Relevant document chunks are collected
        ↓
Context is passed to DeepSeek-V3
        ↓
Answer is generated only from the context
        ↓
Streamlit displays the response
        ↓
Source document + page number are shown
```

---

## Knowledge Base

The knowledge base was created from a collection of college-related PDF documents covering topics such as:

* Admission
* Fees
* Placements
* Hostel
* Attendance
* Examinations
* NIRF rankings
* Student policies
* Rules and regulations

The original PDF collection is not included in this public repository.

---

## Important Security Note

The project uses a Hugging Face API token through the environment variable:

```text
HF_TOKEN
```

API keys, tokens, passwords, and other sensitive credentials should never be committed to the repository.

---

## Future Improvements

Possible future enhancements include:

* Metadata-aware filtered retrieval
* Reranking retrieved documents
* Conversational memory
* Multi-college selection
* Document upload directly from Streamlit
* Answer confidence scoring
* Improved table extraction
* Better citation formatting
* OCR support for scanned PDFs
* Deployment using Streamlit Cloud or another hosting platform

---

## Skills Demonstrated

* Generative AI
* Retrieval-Augmented Generation (RAG)
* Large Language Models
* LangChain
* FAISS
* Vector Databases
* Semantic Search
* Hybrid Search
* Natural Language Processing
* Prompt Engineering
* Hugging Face
* DeepSeek-V3
* Streamlit
* Python
* Document Processing
* Metadata Engineering
* Information Retrieval

---

## Author

**Himanshu Verma**

Aspiring Data Analyst | Data Scientist | AI/ML Enthusiast

GitHub: [Enverque](https://github.com/Enverque)
