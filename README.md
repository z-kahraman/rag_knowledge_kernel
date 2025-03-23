# 🧠 Knowledge Kernel - RAG System for Document Querying

![Knowledge Kernel Interface](./docs/images/arayuz.png)

## 🌟 Overview

Knowledge Kernel is an open-source Retrieval Augmented Generation (RAG) system that enables you to query your documents using artificial intelligence. This project demonstrates the power of combining vector databases with large language models to create an effective document question-answering system.

## 📋 Why We Built This

We created Knowledge Kernel to address several challenges in information retrieval:

1. **Information Overload**: Many organizations struggle with extracting relevant information from their large document repositories
2. **Accessibility**: Technical barriers often prevent users from utilizing advanced AI capabilities
3. **Flexibility**: Most existing solutions lock users into specific providers or models
4. **Language Support**: Many RAG systems lack robust multilingual capabilities (we support both English and Turkish)
5. **Local Deployment**: Privacy concerns often necessitate local processing of sensitive documents

Our goal was to create a user-friendly RAG system that could be easily deployed locally, work with multiple LLM providers, and deliver accurate answers from user documents.

## 🛠️ How We Built It

Knowledge Kernel is built on a modular architecture:

1. **Document Processing**: We use LangChain's document loaders to process PDF files, breaking them into manageable chunks
2. **Vector Embeddings**: Document chunks are converted to vector embeddings using models from OpenAI or Ollama
3. **Vector Database**: FAISS is used to store and efficiently retrieve these embeddings
4. **Query Processing**: User questions are processed through a RAG chain that:
   - Converts the question to a vector
   - Finds the most relevant document chunks
   - Sends these chunks along with the question to an LLM
   - Returns a contextualized answer with source references
5. **User Interface**: A clean Streamlit interface makes the system accessible to non-technical users
6. **Multilingual Support**: Complete localization system for English and Turkish

## 🔍 Key Features

- 📄 PDF document indexing to vector database
- 🔄 Flexible embedding model selection (OpenAI, Ollama)
- 🤖 Multiple LLM provider support (OpenAI, Ollama)
- 🌐 Multilingual interface and responses (English, Turkish)
- 📊 Collection management and statistics
- 🔍 Natural language querying with source citations
- 💻 Both web interface and command line options

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- For local models: [Ollama](https://ollama.ai/)
- For OpenAI models: An OpenAI API key

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/knowledge_kernel.git
cd knowledge_kernel
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

3. (Optional) For Ollama users, install and pull a model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3.2:latest
```

4. (Optional) For OpenAI users, set your API key:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Usage

#### Web Interface

1. Start the web application:
```bash
streamlit run app.py
```

2. Open your browser to http://localhost:8501

3. Use the interface to:
   - Upload PDF documents
   - Create and manage collections
   - Ask questions about your documents
   - View detailed statistics

#### Command Line

For those who prefer command line:

```bash
# Upload a document
python main.py load_pdf /path/to/document.pdf --collection my_collection

# Query your documents
python main.py query "What is discussed in the document?"
```

## 📊 Project Status

### ✅ Implemented Features
- [x] PDF document processing and vectorization
- [x] Document collection management
- [x] Natural language document querying
- [x] Automatic cleanup of temporary files
- [x] Support for both OpenAI and Ollama models
- [x] Multilingual interface (English and Turkish)
- [x] Statistics view for collections
- [x] Source citation for answers
- [x] Caching system for faster repeat queries

### 📝 Planned Features
- [ ] Support for additional document formats (DOCX, TXT, etc.)
- [ ] Advanced query techniques (HyDE, query transformation)
- [ ] User feedback mechanisms for answer improvement
- [ ] Collection backup and restore utilities
- [ ] Customizable chunking strategies
- [ ] Performance optimizations for larger document sets
- [ ] API endpoint for integration with other applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

Happy querying! 🚀 