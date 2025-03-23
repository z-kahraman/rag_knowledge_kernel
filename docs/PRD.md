# Knowledge Kernel - Product Requirements Document

## 1. Introduction

### 1.1 Purpose

The Knowledge Kernel project aims to make users' personal or corporate documents (PDF, notes, CVs, etc.) queryable through artificial intelligence by uploading them to a vector database. The project will enable users to quickly access information in their documents by making natural language queries.

### 1.2 Scope

This PRD contains the necessary requirements, features, and technical details for the development of the Knowledge Kernel project. The document will serve as a guide during the product development process.

### 1.3 Motivation

In the modern world, individuals and organizations possess an increasing amount of digital documents. Accessing information within these documents, especially when there are numerous documents involved, can be difficult and time-consuming. This project will:

- Provide fast and accurate information retrieval across documents
- Enable users to use their knowledge bases more effectively
- Democratize information access with artificial intelligence technologies
- Facilitate personal or corporate knowledge management

## 2. Product Features

### 2.1 Core Features

#### 2.1.1 Document Loading and Processing
- [x] PDF document format support
- [ ] TXT, DOCX document format support
- [x] Dividing documents into meaningful chunks
- [x] Metadata extraction and storage

#### 2.1.2 Vectorization and Storage
- [x] Vectorizing document chunks (embedding)
- [x] Storing vectors in FAISS database
- [ ] Storing vectors in Pinecone database (optional)
- [x] Creating and managing vector indices

#### 2.1.3 Querying and Response Generation
- [x] Natural language querying interface
- [x] Identifying and extracting relevant document chunks
- [x] Response synthesis and source information provision

### 2.2 Advanced Features (Sprint 2+)

#### 2.2.1 User Interface
- [x] Web-based user interface (with Streamlit)
- [x] Visual interface for document uploading, viewing, and querying
- [x] User sessions and settings

#### 2.2.2 Advanced Processing Features
- [x] Multilingual support (Turkish and English)
- [ ] Text extraction from visual content (OCR)
- [ ] Text extraction from audio files

#### 2.2.3 Categorization and Organization
- [x] Organizing documents in collections
- [ ] Automatic document categorization
- [ ] Tagging and organization tools
- [ ] Suggesting related documents

## 3. Technical Requirements

### 3.1 Architecture

Knowledge Kernel will have a modular architecture and include the following main components:

1. [x] **Document Processing Module**: Loading, processing, and chunking documents
2. [x] **Vectorization Module**: Converting document chunks to vectors
3. [x] **Vector Database Module**: Storing and querying vectors
4. [x] **QA Chain Module**: Processing queries and generating responses
5. [x] **CLI/API Interface**: Interface for user interaction

### 3.2 Technology Stack

#### 3.2.1 Programming Language and Frameworks
- [x] Python 3.8+
- [x] LangChain (for RAG applications)
- [x] Streamlit (for UI)

#### 3.2.2 Artificial Intelligence and Vector Processing
- [x] OpenAI API support
- [x] Ollama (local LLM) support
- [x] OpenAI Embeddings support
- [x] Ollama Embeddings support
- [x] FAISS (local vector database)
- [ ] Pinecone (cloud-based) support (optional)

#### 3.2.3 Storage and Data Management
- [x] Local file system (for vector indices)
- [x] JSON (for metadata storage)

### 3.3 Security Requirements
- [x] Secure management of API keys (.env files)
- [x] File security for local storage
- [ ] Protection of sensitive information (in future phases)

## 4. Development Roadmap

### 4.1 Sprint 1: Core Infrastructure
- [x] Document loading and processing infrastructure
- [x] Vectorization system
- [x] Local FAISS database integration
- [x] Basic QA chain
- [x] Command-line interface

### 4.2 Sprint 2: User Experience and Advanced Features
- [x] Web-based user interface
- [x] Multiple document management
- [x] Adding metadata to documents
- [x] Advanced querying options
- [x] Multilingual support (Turkish and English)

### 4.3 Sprint 3: Integration and Optimization
- [ ] Categorization features
- [ ] Cloud-based vector database option
- [x] Performance optimizations
- [ ] Export/import features
- [x] Caching system for fast response delivery

## 5. Evaluation Criteria

### 5.1 Performance Metrics
- [x] Query response time (< 2 seconds target)
- [x] Response accuracy and relevance
- [x] Document processing speed

### 5.2 User Experience Metrics
- [x] Ease of use
- [x] Simplicity of setup process
- [x] Documentation quality

## 6. Constraints and Assumptions

### 6.1 Constraints
- Costs for API usage
- Performance limits for large document collections
- Language model and embedding capacity limitations

### 6.2 Assumptions
- Users will have access to valid API keys
- Documents will mostly contain text content
- Basic Python knowledge users are targeted (in the first phase)

## 7. Additional Information

### 7.1 References
- LangChain documentation
- OpenAI API documentation
- FAISS documentation
- RAG (Retrieval Augmented Generation) academic papers

### 7.2 Glossary
- **RAG**: Retrieval Augmented Generation
- **Embedding**: The process of converting documents to vectors
- **Chunking**: Dividing documents into meaningful parts
- **LLM**: Large Language Model 