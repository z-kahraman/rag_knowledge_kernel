"""
PDF Dokümanlarını Yükleme ve İşleme Modülü
------------------------------
Bu modül, PDF dokümanlarını yükler ve vektor veritabanına eklemek için hazırlar.
Metin bölümleri, paragraflar veya sayfalar halinde ayrıştırabilir.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Merkezi loglama sistemini kullan
from utils.logging_config import get_logger
logger = get_logger(__name__)

# LangChain bileşenlerini import etme
LANGCHAIN_PYPDF_AVAILABLE = False
try:
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_PYPDF_AVAILABLE = True
except ImportError:
    logger.warning("PyPDFLoader yüklenemedi. 'pip install langchain-community' komutunu çalıştırın.")

LANGCHAIN_TEXT_SPLITTER_AVAILABLE = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_TEXT_SPLITTER_AVAILABLE = True
except ImportError:
    logger.warning("RecursiveCharacterTextSplitter yüklenemedi. 'pip install langchain' komutunu çalıştırın.")

from embeddings.embedder import DocumentEmbedder, EmbeddingConfig
from vectorstore.vector_db import VectorDatabase

def load_pdf(
    filepath: Union[str, Path],
    vector_db: Optional[VectorDatabase] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_provider: str = "openai",
    model_name: Optional[str] = None,
    collection_name: str = "documents"
) -> Dict[str, Any]:
    """
    PDF dosyasını yükler, belgelere böler ve isteğe bağlı olarak vektor veritabanına ekler.
    
    Args:
        filepath: PDF dosyasının yolu
        vector_db: Vektor veritabanı nesnesi (isteğe bağlı)
        chunk_size: Metin bölümü boyutu (karakter olarak)
        chunk_overlap: Metin bölümleri arasındaki örtüşme miktarı
        embedding_provider: Gömme sağlayıcısı ('openai', 'ollama', 'instructor')
        model_name: Belirli bir model adı (isteğe bağlı)
        collection_name: Vektor veritabanındaki koleksiyon adı
        
    Returns:
        Dict: İşleme sonuçlarını içeren bir sözlük
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        ImportError: Gerekli bağımlılıklar yüklü değilse
    """
    # Dosya kontrolü
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    if not filepath.exists():
        error_msg = f"PDF dosyası bulunamadı: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # LangChain bağımlılıklarını kontrol et
    if not LANGCHAIN_PYPDF_AVAILABLE:
        error_msg = "PyPDFLoader yüklü değil. 'pip install langchain-community' komutunu çalıştırın."
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    if not LANGCHAIN_TEXT_SPLITTER_AVAILABLE:
        error_msg = "RecursiveCharacterTextSplitter yüklü değil. 'pip install langchain' komutunu çalıştırın."
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    logger.info(f"PDF dosyası yükleniyor: {filepath}")
    
    # PDF dosyasını yükle ve sayfalara ayır
    try:
        loader = PyPDFLoader(str(filepath))
        pages = loader.load()
        logger.info(f"{len(pages)} sayfa yüklendi: {filepath}")
    except Exception as e:
        error_msg = f"PDF yüklenirken hata oluştu: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Metadata ekle
    for page in pages:
        # Orijinal metadata'yı koru ve yeni alanlar ekle
        page.metadata.update({
            "source": str(filepath),
            "filename": filepath.name,
            "filetype": "pdf",
            "page": page.metadata.get("page", 0)
        })
    
    # Metni bölümlere ayır
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"Metin bölümlere ayrılıyor (chunk_size={chunk_size}, overlap={chunk_overlap})")
        docs = text_splitter.split_documents(pages)
        logger.info(f"{len(docs)} metin bölümü oluşturuldu")
    except Exception as e:
        error_msg = f"Metin bölümlere ayrılırken hata: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Vektor veritabanına ekleme
    doc_count = len(docs)
    if vector_db is not None:
        logger.info(f"Dökümanlar {collection_name} koleksiyonuna ekleniyor...")
        
        try:
            # Embedding modelini başlat
            
            # Embedding konfigürasyonu oluştur
            embedding_config = EmbeddingConfig(
                provider=embedding_provider
            )
            
            # Model adı belirtilmişse, konfigürasyona ekle
            if model_name:
                if embedding_provider == "openai":
                    embedding_config.openai_model = model_name
                elif embedding_provider == "ollama":
                    # Eğer modelde ':latest' yoksa ekle
                    if ':latest' not in model_name and model_name == 'llama3.2':
                        model_name = 'llama3.2:latest'
                    embedding_config.ollama_model = model_name
                elif embedding_provider == "instructor":
                    embedding_config.instructor_model_name = model_name
            
            # Embedding sınıfını başlat
            embedder = DocumentEmbedder(config=embedding_config)
            
            # Vektör veritabanına ekle
            ids = vector_db.add_documents(
                documents=docs,
                embedder=embedder,
                collection_name=collection_name
            )
            
            logger.info(f"{len(ids)} döküman başarıyla eklendi")
        except Exception as e:
            error_msg = f"Vektör veritabanına eklerken hata: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Sonuç sözlüğü
    result = {
        "status": "success",
        "filename": filepath.name,
        "document_count": doc_count,
        "collection_name": collection_name if vector_db else None,
        "chunks": docs
    }
    
    return result 