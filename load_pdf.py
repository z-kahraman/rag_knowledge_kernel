"""
BilgiÇekirdeği PDF Yükleme Modülü
-----------------------------
PDF dosyalarını yükleyip, bölümlere ayırarak vektör veritabanına kaydeder.
"""

import os
import time
from typing import Dict, Any, Optional, List, Union

# PDF işleme
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Proje modülleri
from vectorstore.vector_db import VectorDatabase
from embeddings.embedder import DocumentEmbedder, EmbeddingConfig

# Loglama
from utils.logging_config import get_logger
logger = get_logger(__name__)

def load_pdf_document(
    pdf_path: str,
    embedding_provider: str = "ollama",
    embedding_model: str = "llama3.2:latest",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = "documents",
    openai_api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    PDF dosyasını yükler, bölümlere ayırır ve vektör veritabanına kaydeder.
    
    Args:
        pdf_path: Yüklenecek PDF dosyasının yolu
        embedding_provider: Embedding sağlayıcısı (ollama, openai, dummy)
        embedding_model: Kullanılacak embedding modeli
        chunk_size: Bölüm boyutu (karakter sayısı)
        chunk_overlap: Örtüşme boyutu (karakter sayısı)
        collection_name: Dokümanın kaydedileceği koleksiyon adı
        openai_api_key: OpenAI API anahtarı (eğer OpenAI kullanılıyorsa)
        
    Returns:
        Dict: İşlem sonucu bilgileri (başarılı/başarısız, doküman sayısı vs.)
        
    Raises:
        Exception: PDF işleme sırasında hata oluşursa
    """
    start_time = time.time()
    logger.info(f"PDF yükleme işlemi başlatılıyor: {pdf_path}")
    
    try:
        # PDF dosyasının varlığını kontrol et
        if not os.path.exists(pdf_path):
            logger.error(f"PDF dosyası bulunamadı: {pdf_path}")
            return None
        
        # PDF boyutunu kontrol et
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        logger.info(f"PDF boyutu: {file_size_mb:.2f} MB")
        
        # PDF yükleyici
        logger.info(f"PDF yükleniyor: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Sayfa sayısını logla
        logger.info(f"PDF yüklendi, toplam sayfa sayısı: {len(pages)}")
        
        # Metin parçalayıcı
        logger.info(f"Doküman bölümlere ayrılıyor: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Dokümanları parçalara ayır
        chunks = text_splitter.split_documents(pages)
        logger.info(f"Doküman {len(chunks)} parçaya ayrıldı")
        
        # Embedding sağlayıcıyı hazırla
        logger.info(f"Embedding hazırlanıyor: {embedding_provider}/{embedding_model}")
        
        # OpenAI için API anahtarını ayarla
        if embedding_provider == "openai" and openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
        # Embedding konfigürasyonunu oluştur
        embedding_config = EmbeddingConfig(
            provider=embedding_provider,
            ollama_model=embedding_model if embedding_provider == "ollama" else "llama3.2:latest",
            openai_model=embedding_model if embedding_provider == "openai" else "text-embedding-ada-002",
            instructor_model_name=embedding_model if embedding_provider == "instructor" else "hkunlp/instructor-large"
        )
        
        # Embedding modeli oluştur
        embedder = DocumentEmbedder(config=embedding_config)
        
        # Vektör veritabanını başlat
        logger.info(f"Vektör veritabanı başlatılıyor: {collection_name}")
        vector_db = VectorDatabase()
        
        # Dokümanları vektör veritabanına ekle
        logger.info(f"Dokümanlar vektör veritabanına ekleniyor...")
        doc_ids = vector_db.add_documents(
            documents=chunks,
            embedder=embedder,
            collection_name=collection_name
        )
        
        # Dosya adını PDF'den al
        filename = os.path.basename(pdf_path)
        
        # İşlem süresini hesapla
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Sonuç bilgilerini oluştur
        result = {
            "success": True,
            "filename": filename,
            "file_size_mb": file_size_mb,
            "page_count": len(pages),
            "document_count": len(doc_ids),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "collection_name": collection_name,
            "execution_time": execution_time
        }
        
        logger.info(f"PDF başarıyla yüklendi: {filename}, {len(doc_ids)} bölüm, {execution_time:.2f} saniye")
        return result
        
    except Exception as e:
        logger.error(f"PDF yükleme hatası: {str(e)}", exc_info=True)
        raise 