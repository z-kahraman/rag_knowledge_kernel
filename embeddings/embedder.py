"""
BilgiÇekirdeği Embedding Modülü
----------------------------
Bu modül, doküman parçalarını vektörlere dönüştürür.
OpenAI, Ollama veya HuggingFace InstructorEmbedding kullanılabilir.
"""

import os
import logging
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel
from tqdm import tqdm
import numpy as np

# Temel LangChain sınıfları
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Merkezi loglama sistemini kullan
from utils.logging_config import get_logger
logger = get_logger(__name__)

# OpenAI Embeddings için import
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI embeddings yüklenemedi. 'pip install langchain-openai' komutunu çalıştırın.")
    OPENAI_AVAILABLE = False

# Ollama Embeddings için import - alternatif yöntem
OLLAMA_AVAILABLE = False
try:
    # Güncellenmiş import yöntemi 
    try:
        from langchain_ollama import OllamaEmbeddings
        OLLAMA_AVAILABLE = True
    except ImportError:
        # İkinci yöntem - eskisi
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            OLLAMA_AVAILABLE = True
        except (ImportError, TypeError):
            # Üçüncü yöntem - direkt tanımlama
            from langchain_community.embeddings.ollama import OllamaEmbeddings
            OLLAMA_AVAILABLE = True
except Exception as e:
    logger.warning(f"Ollama embeddings yüklenemedi: {str(e)}")
    logger.warning("'pip install langchain-community langchain-ollama' komutunu çalıştırın.")

# InstructorEmbedding için import
try:
    from langchain_community.embeddings import HuggingFaceInstructEmbeddings
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    logger.warning("InstructorEmbedding yüklenemedi. 'pip install InstructorEmbedding sentence-transformers' komutunu çalıştırın.")
    INSTRUCTOR_AVAILABLE = False

class EmbeddingConfig(BaseModel):
    """Embedding yapılandırma sınıfı."""
    provider: str = "openai"  # openai, ollama, instructor veya dummy
    
    # OpenAI Embedding ayarları
    openai_model: str = "text-embedding-ada-002"
    
    # Ollama Embedding ayarları
    ollama_model: str = "llama3.2:latest"
    ollama_base_url: str = "http://localhost:11434"
    
    # InstructorEmbedding ayarları
    instructor_model_name: str = "hkunlp/instructor-large"
    embedding_instruction: str = "Represent the document for retrieval: "  # InstructorEmbedding yönergesi

class DummyEmbeddings(Embeddings):
    """
    Hiçbir embedding provider yüklenemediğinde kullanılacak 
    dummy embeddings sınıfı. Gerçek bir uygulamada kullanmayın.
    """
    
    def __init__(self, dim: int = 4096):
        """Belirtilen boyutta dummy embedding oluşturur."""
        self.dim = dim
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Belirtilen boyutta sabit vektör döndürür."""
        logger.warning("DummyEmbeddings kullanılıyor! Gerçek embedding modeli yüklenemedi.")
        return [[0.1] * self.dim for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Belirtilen boyutta sabit vektör döndürür."""
        logger.warning("DummyEmbeddings kullanılıyor! Gerçek embedding modeli yüklenemedi.")
        return [0.1] * self.dim

class DocumentEmbedder:
    """Dokümanları vektörlere dönüştüren sınıf."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None, embedding_config: Optional[EmbeddingConfig] = None):
        """
        Embedding işleyicisini başlatır.
        
        Args:
            config: Embedding yapılandırması. None ise varsayılan ayarlar kullanılır.
            embedding_config: Geriye uyumluluk için eski parametre adı (config tercih edilir)
        """
        # Geriye dönük uyumluluk için hem config hem de embedding_config'i destekle
        self.config = config or embedding_config or EmbeddingConfig()
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> Embeddings:
        """
        Yapılandırmaya göre uygun embedding modelini başlatır.
        
        Returns:
            Embeddings: LangChain Embeddings nesnesi
        """
        provider = self.config.provider.lower()
        
        if provider == "openai":
            # OpenAI embedding modelini başlat
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI embedding kullanmak için 'pip install langchain-openai' komutunu çalıştırın.")
                return DummyEmbeddings()
            
            logger.info(f"OpenAI embeddings başlatılıyor: {self.config.openai_model}")
            return OpenAIEmbeddings(
                model=self.config.openai_model,
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )
            
        elif provider == "ollama":
            # Ollama embedding modelini başlat
            if not OLLAMA_AVAILABLE:
                logger.error("Ollama embedding kullanmak için 'pip install langchain-community langchain-ollama' komutunu çalıştırın.")
                return DummyEmbeddings()
            
            logger.info(f"Ollama embeddings başlatılıyor: {self.config.ollama_model}")
            try:
                return OllamaEmbeddings(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url
                )
            except Exception as e:
                logger.error(f"Ollama embeddings başlatılamadı: {str(e)}")
                return DummyEmbeddings()
            
        elif provider == "instructor":
            # InstructorEmbedding modelini başlat
            if not INSTRUCTOR_AVAILABLE:
                logger.error("InstructorEmbedding kullanmak için 'pip install InstructorEmbedding sentence-transformers' komutunu çalıştırın.")
                return DummyEmbeddings()
            
            logger.info(f"Instructor embeddings başlatılıyor: {self.config.instructor_model_name}")
            return HuggingFaceInstructEmbeddings(
                model_name=self.config.instructor_model_name,
                embed_instruction=self.config.embedding_instruction
            )
        elif provider == "dummy":
            # Dummy embedding modelini başlat
            logger.info("Dummy embeddings başlatılıyor (sadece vektör veritabanını yüklemek için)")
            return DummyEmbeddings()
        else:
            logger.error(f"Desteklenmeyen embedding sağlayıcı tipi: {provider}")
            return DummyEmbeddings()
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Doküman listesini vektörlere dönüştürür.
        
        Args:
            documents: Vektörleştirilecek dokümanların listesi
            
        Returns:
            List: Her doküman için {id, text, embedding, metadata} içeren sözlüklerin listesi
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        logger.info(f"{len(texts)} doküman parçası vektörleştiriliyor...")
        
        # Batch halinde vektörleştir
        embeddings = self.embeddings.embed_documents(texts)
        
        # Her doküman için sonuç formatını oluştur
        results = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            results.append({
                "id": f"doc_{i}",
                "text": text,
                "embedding": embedding,
                "metadata": metadata
            })
        
        logger.info(f"Vektörleştirme tamamlandı: {len(results)} doküman")
        return results
    
    def embed_query(self, query: str) -> List[float]:
        """
        Sorguyu vektöre dönüştürür.
        
        Args:
            query: Vektörleştirilecek sorgu metni
            
        Returns:
            List[float]: Sorgunun vektör temsili
        """
        return self.embeddings.embed_query(query) 
        
    def get_embedding_model(self) -> Embeddings:
        """
        Embeddings nesnesini döndürür.
        
        Returns:
            Embeddings: LangChain Embeddings sınıfı
        """
        return self.embeddings 