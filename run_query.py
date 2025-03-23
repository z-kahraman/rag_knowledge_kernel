"""
BilgiÇekirdeği Sorgu Yürütücü
---------------------------
Kullanıcı sorgularını yürütmek için gerekli fonksiyonları içerir.
"""

import os
import time
import functools
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain.schema import Document

# Proje modülleri
from vectorstore.vector_db import VectorDatabase
from qa.rag_chain import RAGChain
from embeddings.embedder import DocumentEmbedder, EmbeddingConfig

# Loglama
from utils.logging_config import get_logger
logger = get_logger(__name__)

# Önbellek mekanizması - koleksiyon bazlı önbellekler
_cache = {}  # { "collection_name": { "query_key": result } }
_rag_chains = {}  # { "collection_name": { "config_key": rag_chain } }

def run_query(
    query: str,
    embedding_provider: str = "ollama",
    embedding_model: str = "llama3.2:latest",
    llm_provider: str = "ollama",
    llm_model: str = "llama3.2:latest",
    temperature: float = 0.2,
    top_k: int = 3,
    collection_name: str = "documents",
    openai_api_key: Optional[str] = None,
    use_cache: bool = True,
    language: str = "tr"  # Varsayılan dil Türkçe
) -> Tuple[Optional[str], Optional[List[Document]]]:
    """
    Bir koleksiyona karşı bir sorgu yürütür ve sonuçları döndürür.
    
    Args:
        query: Yürütülecek sorgu
        embedding_provider: Embedding sağlayıcısı (ollama, openai, dummy)
        embedding_model: Kullanılacak embedding modeli
        llm_provider: LLM sağlayıcısı (ollama, openai)
        llm_model: Kullanılacak LLM modeli
        temperature: LLM sıcaklık değeri
        top_k: Sorgulanacak en iyi doküman sayısı
        collection_name: Sorgulanacak koleksiyon adı
        openai_api_key: OpenAI API anahtarı (eğer OpenAI kullanılıyorsa)
        use_cache: Önbellek kullanılsın mı?
        language: Yanıt dili ("tr" veya "en")
        
    Returns:
        Tuple[str, List[Document]]: Yanıt ve kaynak dokümanlar
    """
    start_time = time.time()
    
    # Koleksiyon önbelleğini kontrol et veya oluştur
    if collection_name not in _cache:
        _cache[collection_name] = {}
    
    if collection_name not in _rag_chains:
        _rag_chains[collection_name] = {}
    
    # Önbellek anahtarını oluştur (dil bilgisini de ekle)
    cache_key = f"{query}__{llm_provider}__{llm_model}__{top_k}__{temperature}__{language}"
    
    # Önbellekte arama yap (eğer etkinse) - SADECE bu koleksiyon için
    collection_cache = _cache[collection_name]
    if use_cache and cache_key in collection_cache:
        logger.info(f"[{collection_name}] Önbellekte bulunan sorgu yanıtı getiriliyor: '{query[:30]}...'")
        return collection_cache[cache_key]
    
    logger.info(f"[{collection_name}] Sorgu başlatılıyor: '{query[:50]}...'")
    
    try:
        # RAG zinciri için önbellek anahtarını oluştur (dil bilgisini de ekle)
        rag_chain_key = f"{llm_provider}__{llm_model}__{temperature}__{top_k}__{language}"
        collection_rag_chains = _rag_chains[collection_name]
        
        # RAG zincirini başlatma
        if rag_chain_key in collection_rag_chains:
            logger.info(f"[{collection_name}] Önbellekte bulunan RAG zinciri kullanılıyor")
            rag_chain = collection_rag_chains[rag_chain_key]
        else:
            # Vector veritabanını başlat ve koleksiyonu yükle
            vector_db = VectorDatabase()
            
            # Koleksiyonu kontrol et
            if not os.path.exists(os.path.join(vector_db.base_dir, collection_name)):
                logger.error(f"Koleksiyon bulunamadı: {collection_name}")
                return None, None
            
            # Koleksiyon metadata bilgilerini kontrol et
            metadata = vector_db.get_collection_metadata(collection_name)
            
            # Metadata'dan embedding bilgilerini al ve doğru embedding modelini kullan
            if metadata:
                db_embedding_type = metadata.get("embedding_type", "")
                db_embedding_model = metadata.get("embedding_model", "")
                
                if db_embedding_type and db_embedding_model:
                    logger.info(f"[{collection_name}] Koleksiyon embedding: {db_embedding_type}/{db_embedding_model}")
                    
                    # Otomatik olarak koleksiyonla aynı embedding model kullan
                    if "Ollama" in db_embedding_type:
                        embedding_provider = "ollama"
                        embedding_model = db_embedding_model
                    elif "OpenAI" in db_embedding_type:
                        embedding_provider = "openai"
                        embedding_model = db_embedding_model
            
            # Koleksiyonu yükle
            logger.info(f"[{collection_name}] Koleksiyon yükleniyor")
            vector_db.load_collection(collection_name)
            
            # RAG zincirini oluştur
            logger.info(f"[{collection_name}] RAG zinciri oluşturuluyor: {llm_provider}/{llm_model}")
            
            if llm_provider == "ollama":
                # Ollama için base_url'i belirle
                base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                
                rag_chain = RAGChain(
                    vector_db=vector_db,
                    provider="ollama",
                    model_name=llm_model,
                    temperature=temperature,
                    top_k=top_k,
                    base_url=base_url,
                    language=language  # Dil bilgisini aktarıyoruz
                )
            elif llm_provider == "openai":
                if not openai_api_key:
                    logger.error("OpenAI API anahtarı gerekli ancak sağlanmadı")
                    return None, None
                    
                # OpenAI API anahtarını ayarla
                os.environ["OPENAI_API_KEY"] = openai_api_key
                
                rag_chain = RAGChain(
                    vector_db=vector_db,
                    provider="openai",
                    model_name=llm_model,
                    temperature=temperature,
                    top_k=top_k,
                    language=language  # Dil bilgisini aktarıyoruz
                )
            else:
                logger.error(f"Desteklenmeyen LLM sağlayıcı: {llm_provider}")
                return None, None
                
            # RAG zincirini önbelleğe al - SADECE bu koleksiyon için
            collection_rag_chains[rag_chain_key] = rag_chain
        
        # Sorguyu yürüt
        logger.info(f"[{collection_name}] Sorgu yürütülüyor...")
        result = rag_chain.ask(query)
        
        # Sonuçları önbelleğe al - SADECE bu koleksiyon için
        cached_result = (result["answer"], result["source_docs"])
        if use_cache:
            collection_cache[cache_key] = cached_result
        
        # Sorgu süresini ölç ve logla
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"[{collection_name}] Sorgu tamamlandı, süre: {execution_time:.2f} saniye")
        
        return cached_result
        
    except Exception as e:
        logger.error(f"[{collection_name}] Sorgu sırasında hata: {str(e)}", exc_info=True)
        return None, None

# Tüm önbellekleri temizle
def clear_all_caches():
    """
    Tüm RAG zinciri ve sorgu önbelleklerini temizler.
    """
    global _rag_chains, _cache
    _rag_chains = {}
    _cache = {}
    logger.info("Tüm önbellekler temizlendi")

# Belirli bir koleksiyonun önbelleğini temizle
def clear_collection_cache(collection_name: str):
    """
    Belirli bir koleksiyonun RAG zinciri ve sorgu önbelleğini temizler.
    
    Args:
        collection_name: Önbelleği temizlenecek koleksiyon adı
    """
    global _rag_chains, _cache
    
    # Koleksiyon önbelleklerini temizle
    if collection_name in _rag_chains:
        del _rag_chains[collection_name]
        
    if collection_name in _cache:
        del _cache[collection_name]
        
    logger.info(f"[{collection_name}] Koleksiyon önbelleği temizlendi")

# RAG önbelleğini temizle - geriye uyumluluk için
def clear_rag_cache():
    """
    Tüm RAG zinciri önbelleğini temizler.
    """
    global _rag_chains
    _rag_chains = {}
    logger.info("RAG zinciri önbelleği temizlendi")

# Sorgu önbelleğini temizle - geriye uyumluluk için
def clear_query_cache():
    """
    Tüm sorgu önbelleğini temizler.
    """
    global _cache
    _cache = {}
    logger.info("Sorgu önbelleği temizlendi")

def get_available_ollama_models() -> List[str]:
    """
    Sisteme yüklü Ollama modellerini listeler.
    
    Returns:
        List[str]: Mevcut model adları listesi
    """
    try:
        import requests
        import json
        
        # Ollama API'ye istek gönder
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base_url}/api/tags")
        
        if response.status_code == 200:
            data = response.json()
            # Model adlarını listele
            models = [model["name"] for model in data.get("models", [])]
            return models
        else:
            logger.error(f"Ollama API yanıt hatası: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Ollama modelleri listelenirken hata: {str(e)}")
        return [] 