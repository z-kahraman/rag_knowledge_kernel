"""
BilgiÇekirdeği QA Zinciri Modülü
----------------------------
Bu modül, kullanıcı sorguları ve vektör veritabanı arasında bağlantı sağlar.
LangChain RetrievalQA zincirini kullanır.
OpenAI veya Ollama LLM modelleri kullanılabilir.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import json
import time

from langchain_core.documents import Document
# from langchain_community.chains import RetrievalQA
# from langchain_community.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM

# Merkezi loglama sistemini kullan
from utils.logging_config import get_logger
logger = get_logger(__name__)

# LLM Modelleri için importlar
OPENAI_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI LLM yüklenemedi. 'pip install langchain-openai' komutunu çalıştırın.")

OLLAMA_AVAILABLE = False
try:
    # Güncellenmiş import yöntemi
    try:
        from langchain_ollama import OllamaLLM
        OLLAMA_AVAILABLE = True
    except ImportError:
        # Alternatif import yöntemi - eskisi
        try:
            from langchain_ollama import Ollama
            OLLAMA_AVAILABLE = True
        except (ImportError, TypeError):
            # Üçüncü yöntem - direkt tanımlama
            from langchain_community.llms import Ollama
            OLLAMA_AVAILABLE = True
except Exception as e:
    logger.warning(f"Ollama LLM yüklenemedi: {str(e)}")
    logger.warning("'pip install langchain-community langchain-ollama' komutunu çalıştırın.")

from langchain_core.retrievers import BaseRetriever
from vectorstore.vector_db import VectorDatabase

# Varsayılan sorgu şablonu
DEFAULT_QA_PROMPT = """
Aşağıdaki bağlam bilgisi verilmiştir. Bu bilgiyi kullanarak sorulan soruyu yanıtla.
Eğer yanıt bağlam içinde bulunmuyorsa, "Bu bilgi verilen dokümanlarda bulunmamaktadır." diye yanıt ver.
Cevabını, bağlamdan gelen bilgilerden oluştur ve kendi bilgilerinle destekleme.

Bağlam:
{context}

Soru: {question}

Yanıt:
"""

# Dummy LLM sınıfı tanımlama
class DummyLLM(LLM):
    """
    Hiçbir LLM yüklenemediğinde basit bir dummy LLM.
    """
    
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        logger.warning("DummyLLM kullanılıyor! Gerçek LLM modeli yüklenemedi.")
        return "Bu cevap bir dummy LLM tarafından oluşturuldu. LLM yüklenemediği için gerçek bir yanıt verilemiyor."
    
    @property
    def _llm_type(self) -> str:
        return "dummy_llm"

class RAGChain:
    """Retrieval Augmented Generation (RAG) Zinciri sınıfı."""
    
    def __init__(
        self, 
        vector_db: VectorDatabase,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        top_k: int = 4,
        base_url: str = "http://localhost:11434",
        custom_prompt: Optional[str] = None
    ):
        """
        RAG Zincirini başlatır.
        
        Args:
            vector_db: Sorguları yanıtlamak için kullanılacak vektör veritabanı
            provider: LLM sağlayıcı ("openai" veya "ollama")
            model_name: Kullanılacak LLM modeli
            temperature: Model yaratıcılık seviyesi (0: belirleyici, 1: yaratıcı)
            top_k: Sorgulanacak en iyi doküman sayısı
            base_url: Ollama API URL'i (Ollama kullanıldığında)
            custom_prompt: Özel sorgu şablonu (None ise varsayılan kullanılır)
        """
        self.vector_db = vector_db
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.base_url = base_url
        
        # LLM modelini başlat
        self.llm = self._initialize_llm()
        
        # Özel şablon veya varsayılan şablon
        prompt_template = custom_prompt or DEFAULT_QA_PROMPT
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # QA zincirini oluştur
        self.qa_chain = self._create_qa_chain()
    
    def _initialize_llm(self) -> LLM:
        """
        Yapılandırmaya göre uygun LLM modelini başlatır.
        
        Returns:
            LLM: LangChain LLM nesnesi
        """
        provider = self.provider.lower()
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI LLM kullanmak için 'pip install langchain-openai' komutunu çalıştırın.")
                return DummyLLM()
            
            logger.info(f"OpenAI LLM başlatılıyor: {self.model_name}")
            try:
                return ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature
                )
            except Exception as e:
                logger.error(f"OpenAI LLM başlatılamadı: {str(e)}")
                return DummyLLM()
            
        elif provider == "ollama":
            if not OLLAMA_AVAILABLE:
                logger.error("Ollama LLM kullanmak için 'pip install langchain-community langchain-ollama' komutunu çalıştırın.")
                return DummyLLM()
            
            logger.info(f"Ollama LLM başlatılıyor: {self.model_name}")
            try:
                # Güncel sınıf ismini kullan (OllamaLLM) veya geriye dönük uyumluluk için Ollama
                try:
                    return OllamaLLM(
                        model=self.model_name,
                        temperature=self.temperature,
                        base_url=self.base_url
                    )
                except NameError:
                    return Ollama(
                        model=self.model_name,
                        temperature=self.temperature,
                        base_url=self.base_url
                    )
            except Exception as e:
                logger.error(f"Ollama LLM başlatılamadı: {str(e)}")
                return DummyLLM()
            
        else:
            logger.error(f"Desteklenmeyen LLM sağlayıcısı: {provider}")
            return DummyLLM()
    
    def _create_qa_chain(self) -> RetrievalQA:
        """
        QA zincirini oluşturur.
        
        Returns:
            RetrievalQA: QA zinciri
        """
        # Vektör veritabanından retriever oluştur
        if self.vector_db.vector_store is None:
            raise ValueError("Vektör veritabanı henüz oluşturulmadı veya yüklenmedi.")
        
        # Koleksiyonun metadata bilgilerini kontrol et
        collection_name = self.vector_db.current_collection or "documents"
        try:
            import os
            import json
            metadata_path = os.path.join("./indices", collection_name, "metadata", "collection_info.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Metadata'dan embedding modelini ve boyutunu al
                embedding_type = metadata.get("embedding_type", "")
                embedding_model = metadata.get("embedding_model", "")
                embedding_dim = metadata.get("embedding_dimension", 0)
                logger.info(f"Koleksiyon metadata bilgileri: {embedding_type}, {embedding_model}, boyut: {embedding_dim}")
            else:
                logger.warning(f"Koleksiyon {collection_name} için metadata dosyası bulunamadı.")
                embedding_type = ""
                embedding_model = ""
                embedding_dim = 0
        except Exception as e:
            logger.error(f"Metadata okunamadı: {str(e)}")
            embedding_type = ""
            embedding_model = ""
            embedding_dim = 0
            
        # Ollama embedding kullanmak için bir embedding modeli yapılandır
        from embeddings.embedder import DocumentEmbedder, EmbeddingConfig
        
        # Koleksiyonla aynı embed modeli kullanmaya çalış
        if "Ollama" in embedding_type and embedding_model:
            # Orijinal koleksiyonun modeli varsa onu kullan
            logger.info(f"Koleksiyon vektör boyutuyla uyumluluk için {embedding_model} modeli kullanılıyor")
            embedding_config = EmbeddingConfig(
                provider="ollama", 
                ollama_model=embedding_model  # Koleksiyonda kullanılan modeli kullan
            )
        elif "OpenAI" in embedding_type and embedding_model:
            logger.info(f"Koleksiyon vektör boyutuyla uyumluluk için {embedding_model} modeli kullanılıyor")
            embedding_config = EmbeddingConfig(
                provider="openai", 
                openai_model=embedding_model
            )
        else:
            # Varsayılan - koleksiyon hakkında bilgi bulunamadıysa
            logger.info("Koleksiyon bilgisi bulunamadı, varsayılan llama3.2:latest kullanılıyor")
            embedding_config = EmbeddingConfig(
                provider="ollama", 
                ollama_model="llama3.2:latest"
            )
            
        embedder = DocumentEmbedder(config=embedding_config)
        embedding_model = embedder.get_embedding_model()
        
        # Mevcut vector store'un embedding modelini güncelle
        self.vector_db.vector_store.embedding_function = embedding_model
        
        # Retriever'ı oluştur
        retriever = self.vector_db.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        logger.info(f"RetrievalQA zinciri oluşturuluyor (top_k={self.top_k})...")
        
        # RetrievalQA zincirini oluştur
        try:
            # Güncel langchain yöntemi
            from langchain.chains.retrieval_qa.base import RetrievalQA
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt}
            )
            
            logger.info("RetrievalQA zinciri başarıyla oluşturuldu")
            return qa_chain
        except Exception as e:
            logger.error(f"QA Zinciri oluşturulurken hata: {str(e)}")
            import traceback
            logger.error(f"Hata ayrıntıları: {traceback.format_exc()}")
            raise
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Kullanıcı sorusunu yanıtlar.
        
        Args:
            question: Kullanıcı sorusu
            
        Returns:
            Dict: Yanıt ve kaynak dokümanları içeren sözlük
            
        Raises:
            ValueError: Vektör veritabanı henüz oluşturulmamışsa
        """
        if self.qa_chain is None:
            raise ValueError("QA zinciri henüz oluşturulmadı.")
        
        # Sorguyu önbellekte ara (aynı sorular için gereksiz hesaplama yapmamak adına)
        cache_key = f"{question}_{self.provider}_{self.model_name}_{self.temperature}_{self.top_k}"
        if hasattr(self, '_cache') and cache_key in self._cache:
            logger.info(f"Önbellekte bulunan yanıt döndürülüyor")
            return self._cache[cache_key]
        
        logger.info(f"Soru yanıtlanıyor: {question[:50]}..." if len(question) > 50 else f"Soru yanıtlanıyor: {question}")
        
        # Başlangıç zamanını kaydet
        start_time = time.time()
        
        # Soruyu yanıtla
        try:
            # Güncel langchain sürümünde .invoke() metodunu kullan
            try:
                result = self.qa_chain.invoke({"query": question})
            except AttributeError:
                # Eski sürümler için __call__ metodunu kullan
                result = self.qa_chain({"query": question})
                
            # İşlem süresini hesapla
            execution_time = time.time() - start_time    
            logger.info(f"Yanıt başarıyla oluşturuldu (süre: {execution_time:.2f}s)")
            
            # Anahtarları kontrol et ve tutarlı hale getir
            answer = result.get("result", result.get("answer", ""))
            
            # source_documents anahtarını kontrol et ve source_docs olarak standartlaştır
            source_documents = result.get("source_documents", result.get("source_docs", []))
            
            # Sonucu formatla
            formatted_result = {
                "question": question,
                "answer": answer,
                "source_docs": source_documents,
                "execution_time": execution_time
            }
            
            # Sonucu önbelleğe al
            if not hasattr(self, '_cache'):
                self._cache = {}
            self._cache[cache_key] = formatted_result
            
            return formatted_result
        except Exception as e:
            # İşlem süresini hesapla (hata durumunda da)
            execution_time = time.time() - start_time
            
            import traceback
            logger.error(f"Soru yanıtlanırken hata (süre: {execution_time:.2f}s): {str(e)}")
            logger.error(f"Hata ayrıntıları: {traceback.format_exc()}")
            
            # Basit bir fallback yanıt oluştur
            return {
                "question": question,
                "answer": f"Soru yanıtlanırken bir hata oluştu: {str(e)}",
                "source_docs": [],
                "execution_time": execution_time
            }
    
    def format_source_documents(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Kaynak dokümanları okunabilir formata dönüştürür.
        
        Args:
            source_documents: Kaynak dokümanların listesi
            
        Returns:
            List: Kaynak bilgilerini içeren sözlük listesi
        """
        formatted_sources = []
        
        for doc in source_documents:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            formatted_sources.append(source_info)
        
        return formatted_sources 