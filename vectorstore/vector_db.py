"""
BilgiÇekirdeği Vektör Veritabanı Modülü
------------------------------------
Bu modül, belge vektörlerini depolamak ve sorgulamak için FAISS veritabanını kullanır.
İstenirse Pinecone gibi bulut tabanlı çözümlere de geçiş yapılabilir.
"""

import os
import pickle
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time

import numpy as np
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

# Merkezi loglama sistemini kullan
from utils.logging_config import get_logger
logger = get_logger(__name__)

# FAISS Vektor veritabanını import etme girişimi
FAISS_AVAILABLE = False
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS veritabanı yüklenemedi. 'pip install langchain-community faiss-cpu' komutunu çalıştırın.")

from embeddings.embedder import DocumentEmbedder, EmbeddingConfig, DummyEmbeddings

class VectorDatabase:
    """
    Vektör veritabanı yönetimi için sınıf.
    FAISS kullanarak vektörleri saklar ve sorgular.
    """
    
    def __init__(self, base_dir: str = "./indices", embedding_model: Optional[Embeddings] = None):
        """
        Vektör veritabanını başlatır.
        
        Args:
            base_dir: Koleksiyonların kaydedileceği temel dizin
            embedding_model: Vektörleştirme için kullanılacak embedding modeli (opsiyonel)
        """
        self.base_dir = base_dir
        self.embedding_model = embedding_model
        self.vector_store = None
        self.current_collection = None  # Aktif koleksiyon adını takip et
        
        # Veritabanı dizinini oluştur (yoksa)
        os.makedirs(base_dir, exist_ok=True)
        
        # FAISS kullanılabilirliğini kontrol et
        if not FAISS_AVAILABLE:
            error_msg = "FAISS veritabanı yüklü değil. 'pip install langchain-community faiss-cpu' komutunu çalıştırın."
            logger.error(error_msg)
    
    def get_collection_metadata(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Koleksiyon metadata bilgilerini yükler.
        
        Args:
            collection_name: Bilgileri alınacak koleksiyon adı (belirtilmezse yüklü koleksiyon kullanılır)
        
        Returns:
            Dict: Metadata bilgileri veya boş sözlük
        """
        # Koleksiyon adını belirle
        collection = collection_name or self.current_collection
        if not collection:
            logger.warning("Henüz bir koleksiyon yüklenmedi")
            return {}
        
        try:
            metadata_file = os.path.join(self.base_dir, collection, "metadata", "collection_info.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Metadata yüklenirken hata: {str(e)}")
            return {}

    def save_collection_metadata(self, metadata: Dict[str, Any], collection_name: Optional[str] = None) -> bool:
        """
        Koleksiyon için metadata bilgilerini kaydeder.
        
        Args:
            metadata: Kaydedilecek metadata bilgileri
            collection_name: Koleksiyon adı (belirtilmezse yüklü koleksiyon kullanılır)
        
        Returns:
            bool: İşlem başarılıysa True, değilse False
        """
        # Koleksiyon adını belirle
        collection = collection_name or self.current_collection
        if not collection:
            logger.warning("Henüz bir koleksiyon yüklenmedi")
            return False
        
        try:
            metadata_dir = os.path.join(self.base_dir, collection, "metadata")
            os.makedirs(metadata_dir, exist_ok=True)
            
            metadata_file = os.path.join(metadata_dir, "collection_info.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Koleksiyon metadata kaydedildi: {collection}")
            return True
        except Exception as e:
            logger.error(f"Metadata kaydedilirken hata: {str(e)}")
            return False
    
    def add_documents(
        self, 
        documents: List[Document], 
        embedder: DocumentEmbedder,
        collection_name: str = "documents"
    ) -> List[str]:
        """
        Dokümanları veritabanına ekler. Koleksiyon yoksa oluşturur.
        
        Args:
            documents: Veritabanına eklenecek dokümanlar
            embedder: Dökümanları vektörleştirmek için embedding sağlayıcısı
            collection_name: Dökümanların ekleneceği koleksiyon adı
            
        Returns:
            List[str]: Eklenen dokümanların ID'leri
            
        Raises:
            ImportError: FAISS yüklü değilse
            RuntimeError: Dokümanlar eklenirken bir hata oluşursa
        """
        if not FAISS_AVAILABLE:
            error_msg = "FAISS veritabanı yüklü değil. 'pip install langchain-community faiss-cpu' komutunu çalıştırın."
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        try:
            # Embedding modelini al
            embedding_model = embedder.get_embedding_model()
            
            # Koleksiyon yolunu oluştur
            collection_path = os.path.join(self.base_dir, collection_name)
            
            # Koleksiyon zaten varsa yükle
            if os.path.exists(collection_path):
                logger.info(f"Var olan koleksiyon yükleniyor: {collection_name}")
                vector_store = FAISS.load_local(
                    collection_path, 
                    embedding_model,
                    allow_dangerous_deserialization=True  # Güvenli ortamda çalıştığımız için True
                )
                
                # Dokümanları ekle
                logger.info(f"Var olan koleksiyona {len(documents)} doküman ekleniyor")
                doc_ids = vector_store.add_documents(documents)
                
                # Güncellenen koleksiyonu kaydet
                vector_store.save_local(collection_path)
                
            else:
                # Yeni bir koleksiyon oluştur
                logger.info(f"Yeni koleksiyon oluşturuluyor: {collection_name}")
                vector_store = FAISS.from_documents(documents, embedding_model)
                
                # Koleksiyonu kaydet
                os.makedirs(collection_path, exist_ok=True)
                vector_store.save_local(collection_path)
                
                # Doküman ID'lerini dön (yeni koleksiyon için indeks numaralarını kullan)
                doc_ids = [str(i) for i in range(len(documents))]
            
            # Mevcut koleksiyonu güncelle
            self.vector_store = vector_store
            self.current_collection = collection_name
            
            # Embedding modeliyle ilgili metadata bilgilerini kaydet
            try:
                if hasattr(embedding_model, 'model_name'):
                    model_name = embedding_model.model_name
                elif hasattr(embedding_model, 'model'):
                    model_name = embedding_model.model
                else:
                    model_name = str(embedding_model.__class__.__name__)
                
                metadata = {
                    "embedding_type": str(embedding_model.__class__.__name__),
                    "embedding_model": model_name,
                    "embedding_dimension": len(documents[0].embedding) if hasattr(documents[0], 'embedding') else None,
                    "document_count": len(documents),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self.save_collection_metadata(metadata, collection_name)
            except Exception as e:
                logger.warning(f"Embedding metadata kaydedilirken hata: {str(e)}")
            
            logger.info(f"Toplam {len(doc_ids)} doküman başarıyla eklendi")
            return doc_ids
            
        except Exception as e:
            error_msg = f"Dokümanlar eklenirken hata oluştu: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_collection(self, collection_name: str = "documents") -> None:
        """
        Belirtilen koleksiyonu yükler.
        
        Args:
            collection_name: Yüklenecek koleksiyon adı
        
        Raises:
            FileNotFoundError: Belirtilen koleksiyon bulunamazsa
        """
        logger.info(f"Koleksiyon yükleniyor: {collection_name}")
        
        collection_path = os.path.join(self.base_dir, collection_name)
        if not os.path.exists(collection_path):
            error_msg = f"Koleksiyon bulunamadı: {collection_name}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            # FAISS indeksini yükle
            import faiss
            from langchain_community.vectorstores import FAISS
            
            # Doğru embedding modeli seçimi için metadata'yı kontrol et
            try:
                metadata = self.get_collection_metadata(collection_name)
                if metadata and "embedding_type" in metadata and "embedding_model" in metadata:
                    embedding_type = metadata["embedding_type"]
                    embedding_model = metadata["embedding_model"]
                    logger.info(f"Koleksiyon metadata'sı: {embedding_type}/{embedding_model}")
                    
                    # İstemci tarafında doğru embedding modeli oluşturulabilir
            except Exception as e:
                logger.warning(f"Metadata okuma hatası: {str(e)}")
            
            # Koleksiyonu yükle
            self.vector_store = FAISS.load_local(
                collection_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Akif koleksiyonu ayarla
            self.current_collection = collection_name
            
            doc_count = 0
            if hasattr(self.vector_store, "index") and self.vector_store.index is not None:
                doc_count = self.vector_store.index.ntotal
                
            logger.info(f"Koleksiyon başarıyla yüklendi: {doc_count} doküman")
            
        except Exception as e:
            error_msg = f"Koleksiyon yüklenirken hata: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Hata detayları: {traceback.format_exc()}")
            raise RuntimeError(error_msg)
    
    def similarity_search(self, query: str, k: int = 4, collection_name: Optional[str] = None) -> List[Document]:
        """
        Sorguya en benzer dokümanları bulur.
        
        Args:
            query: Sorgu metni
            k: Döndürülecek benzer doküman sayısı
            collection_name: Sorgulanacak koleksiyon adı (belirtilmezse yüklü koleksiyon kullanılır)
            
        Returns:
            Sorguya en benzer k adet doküman
            
        Raises:
            ValueError: Veritabanı henüz oluşturulmamışsa
        """
        # Belirtilen koleksiyon adı varsa, o koleksiyonu yükle
        if collection_name is not None and (self.current_collection != collection_name):
            self.load_collection(collection_name)
        
        # Vector store'un varlığını kontrol et
        if self.vector_store is None:
            error_msg = "Vektör veritabanı henüz yüklenmedi."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Sorguya en benzer dokümanları bul
            logger.info(f"Benzerlik sorgusu: {query}")
            similar_docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"{len(similar_docs)} benzer doküman bulundu")
            return similar_docs
            
        except Exception as e:
            error_msg = f"Benzerlik sorgusu sırasında hata: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def similarity_search_with_score(self, query: str, k: int = 4, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        Sorguya en benzer dokümanları benzerlik skorlarıyla birlikte döndürür.
        
        Args:
            query: Sorgu metni
            k: Döndürülecek benzer doküman sayısı
            collection_name: Sorgulanacak koleksiyon adı (belirtilmezse yüklü koleksiyon kullanılır)
            
        Returns:
            (doküman, benzerlik skoru) çiftlerinin listesi
            
        Raises:
            ValueError: Veritabanı henüz oluşturulmamışsa
        """
        # Belirtilen koleksiyon adı varsa, o koleksiyonu yükle
        if collection_name is not None and (self.current_collection != collection_name):
            self.load_collection(collection_name)
        
        # Vector store'un varlığını kontrol et
        if self.vector_store is None:
            error_msg = "Vektör veritabanı henüz yüklenmedi."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Sorguya en benzer dokümanları skorlarıyla birlikte bul
            logger.info(f"Skorlu benzerlik sorgusu: {query}")
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"{len(docs_and_scores)} benzer doküman bulundu")
            return docs_and_scores
            
        except Exception as e:
            error_msg = f"Skorlu benzerlik sorgusu sırasında hata: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def list_collections(self) -> List[str]:
        """
        Mevcut koleksiyonların listesini döndürür.
        
        Returns:
            List[str]: Koleksiyon adlarının listesi
        """
        try:
            # Temel dizindeki klasörleri listele
            collections = [d for d in os.listdir(self.base_dir) 
                          if os.path.isdir(os.path.join(self.base_dir, d))]
            return collections
            
        except Exception as e:
            logger.error(f"Koleksiyonlar listelenirken hata: {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Belirtilen koleksiyonu siler.
        
        Args:
            collection_name: Silinecek koleksiyon adı
            
        Returns:
            bool: İşlem başarılıysa True, değilse False
        """
        # Koleksiyon yolunu oluştur
        collection_path = os.path.join(self.base_dir, collection_name)
        
        # Koleksiyonun varlığını kontrol et
        if not os.path.exists(collection_path):
            logger.warning(f"Silinecek koleksiyon bulunamadı: {collection_name}")
            return False
        
        try:
            # Koleksiyon dizini ve içeriğini sil
            import shutil
            shutil.rmtree(collection_path)
            
            # Eğer silinen koleksiyon yüklüyse referansı temizle
            if self.current_collection == collection_name:
                self.vector_store = None
                self.current_collection = None
            
            logger.info(f"Koleksiyon silindi: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Koleksiyon silinirken hata: {str(e)}")
            return False
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Belirtilen koleksiyon hakkında istatistiksel bilgiler döndürür.
        
        Args:
            collection_name: İstatistikleri alınacak koleksiyon adı (belirtilmezse yüklü koleksiyon kullanılır)
            
        Returns:
            Dict: Koleksiyon istatistikleri (doküman sayısı vb.)
            
        Raises:
            ValueError: Koleksiyon yüklü değilse
        """
        # Belirtilen koleksiyon adı varsa, o koleksiyonu yükle
        if collection_name is not None and (self.current_collection != collection_name):
            self.load_collection(collection_name)
        
        # Yüklü koleksiyonu kontrol et
        if self.vector_store is None or self.current_collection is None:
            error_msg = "Henüz bir koleksiyon yüklenmedi."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Koleksiyon hakkında bilgi topla
            doc_count = len(self.vector_store.index_to_docstore_id)
            
            # Temel istatistikler
            stats = {
                "collection_name": self.current_collection,
                "document_count": doc_count,
                "storage_type": "FAISS",
                "index_path": os.path.join(self.base_dir, self.current_collection)
            }
            
            # Metadata bilgilerini de ekle
            metadata = self.get_collection_metadata()
            if metadata:
                stats["metadata"] = metadata
            
            return stats
            
        except Exception as e:
            logger.error(f"Koleksiyon istatistikleri alınırken hata: {str(e)}")
            raise RuntimeError(f"İstatistikler alınırken hata: {str(e)}") 