#!/usr/bin/env python3
"""
BilgiÇekirdeği - Kişisel Dokümanları Yapay Zeka ile Sorgulama Sistemi
--------------------------------------------------------------------
Bu program, PDF, TXT ve DOCX formatlarındaki belgeleri vektör veritabanına 
yükleyerek doğal dil sorguları ile bilgiye ulaşmanızı sağlar.

OpenAI API veya yerel Ollama modelleri kullanabilirsiniz.

Kullanım:
  python main.py load --path "./data/belgelerim.pdf"   # Belge yükle
  python main.py ask "Bu belgelerdeki önemli tarihler nelerdir?"  # Soru sor
  python main.py info  # Veritabanı bilgilerini görüntüle
"""

import os
import sys
import argparse
import json
import dotenv
from typing import List, Dict, Any, Optional, Union
from colorama import Fore, Style, init
import logging
from pathlib import Path

# Loglama yapılandırmasını içe aktar
from utils.logging_config import setup_logging, get_logger

# .env dosyasını yükle
dotenv.load_dotenv()

# Gerekli modüller
from loader.document_loader import DocumentLoader
from embeddings.embedder import DocumentEmbedder, EmbeddingConfig
from vectorstore.vector_db import VectorDatabase
from qa.rag_chain import RAGChain
from ingestion.load_pdf import load_pdf

# Colorama'yı başlat
init()

# Varsayılan dizinler ve değerler
DEFAULT_INDEX_DIR = "./indices"
DEFAULT_DATA_DIR = "./data"

# LLM Ayarları
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding Ayarları
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
INSTRUCTOR_MODEL_NAME = os.getenv("INSTRUCTOR_MODEL_NAME", "hkunlp/instructor-large")

# Chunk Ayarları
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Ana modül logger'ı
setup_logging()  # Varsayılan ayarlarla loglama yapılandırması
logger = get_logger(__name__)

class BilgiCekirdegi:
    """Ana program sınıfı."""
    
    def __init__(self):
        """Program bileşenlerini başlatır."""
        self.document_loader = DocumentLoader(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Embedding ayarlarını yükle
        embedding_config = self._get_embedding_config()
        self.embedder = DocumentEmbedder(config=embedding_config)
        
        # Vektör veritabanını başlat
        self.vector_db = VectorDatabase(embedding_model=self.embedder.embeddings)
        
        # Varsa önceki indeksi yükle
        self._try_load_index()
        
        # RAG zincirini başlat
        self.rag_chain = None  # Başlangıçta None, veritabanı yüklenince oluşacak
        
    def _get_embedding_config(self) -> EmbeddingConfig:
        """
        Çevre değişkenlerine göre uygun embedding konfigürasyonunu oluşturur.
        
        Returns:
            EmbeddingConfig: Embedding yapılandırması
        """
        if EMBEDDING_PROVIDER == "openai":
            return EmbeddingConfig(
                provider="openai",
                openai_model=OPENAI_EMBEDDING_MODEL
            )
        elif EMBEDDING_PROVIDER == "ollama":
            return EmbeddingConfig(
                provider="ollama",
                ollama_model=OLLAMA_EMBEDDING_MODEL,
                ollama_base_url=OLLAMA_BASE_URL
            )
        elif EMBEDDING_PROVIDER == "instructor":
            return EmbeddingConfig(
                provider="instructor",
                instructor_model_name=INSTRUCTOR_MODEL_NAME
            )
        else:
            print(f"{Fore.YELLOW}Uyarı: Desteklenmeyen embedding sağlayıcısı '{EMBEDDING_PROVIDER}'. 'openai' kullanılıyor.{Style.RESET_ALL}")
            return EmbeddingConfig(provider="openai")
        
    def _try_load_index(self) -> None:
        """
        Varsa önceki indeksi yüklemeyi dener.
        """
        try:
            self.vector_db.load(directory=DEFAULT_INDEX_DIR)
            print(f"{Fore.GREEN}✓ Önceki indeks başarıyla yüklendi.{Style.RESET_ALL}")
        except FileNotFoundError:
            print(f"{Fore.YELLOW}ℹ Önceki indeks bulunamadı. Yeni dokümanlar yüklendiğinde oluşturulacak.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ İndeks yüklenirken hata oluştu: {str(e)}{Style.RESET_ALL}")
    
    def load_document(self, file_path: str) -> None:
        """
        Dokümanı yükler, işler ve veritabanına ekler.
        
        Args:
            file_path: Yüklenecek doküman dosyasının yolu
        """
        try:
            # Dosyayı yükle ve parçalara böl
            print(f"{Fore.BLUE}→ Doküman yükleniyor: {file_path}{Style.RESET_ALL}")
            documents = self.document_loader.load_document(file_path)
            
            # Doküman parçalarını vektör veritabanına ekle
            print(f"{Fore.BLUE}→ Doküman vektörleştiriliyor ve indeksleniyor...{Style.RESET_ALL}")
            self.vector_db.add_documents(documents)
            
            # Vektör veritabanını kaydet
            self.vector_db.save(directory=DEFAULT_INDEX_DIR)
            print(f"{Fore.GREEN}✓ Doküman başarıyla işlendi ve indekslendi.{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Doküman yüklenirken hata oluştu: {str(e)}{Style.RESET_ALL}")
    
    def load_documents_from_directory(self, directory_path: str) -> None:
        """
        Dizindeki tüm dokümanları yükler.
        
        Args:
            directory_path: Dokümanların bulunduğu klasör yolu
        """
        try:
            # Dizindeki tüm dokümanları yükle
            print(f"{Fore.BLUE}→ Dizindeki dokümanlar yükleniyor: {directory_path}{Style.RESET_ALL}")
            documents = self.document_loader.load_documents_from_directory(directory_path)
            
            if not documents:
                print(f"{Fore.YELLOW}ℹ Dizinde hiç desteklenen doküman bulunamadı.{Style.RESET_ALL}")
                return
            
            # Dokümanları vektör veritabanına ekle
            print(f"{Fore.BLUE}→ {len(documents)} doküman parçası vektörleştiriliyor ve indeksleniyor...{Style.RESET_ALL}")
            self.vector_db.add_documents(documents)
            
            # Vektör veritabanını kaydet
            self.vector_db.save(directory=DEFAULT_INDEX_DIR)
            print(f"{Fore.GREEN}✓ Toplam {len(documents)} doküman parçası başarıyla işlendi ve indekslendi.{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Dokümanlar yüklenirken hata oluştu: {str(e)}{Style.RESET_ALL}")
    
    def ask_question(self, question: str) -> None:
        """
        Sisteme soru sorar ve yanıtı gösterir.
        
        Args:
            question: Kullanıcı sorusu
        """
        try:
            # Veritabanı var mı kontrol et
            if self.vector_db.vector_store is None:
                print(f"{Fore.RED}✗ Henüz hiç doküman yüklenmemiş.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}ℹ Önce 'python main.py load --path <doküman_yolu>' komutunu kullanın.{Style.RESET_ALL}")
                return
            
            # RAG zincirini başlat (ilk kullanımda)
            if self.rag_chain is None:
                if LLM_PROVIDER == "openai":
                    self.rag_chain = RAGChain(
                        vector_db=self.vector_db,
                        provider="openai",
                        model_name=OPENAI_MODEL
                    )
                elif LLM_PROVIDER == "ollama":
                    self.rag_chain = RAGChain(
                        vector_db=self.vector_db,
                        provider="ollama",
                        model_name=OLLAMA_MODEL,
                        base_url=OLLAMA_BASE_URL
                    )
                else:
                    print(f"{Fore.YELLOW}Uyarı: Desteklenmeyen LLM sağlayıcısı '{LLM_PROVIDER}'. 'openai' kullanılıyor.{Style.RESET_ALL}")
                    self.rag_chain = RAGChain(
                        vector_db=self.vector_db,
                        provider="openai",
                        model_name=OPENAI_MODEL
                    )
            
            # Soruyu yanıtla
            print(f"{Fore.BLUE}→ Sorgunuz yanıtlanıyor: {question}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}→ Bu biraz zaman alabilir...{Style.RESET_ALL}")
            
            result = self.rag_chain.ask(question)
            
            # Yanıtı görüntüle
            print(f"\n{Fore.GREEN}Soru: {Style.RESET_ALL}{question}")
            print(f"{Fore.GREEN}Yanıt: {Style.RESET_ALL}{result['answer']}\n")
            
            # Kaynakları görüntüle
            print(f"{Fore.YELLOW}Kaynaklar:{Style.RESET_ALL}")
            for i, doc in enumerate(result['source_documents']):
                print(f"{Fore.YELLOW}[{i+1}]{Style.RESET_ALL} {doc.metadata.get('file_name', 'Bilinmeyen Kaynak')}")
                print(f"    {doc.page_content[:150]}..." if len(doc.page_content) > 150 else f"    {doc.page_content}")
                print()
            
        except Exception as e:
            print(f"{Fore.RED}✗ Soru yanıtlanırken hata oluştu: {str(e)}{Style.RESET_ALL}")
    
    def show_info(self) -> None:
        """
        Sistem hakkında bilgileri görüntüler.
        """
        try:
            print(f"\n{Fore.BLUE}BilgiÇekirdeği Sistem Bilgisi:{Style.RESET_ALL}")
            print(f"{Fore.BLUE}───────────────────────────{Style.RESET_ALL}")
            
            # LLM Bilgileri
            print(f"LLM Sağlayıcı: {LLM_PROVIDER}")
            if LLM_PROVIDER == "openai":
                print(f"LLM Modeli: {OPENAI_MODEL}")
            elif LLM_PROVIDER == "ollama":
                print(f"LLM Modeli: {OLLAMA_MODEL}")
                print(f"Ollama URL: {OLLAMA_BASE_URL}")
            
            # Embedding Bilgileri
            print(f"Embedding Sağlayıcı: {EMBEDDING_PROVIDER}")
            if EMBEDDING_PROVIDER == "openai":
                print(f"Embedding Modeli: {OPENAI_EMBEDDING_MODEL}")
            elif EMBEDDING_PROVIDER == "ollama":
                print(f"Embedding Modeli: {OLLAMA_EMBEDDING_MODEL}")
            elif EMBEDDING_PROVIDER == "instructor":
                print(f"Embedding Modeli: {INSTRUCTOR_MODEL_NAME}")
            
            # Chunk Bilgileri
            print(f"Parça Boyutu: {CHUNK_SIZE} karakter")
            print(f"Parça Örtüşmesi: {CHUNK_OVERLAP} karakter")
            
            # Veritabanı Bilgileri
            if self.vector_db.vector_store is not None:
                stats = self.vector_db.get_collection_stats()
                print(f"İndeks Adı: {stats['index_name']}")
                print(f"İndeks Tipi: {stats['index_type']}")
                print(f"Toplam Doküman Parçası: {stats['document_count']}")
            else:
                print(f"{Fore.YELLOW}ℹ Henüz hiç doküman yüklenmemiş.{Style.RESET_ALL}")
                
            print(f"{Fore.BLUE}───────────────────────────{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Bilgiler görüntülenirken hata oluştu: {str(e)}{Style.RESET_ALL}")


def setup_argparse():
    """Komut satırı argümanlarını ayarlar."""
    parser = argparse.ArgumentParser(
        description="BilgiÇekirdeği: Dokümanlarınızı vektor veritabanında indeksleyip sorgulayın.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Komut")
    
    # PDF yükleme komutu
    pdf_parser = subparsers.add_parser("load_pdf", help="PDF dokümanı yükle")
    pdf_parser.add_argument("filepath", type=str, help="PDF dosyasının yolu")
    pdf_parser.add_argument("--chunk-size", type=int, default=1000, help="Metin bölümü boyutu")
    pdf_parser.add_argument("--chunk-overlap", type=int, default=200, help="Metin bölümleri örtüşme miktarı")
    pdf_parser.add_argument("--collection", type=str, default="documents", help="Koleksiyon adı")
    pdf_parser.add_argument("--embedding-provider", type=str, default=os.getenv("EMBEDDING_PROVIDER", "openai"),
                          help="Embedding sağlayıcısı (openai, ollama, instructor)")
    pdf_parser.add_argument("--model-name", type=str, help="Belirli bir model adı")
    
    # Sorgu komutu
    query_parser = subparsers.add_parser("query", help="Veritabanını sorgula")
    query_parser.add_argument("question", type=str, help="Sorulacak soru")
    query_parser.add_argument("--collection", type=str, default="documents", help="Sorgulanacak koleksiyon")
    query_parser.add_argument("--llm-provider", type=str, default=os.getenv("LLM_PROVIDER", "openai"),
                            help="LLM sağlayıcısı (openai, ollama)")
    query_parser.add_argument("--model-name", type=str, 
                            help="LLM model adı (belirtilmezse varsayılan kullanılır)")
    
    return parser.parse_args()

def main():
    """Ana fonksiyon: Komut satırı argümanlarını ayrıştırır ve uygun işlevi çağırır."""
    args = setup_argparse()
    
    # Vektor veritabanını başlat
    vector_db = VectorDatabase()
    
    if args.command == "load_pdf":
        try:
            # Model adı için ortam değişkenlerini kontrol et
            model_name = args.model_name
            if model_name is None:
                if args.embedding_provider == "openai":
                    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
                elif args.embedding_provider == "ollama":
                    model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama2")
                elif args.embedding_provider == "instructor":
                    model_name = os.getenv("INSTRUCTOR_MODEL", "hkunlp/instructor-large")
            
            # PDF dosyasını yükle ve işle
            logger.info(f"PDF yükleniyor: {args.filepath}")
            result = load_pdf(
                filepath=args.filepath,
                vector_db=vector_db,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_provider=args.embedding_provider,
                model_name=model_name,
                collection_name=args.collection
            )
            
            # Özet bilgileri göster
            logger.info(f"İşlem tamamlandı: {result['filename']}")
            logger.info(f"Toplam bölüm sayısı: {result['document_count']}")
            logger.info(f"Koleksiyon: {result['collection_name']}")
            
        except Exception as e:
            logger.error(f"PDF yükleme hatası: {str(e)}")
            sys.exit(1)
    
    elif args.command == "query":
        try:
            # Model adı için ortam değişkenlerini kontrol et
            model_name = args.model_name
            if model_name is None:
                if args.llm_provider == "openai":
                    model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                elif args.llm_provider == "ollama":
                    model_name = os.getenv("OLLAMA_MODEL", "llama2")
            
            # Ollama base URL
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            # Koleksiyonu yükle
            logger.info(f"Koleksiyon yükleniyor: {args.collection}")
            vector_db.load_collection(args.collection)
            
            # RAG zincirini oluştur
            logger.info(f"RAG zinciri oluşturuluyor ({args.llm_provider}/{model_name})")
            rag_chain = RAGChain(
                vector_db=vector_db,
                provider=args.llm_provider,
                model_name=model_name,
                base_url=base_url
            )
            
            # Soruyu sor
            logger.info(f"Soru soruluyor: {args.question}")
            result = rag_chain.ask(args.question)
            
            # Yanıtı göster
            print("\n" + "=" * 60)
            print("SORU:", result["question"])
            print("=" * 60)
            print("YANIT:")
            print(result["answer"])
            print("-" * 60)
            
            # Kaynakları göster
            print("KAYNAKLAR:")
            for i, source in enumerate(rag_chain.format_source_documents(result["source_documents"])):
                print(f"\nKaynak {i+1}:")
                print(f"- İçerik: {source['content']}")
                print(f"- Dosya: {source['metadata'].get('filename', 'Bilinmiyor')}")
                print(f"- Sayfa: {source['metadata'].get('page', 'Bilinmiyor')}")
            
        except Exception as e:
            logger.error(f"Sorgu hatası: {str(e)}")
            sys.exit(1)
    
    else:
        logger.error("Geçersiz komut. 'load_pdf' veya 'query' kullanın.")
        sys.exit(1)


if __name__ == '__main__':
    # Karşılama mesajı
    print(f"\n{Fore.BLUE}╔══════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.BLUE}║      BilgiÇekirdeği - Knowledge Kernel     ║{Style.RESET_ALL}")
    print(f"{Fore.BLUE}╚══════════════════════════════════════════╝{Style.RESET_ALL}\n")
    
    # LLM Sağlayıcı Bilgisi
    if LLM_PROVIDER == "openai":
        print(f"{Fore.GREEN}ℹ Kullanılan LLM: OpenAI ({OPENAI_MODEL}){Style.RESET_ALL}")
    elif LLM_PROVIDER == "ollama":
        print(f"{Fore.GREEN}ℹ Kullanılan LLM: Ollama ({OLLAMA_MODEL}){Style.RESET_ALL}")
    
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Program kullanıcı tarafından sonlandırıldı.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Beklenmeyen hata: {str(e)}{Style.RESET_ALL}") 