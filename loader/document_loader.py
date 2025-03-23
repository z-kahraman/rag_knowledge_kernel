"""
BilgiÇekirdeği Doküman Yükleyici Modülü
-----------------------------
Bu modül, çeşitli formatlardaki dokümanları okuyup parçalara böler.
Desteklenen formatlar: PDF, DOCX, TXT
"""

import os
from typing import List, Optional, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain.schema import Document

class DocumentLoader:
    """Dokümanları yükleyip işleyen sınıf."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Doküman yükleyiciyi başlatır.
        
        Args:
            chunk_size: Her bir parçanın maksimum karakter sayısı
            chunk_overlap: Parçalar arasındaki örtüşme miktarı
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Verilen yoldaki dokümanı yükler ve parçalara böler.
        
        Args:
            file_path: Yüklenecek dokümanın yolu
            
        Returns:
            Document nesnelerinin listesi
        
        Raises:
            ValueError: Desteklenmeyen dosya formatı
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # Dosya formatına göre uygun yükleyiciyi seç
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                # Desteklenmeyen format için genel yükleyici dene
                print(f"Uyarı: {file_extension} için özel yükleyici yok. Genel yükleyici deneniyor.")
                loader = UnstructuredFileLoader(file_path)
                
            # Dokümanı yükle
            documents = loader.load()
            
            # Metadata ekle (dosya adı ve yolu)
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["source"] = file_path
                doc.metadata["file_name"] = os.path.basename(file_path)
            
            # Dokümanı parçalara böl
            split_documents = self.text_splitter.split_documents(documents)
            
            print(f"{file_path} dokümani yüklendi ve {len(split_documents)} parçaya bölündü.")
            return split_documents
            
        except Exception as e:
            print(f"Hata: {file_path} dokümani yüklenirken bir hata oluştu: {str(e)}")
            raise
    
    def load_documents_from_directory(self, directory_path: str, 
                                     extensions: List[str] = ['.pdf', '.txt', '.docx']) -> List[Document]:
        """
        Verilen dizindeki tüm dokümanları yükler.
        
        Args:
            directory_path: Dokümanların bulunduğu klasör
            extensions: İşlenecek dosya uzantıları listesi
            
        Returns:
            Tüm dokümanlardan oluşan Document nesnelerinin listesi
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError(f"Geçersiz dizin: {directory_path}")
        
        all_documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Dosya mı ve desteklenen bir uzantıya sahip mi kontrol et
            if (os.path.isfile(file_path) and 
                any(filename.lower().endswith(ext) for ext in extensions)):
                try:
                    documents = self.load_document(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Uyarı: {file_path} yüklenemedi, atlanıyor. Hata: {str(e)}")
        
        return all_documents 