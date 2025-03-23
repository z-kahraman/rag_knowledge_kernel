"""
BilgiÇekirdeği - Kişisel Dokümanları Yapay Zeka ile Sorgulama Sistemi
--------------------------------------------------------------
Bu uygulama, dokümanlarınızı yerel ve çevrimiçi yapay zeka modelleri ile
sorgulamanızı sağlayan açık kaynaklı bir bilgi erişim sistemidir.

Knowledge Kernel - Personal Document AI Query System
--------------------------------------------------------------
This application is an open-source information retrieval system that allows you to query
your documents using local and online artificial intelligence models.
"""

import os
import sys
import time
import json
import requests
import streamlit as st
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import glob
import atexit
import shutil
import uuid
import re

# Dil desteği için çeviri modülünü içe aktar
# Import translation module for language support
from localization.translations import get_text, DEFAULT_LANGUAGE

# Geçici dosyaların yönetimi için fonksiyonlar
TEMP_DIR = "./temp_files"

def setup_temp_directory():
    """Geçici dosyalar için dizin oluşturur"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR

def cleanup_temp_files():
    """Mevcut geçici PDF dosyalarını ana dizinden temizler"""
    # Ana dizindeki temp_*.pdf dosyalarını temizle
    for temp_file in glob.glob("temp_*.pdf"):
        try:
            os.remove(temp_file)
            print(f"Geçici dosya temizlendi: {temp_file}")
        except Exception as e:
            print(f"Dosya temizlenirken hata: {e}")
    
    # Eğer varsa temp_files dizinindeki dosyaları temizle
    if os.path.exists(TEMP_DIR):
        for temp_file in glob.glob(os.path.join(TEMP_DIR, "*")):
            try:
                os.remove(temp_file)
                print(f"Geçici dosya temizlendi: {temp_file}")
            except Exception as e:
                print(f"Dosya temizlenirken hata: {e}")

def exit_handler():
    """Uygulama kapatılırken çalışacak temizleme işlevi"""
    print("Uygulama kapatılıyor, geçici dosyalar temizleniyor...")
    cleanup_temp_files()
    # İsteğe bağlı: Geçici dizini tamamen kaldırma
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"{TEMP_DIR} dizini silindi")
        except Exception as e:
            print(f"Dizin silinirken hata: {e}")

# Başlangıçta geçici dosyaları temizle
cleanup_temp_files()
# Geçici dosyalar için dizin oluştur
setup_temp_directory()
# Uygulama kapanırken temizleme işlevini kaydet
atexit.register(exit_handler)

# BilgiÇekirdeği modülleri
from vectorstore.vector_db import VectorDatabase
from ingestion.load_pdf import load_pdf
from qa.rag_chain import RAGChain
from utils.logging_config import setup_logging, get_logger
from load_pdf import load_pdf_document
from run_query import run_query, clear_query_cache, clear_rag_cache, clear_collection_cache

# Loglama yapılandırmasını etkinleştir
setup_logging()
logger = get_logger(__name__)

# Dil ve çeviri fonksiyonları
# Language and translation functions

def get_current_language():
    """
    Geçerli dil kodunu döndürür. Varsayılan "tr" (Türkçe).
    
    Returns the current language code. Default is "tr" (Turkish).
    """
    return st.session_state.get("language", DEFAULT_LANGUAGE)

def t(key):
    """
    Geçerli dil için çeviriyi döndürür.
    
    Args:
        key: Metin anahtarı
        
    Returns:
        str: Çevrilmiş metin
    
    Returns the translation for the current language.
    
    Args:
        key: Text key
        
    Returns:
        str: Translated text
    """
    return get_text(key, get_current_language())

# Yardımcı Fonksiyonlar
def get_ollama_models(base_url="http://localhost:11434"):
    """
    Ollama API'sini sorgulayarak yüklü modelleri listeler.
    
    Args:
        base_url: Ollama API URL'i
        
    Returns:
        list: Yüklü model adlarının listesi
    """
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            # Model isimlerini çıkar
            return [model['name'] for model in models]
        else:
            logger.warning(f"Ollama API'ye erişilemedi. Durum kodu: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Ollama modelleri listelenirken hata: {str(e)}")
        return []

def save_collection_metadata(collection_name, metadata):
    """
    Koleksiyon için metadata bilgilerini kaydeder.
    
    Args:
        collection_name: Koleksiyon adı
        metadata: Kaydedilecek metadata bilgileri
    """
    try:
        metadata_dir = os.path.join("./indices", collection_name, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata_file = os.path.join(metadata_dir, "collection_info.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Koleksiyon metadata kaydedildi: {collection_name}")
    except Exception as e:
        logger.error(f"Metadata kaydedilirken hata: {str(e)}")

def load_collection_metadata(collection_name):
    """
    Koleksiyon metadata bilgilerini yükler ve eksik bilgileri varsayılan değerlerle doldurur.
    
    Args:
        collection_name: Koleksiyon adı
        
    Returns:
        dict: Metadata bilgileri veya boş sözlük
    """
    try:
        metadata_file = os.path.join("./indices", collection_name, "metadata", "collection_info.json")
        metadata = {}
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Metadata bulunamamışsa veya boşsa
        if not metadata:
            logger.warning(f"Metadata bulunamadı veya boş: {collection_name}")
            return {
                "created_date": t("unknown"),
                "num_documents": 0,
                "num_vectors": 0,
                "chunk_size": t("unknown"),
                "chunk_overlap": t("unknown"),
                "embedding_type": t("unknown"),
                "embedding_model": t("unknown")
            }
        
        # Eksik alanları varsayılan değerlerle doldur
        default_values = {
            "created_date": t("unknown"),
            "num_documents": 0,
            "num_vectors": 0,
            "chunk_size": t("unknown"),
            "chunk_overlap": t("unknown"),
            "embedding_type": metadata.get("embedding_provider", t("unknown")),
            "embedding_model": t("unknown")
        }
        
        # Metadata'daki eksik alanları doldur
        for key, value in default_values.items():
            if key not in metadata or metadata[key] is None or metadata[key] == "":
                metadata[key] = value
        
        return metadata
    except Exception as e:
        logger.error(f"Metadata yüklenirken hata: {str(e)}")
        return {
            "created_date": t("unknown"),
            "num_documents": 0,
            "num_vectors": 0,
            "chunk_size": t("unknown"), 
            "chunk_overlap": t("unknown"),
            "embedding_type": t("unknown"),
            "embedding_model": t("unknown")
        }

# Varsayılan değerler
DEFAULT_COLLECTION = "documents"
LLM_PROVIDER = "ollama" 
LLM_MODEL = "llama3.2:latest"
EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "llama3.2:latest"

# Sayfa yapılandırması
st.set_page_config(
    page_title="BilgiÇekirdeği - Knowledge Kernel",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state başlatma
# Initialize session state
if "language" not in st.session_state:
    st.session_state["language"] = DEFAULT_LANGUAGE

# CSS stilleri ekle - Koyu tema için güncellendi
st.markdown("""
<style>
    body {
        background-color: #121212;
        color: #f0f0f0;
    }
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #2196F3 !important;
        margin-bottom: 0.5rem !important;
    }
    .sub-header {
        font-size: 1.1rem !important;
        color: #e0e0e0 !important;
        margin-bottom: 2rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px;
        padding: 10px 15px;
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2196F3 !important;
        color: white !important;
    }
    .stSidebar [data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-weight: 600;
        color: #2196F3;
    }
    div[data-testid="stExpander"] {
        border-radius: 5px;
        border: 1px solid #333333;
    }
    .collection-card {
        background-color: #252525;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #2196F3;
    }
    .stButton button {
        border-radius: 5px;
        font-weight: 500;
    }
    .info-box {
        background-color: #1a2733;
        border-radius: 5px;
        padding: 15px;
        border-left: 4px solid #2196F3;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }
    .stAlert {
        border-radius: 5px;
    }
    .stDataFrame {
        color: #e0e0e0;
    }
    .stDataFrame [data-testid="stTable"] {
        background-color: #1e1e1e !important;
    }
    .stDataFrame [data-testid="stTable"] table {
        color: #e0e0e0 !important;
    }
    .stDataFrame [data-testid="stTable"] th {
        background-color: #333333 !important;
        color: #e0e0e0 !important;
    }
    .stDataFrame [data-testid="stTable"] td {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    input, textarea, [data-baseweb=select] {
        background-color: #333333 !important;
        color: #e0e0e0 !important;
    }
    div[data-testid="stForm"] {
        background-color: #1e1e1e;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Kenar çubuğunda dil seçimi ekle
# Add language selection in the sidebar
with st.sidebar:
    # Uygulama bilgileri
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f9e0.svg", width=50)
    
    # Uygulama adı - Mevcut dile göre
    st.title(t("app_title"))
    
    # Dil seçimi
    st.subheader(t("settings_language"))
    language_col1, language_col2 = st.columns(2)
    
    with language_col1:
        if st.button("🇹🇷 " + t("settings_language_turkish"), use_container_width=True, 
                   disabled=get_current_language()=="tr"):
            st.session_state["language"] = "tr"
            st.success(t("settings_language_change"))
            st.rerun()
    
    with language_col2:
        if st.button("🇬🇧 " + t("settings_language_english"), use_container_width=True,
                   disabled=get_current_language()=="en"):
            st.session_state["language"] = "en"
            st.success(t("settings_language_change"))
            st.rerun()
    
    st.divider()
    
    # Ayarlar bölümü
    st.markdown(f'<h3 style="color:#e0e0e0; font-weight:600; font-size:1.3rem;">⚙️ {t("settings_title")}</h3>', unsafe_allow_html=True)
    
    # Koleksiyon adı
    st.markdown(f'<p style="font-weight:500; margin-bottom:5px;">{t("upload_collection_label")}</p>', unsafe_allow_html=True)
    collection_name = st.text_input(
        label=t("upload_collection_label"),
        value=DEFAULT_COLLECTION, 
        key="collection_name_input", 
        label_visibility="collapsed"
    )
    
    # LLM Ayarları
    with st.expander(f"🤖 {t('settings_llm_section')}", expanded=True):
        llm_provider = st.selectbox(
            t("settings_llm_provider"), 
            ["ollama", "openai"], 
            index=0,
            key="llm_provider_select"
        )
        
        if llm_provider == "ollama":
            ollama_base_url = st.text_input(
                "Ollama API URL", 
                "http://localhost:11434",
                key="ollama_url_input"
            )
            
            # Mevcut Ollama modellerini göster
            ollama_models = st.session_state.get('ollama_models', [])
            if ollama_models:
                llm_model = st.selectbox(
                    t("settings_llm_model"), 
                    options=ollama_models,
                    index=ollama_models.index(LLM_MODEL) if LLM_MODEL in ollama_models else 0,
                    help="Ollama üzerinde yüklü olan modellerden birini seçin",
                    key="ollama_model_select"
                )
            else:
                llm_model = st.text_input(
                    t("settings_llm_model"), 
                    LLM_MODEL,
                    help="Ollama API'ye ulaşılamadı. Model adını manuel olarak girin.",
                    key="ollama_model_input"
                )
                st.button(
                    "🔄 Modelleri Yenile", 
                    key="refresh_models", 
                    on_click=lambda: st.session_state.update({'ollama_models': get_ollama_models(ollama_base_url)})
                )
        else:
            llm_model = st.selectbox(
                t("settings_llm_model"), 
                ["gpt-3.5-turbo", "gpt-4"], 
                index=0,
                help="OpenAI'nin üretimde olan modellerinden birini seçin",
                key="openai_model_select"
            )
            openai_api_key = st.text_input(
                t("settings_openai_api_key"), 
                type="password",
                key="openai_api_key_input"
            )
    
    # Embedding Ayarları
    with st.expander(f"🧬 {t('settings_embedding_section')}", expanded=True):
        embedding_provider = st.selectbox(
            t("settings_embedding_provider"), 
            ["ollama", "openai", "instructor"], 
            index=0,
            help="Metinleri vektörlere dönüştürmek için kullanılacak servis.",
            key="embedding_provider_select"
        )
        
        if embedding_provider == "ollama":
            # Mevcut Ollama modellerini göster
            ollama_models = st.session_state.get('ollama_models', [])
            if ollama_models:
                embedding_model = st.selectbox(
                    "Ollama Embedding Model", 
                    options=ollama_models,
                    index=ollama_models.index(EMBEDDING_MODEL) if EMBEDDING_MODEL in ollama_models else 0,
                    help="Vektörleştirme için kullanılacak model",
                    key="embedding_ollama_model_select"
                )
            else:
                embedding_model = st.text_input(
                    "Ollama Embedding Model", 
                    EMBEDDING_MODEL,
                    help="Vektörleştirme için kullanılacak model",
                    key="embedding_ollama_model_input"
                )
        elif embedding_provider == "openai":
            embedding_model = st.text_input(
                "OpenAI Embedding Model", 
                "text-embedding-ada-002",
                help="OpenAI'nin embedding API'si için model adı",
                key="embedding_openai_model_input"
            )
            if 'openai_api_key' not in locals():
                openai_api_key = st.text_input(
                    "OpenAI API Anahtarı", 
                    type="password", 
                    key="openai_api_key_embedding"
                )
        elif embedding_provider == "instructor":
            embedding_model = st.text_input(
                "Instructor Model", 
                "hkunlp/instructor-large",
                help="Instructor modelinin HuggingFace ismi",
                key="embedding_instructor_model_input"
            )

    # Gelişmiş Ayarlar
    with st.expander("🔧 Gelişmiş Ayarlar"):
        # Bölümleme ayarları
        st.markdown('<p style="font-weight:500; color:#2196F3;">Bölümleme Ayarları</p>', unsafe_allow_html=True)
        chunk_size = st.slider(
            "Bölüm Boyutu", 
            200, 2000, 1000,
            help="Her bir metin parçasının maksimum karakter sayısı",
            key="chunk_size_slider"
        )
        chunk_overlap = st.slider(
            "Bölüm Örtüşmesi", 
            0, 400, 200,
            help="Ardışık bölümler arasındaki örtüşme miktarı (karakter sayısı)",
            key="chunk_overlap_slider"
        )
        
        # LLM sıcaklık ayarı
        st.markdown('<p style="font-weight:500; color:#2196F3;">LLM Ayarları</p>', unsafe_allow_html=True)
        temperature = st.slider(
            "Sıcaklık", 
            0.0, 1.0, 0.2, 0.1,
            help="Daha düşük değerler daha tutarlı, daha yüksek değerler daha yaratıcı yanıtlar üretir",
            key="temperature_slider"
        )
        
        # Top K değeri
        top_k = st.slider(
            "Top K", 
            1, 10, 3,
            help="Sorgu için kaç doküman parçasının kullanılacağı",
            key="top_k_slider"
        )

    # Bilgi kutucuğu
    st.markdown(f'<div class="info-box"><strong>⚠️ {t("important_note")}</strong> {t("embedding_model_warning")}</div>', unsafe_allow_html=True)
    
    # Koleksiyon bilgileri
    st.markdown(f'<h3 style="color:#424242; font-weight:600; font-size:1.3rem; margin-top:20px;">📚 {t("collections_title")}</h3>', unsafe_allow_html=True)
    
    # İndeks bilgilerini göster
    vector_db = VectorDatabase()
    try:
        collections = vector_db.list_collections()
        if collections:
            st.success(f"{len(collections)} {t('collection_found')}")
            for collection in collections:
                # Metadata'yı yükle
                metadata = load_collection_metadata(collection)
                if metadata:
                    model_info = f"({metadata.get('embedding_provider', '')}/" \
                                 f"{metadata.get('embedding_model', '')})"
                    st.markdown(f'<div class="collection-card"><strong>{collection}</strong><br/>' \
                                f'<small>{model_info}</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="collection-card"><strong>{collection}</strong></div>', unsafe_allow_html=True)
        else:
            st.warning(t("no_collections_warning"))
    except Exception as e:
        st.error(f"{t('collections_load_error')}: {str(e)}")

# Başlık ve açıklama
st.markdown(f'<h1 class="main-header">{t("app_title")}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{t("welcome_text")}</p>', unsafe_allow_html=True)

# Arka planda Ollama modellerini yükle
if 'ollama_models' not in st.session_state:
    st.session_state['ollama_models'] = get_ollama_models()

# Sekmeler
# Aktif sekmeyi belirle
active_tab_index = 0  # Varsayılan olarak ilk sekme (Ana Sayfa)
if "active_tab" in st.session_state:
    active_tab_index = st.session_state["active_tab"]
    del st.session_state["active_tab"]  # Kullanıldıktan sonra temizle

# Sekmeleri oluştur
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📄 {t('menu_home')}", 
    f"📥 {t('menu_pdf_upload')}", 
    f"❓ {t('menu_ask')}", 
    f"📚 {t('menu_collections')}", 
    f"📊 {t('collections_stats')}"
])

# Aktif sekmeye göre içeriği göster
# 1. Ana Sayfa Sekmesi
with tab1:
    st.header(t("welcome_title"))
    
    # Ana sayfa bilgileri ve hoş geldin mesajı
    st.markdown(f"### {t('home_subtitle')}")
    st.markdown(t("home_description"))
    
    # İki sütunlu düzen
    home_col1, home_col2 = st.columns([3, 2])
    
    with home_col1:
        # Kullanıcı bilgi kartı
        st.markdown(f"""
        <div style="background-color: #1a273a; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #2196F3;">
            <h4 style="color: #42a5f5; margin-top: 0;">{t("app_usage_title")}</h4>
            <p>1. "{t('menu_pdf_upload')}" {t("app_usage_step1")}</p>
            <p>2. "{t('menu_ask')}" {t("app_usage_step2")}</p>
            <p>3. "{t('menu_collections')}" {t("app_usage_step3")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(t("home_get_started"))
    
    with home_col2:
        # Logo ve görsel öğeler
        st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f9e0.svg", width=150)
        
        # Sürüm bilgisi
        st.markdown(f"**{t('app_version')} v1.0.0**")
        
        # GitHub linki
        st.markdown("[GitHub](https://github.com/user/knowledge_kernel) • [Docs](https://github.com/user/knowledge_kernel/docs)")

# 2. Doküman Yükleme Sekmesi
with tab2:
    st.header(t("upload_title"))
    st.markdown(t("upload_description"))
    
    # İki sütunlu düzen
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Kullanıcı bilgi kartı
        st.markdown(f"""
        <div style="background-color: #1a273a; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #2196F3;">
            <h4 style="color: #42a5f5; margin-top: 0;">{t("pdf_upload_instructions_title")}</h4>
            <p>1. {t("process_step_1")}</p>
            <p>2. {t("process_step_2")}</p>
            <p>3. {t("process_step_3")}</p>
            <p>4. {t("process_step_4")}</p>
            <p>5. {t("process_step_5")}</p>
            <p>6. {t("process_step_6")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            t("upload_button"), 
            type="pdf",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            # Geçici dosyayı TEMP_DIR dizinine kaydet ve işle
            temp_filename = f"temp_{int(time.time())}_{uploaded_file.name}"
            file_path = os.path.join(TEMP_DIR, temp_filename)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"'{uploaded_file.name}' {t('upload_success')}! {t('file_upload_prompt')}")
            
            # Session state'e dosya yolunu kaydet (daha sonra silinebilir)
            if 'temp_files' not in st.session_state:
                st.session_state['temp_files'] = []
            st.session_state['temp_files'].append(file_path)
            
            process_col1, process_col2 = st.columns([1, 1])
            
            with process_col1:
                # Yükleme düğmesi
                process_button = st.button(
                    t("upload_process_button"), 
                    use_container_width=True,
                    key="isle_button"
                )
            with process_col2:
                # İptal düğmesi
                cancel_button = st.button(
                    t("upload_cancel_button"), 
                    use_container_width=True,
                    key="iptal_button"
                )
            
            if process_button:
                with st.spinner("PDF işleniyor ve vektör veritabanına ekleniyor..."):
                    try:
                        # PDF'yi yükle
                        result = load_pdf_document(
                            pdf_path=file_path,
                            embedding_provider=embedding_provider,
                            embedding_model=embedding_model,
                            collection_name=collection_name,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            openai_api_key=openai_api_key if embedding_provider == "openai" else None
                        )
                        
                        if result:
                            st.balloons()
                            st.success("PDF başarıyla işlendi ve vektör veritabanına eklendi!")
                            # Sonuçları göster
                            st.json(result)
                            # Geçici dosyayı temizle
                            try:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                    print(f"Geçici dosya başarıyla silindi: {file_path}")
                                # Session state'den dosya yolunu kaldır
                                if 'temp_files' in st.session_state and file_path in st.session_state['temp_files']:
                                    st.session_state['temp_files'].remove(file_path)
                            except Exception as e:
                                print(f"Geçici dosya silinirken hata: {str(e)}")
                        else:
                            st.error("PDF işlenirken bir hata oluştu.")
                    except Exception as e:
                        st.error(f"Hata: {str(e)}")
                        # Hata durumunda da geçici dosyayı temizlemeyi dene
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"Hata sonrası geçici dosya silindi: {file_path}")
                            # Session state'den dosya yolunu kaldır
                            if 'temp_files' in st.session_state and file_path in st.session_state['temp_files']:
                                st.session_state['temp_files'].remove(file_path)
                        except Exception as cleanup_error:
                            print(f"Hata durumunda geçici dosya silinirken hata: {str(cleanup_error)}")
            
            if cancel_button:
                # İptal edilirse geçici dosyayı temizle
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        st.success(f"İşlem iptal edildi ve geçici dosya silindi.")
                    # Session state'den dosya yolunu kaldır
                    if 'temp_files' in st.session_state and file_path in st.session_state['temp_files']:
                        st.session_state['temp_files'].remove(file_path)
                except Exception as e:
                    st.error(f"Geçici dosya silinirken hata: {str(e)}")
                st.rerun()  # Sayfayı yenile
    
    with col2:
        # İşlem sonuçları ve bilgiler burada gösterilecek
        if uploaded_file is None:
            st.markdown(f"""
            <div style="background-color: #2c2c00; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #FFD600;">
                <h4 style="color: #FFD600; margin-top: 0;">{t("pdf_upload_instructions_title")}</h4>
                <p>1. {t("process_step_1")}</p>
                <p>2. {t("process_step_2")}</p>
                <p>3. {t("process_step_3")}</p>
                <p>4. {t("process_step_4")}</p>
                <p>5. {t("process_step_5")}</p>
                <p>6. {t("process_step_6")}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Yüklenen dosya bilgilerini göster
            st.markdown(f"""
            <div style="background-color: #203a25; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #4CAF50;">
                <h4 style="color: #81c784; margin-top: 0;">Yüklenen Dosya Bilgileri</h4>
                <p><strong>Dosya Adı:</strong> {uploaded_file.name}</p>
                <p><strong>Dosya Boyutu:</strong> {uploaded_file.size / 1024:.2f} KB</p>
                <p><strong>Hedef Koleksiyon:</strong> {collection_name}</p>
                <p><strong>Embedding Sağlayıcı:</strong> {embedding_provider}</p>
                <p><strong>Embedding Modeli:</strong> {embedding_model}</p>
            </div>
            """, unsafe_allow_html=True)

# 3. Soru Sorma Sekmesi 
with tab3:
    # Eğer sorgu sekmesine yönlendirme varsa, bilgi mesajı göster
    if st.session_state.get("redirect_to_query", False):
        st.info(f"'{st.session_state.get('selected_collection', 'documents')}' koleksiyonuna sorgu yapabilirsiniz.")
        # Yönlendirme durumunu sıfırla
        st.session_state["redirect_to_query"] = False
        
    st.header(t("ask_title"))
    st.markdown(t("ask_description"))
    
    # İki sütunlu düzen
    query_col1, query_col2 = st.columns([3, 2])
    
    with query_col1:
        # Sorgu alanı
        st.markdown('<p style="font-weight:500; margin-bottom:5px;">' + t("ask_enter_question") + '</p>', unsafe_allow_html=True)
        
        # Session state'teki sorguyu başlangıçta al
        if 'query' not in st.session_state:
            st.session_state['query'] = ""
        
        # Örnek soru seçim izleyicisi
        if 'last_selected_question' not in st.session_state:
            st.session_state['last_selected_question'] = None
            
        # Sorgu değiştiğinde bu fonksiyon çalışacak
        def on_query_change():
            st.session_state['current_query'] = st.session_state.soru_input
        
        # Sorgu alanı
        query = st.text_area(
            label=t("ask_question"),
            value=st.session_state.get('query', ''),
            height=120, 
            placeholder=t("ask_placeholder"), 
            label_visibility="collapsed",
            key="soru_input",
            on_change=on_query_change  # Değişiklik olduğunda bu fonksiyonu çağır
        )
        
        # Session state'i güncelle - kullanıcı yazdıkça güncellenir
        st.session_state['query'] = query
        st.session_state['current_query'] = query  # En son sorguyu her zaman güncel tut
        
        # Sorgu ve ayarlar satırı
        button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
        
        with button_col1:
            # Sorgu düğmesi
            ask_button = st.button(
                f"🔍 {t('ask_button')}", 
                use_container_width=True, 
                type="primary",
                key="soru_sor_button"
            )
        
        with button_col2:
            # Temizle düğmesi
            clear_button = st.button(
                f"🧹 {t('clear_button')}", 
                use_container_width=True,
                key="temizle_button"
            )

        # Örnek soru bölümünü butonlardan sonra doğrudan yerleştir
        # Örnek soru dropdown menüsü - her zaman görünür
        example_questions = [
            # Genel sorular - doküman inceleme
            t("example_question_1"),
            t("example_question_2"),
            t("example_question_3"),
            t("example_question_4"),
            t("example_question_5"),
            t("example_question_6"),
            
            # Alan bazlı spesifik sorular
            t("example_question_7"),
            t("example_question_8"),
            t("example_question_9"),
            t("example_question_10"),
            
            # Kullanım senaryolarına göre sorular
            t("example_question_11"),
            t("example_question_12"),
            t("example_question_13"),
        ]
        
        # Dropdown ile soru seçimi - seçim yapıldığında callback'i tetikleyecek
        def on_example_select():
            # Seçilen değer varsa ve öncekinden farklıysa
            if st.session_state.example_question_select and st.session_state.example_question_select != st.session_state.get('last_selected_question'):
                # Son seçilen soruyu güncelle
                st.session_state['last_selected_question'] = st.session_state.example_question_select
                # Soru metnini güncelle
                st.session_state.query = st.session_state.example_question_select
                st.session_state.soru_input = st.session_state.example_question_select
            
        # Dropdown ile soru seçimi
        selected_question = st.selectbox(
            t("example_question_select_prompt"),
            options=example_questions,
            index=None,
            placeholder=t("example_question_select_placeholder"),
            key="example_question_select",
            on_change=on_example_select  # Değişiklikte callback çağır
        )
        
        # Seçilen soruysa bilgi ver
        if selected_question and selected_question == st.session_state.get('last_selected_question'):
            st.info(f"{t('selected_question')}: {selected_question}")
        
        # Eğer temizle istenirse
        if clear_button:
            # Burada doğrudan session state'i temizliyoruz ve sayfayı yeniliyoruz
            st.session_state['query'] = ""
            if 'answer' in st.session_state:
                del st.session_state['answer']
            if 'source_docs' in st.session_state:
                del st.session_state['source_docs']
            st.rerun()
        
        # Sorgu düğmesi veya enter tuşuna basıldıysa
        if ask_button and query:
            # Burada güncel sorgu metin alanından alınmış olmalı
            current_query = query
            # Sorgu metnini güncel tut, burada çok kritik!
            st.session_state['current_query'] = current_query
            
            # Daha önce aynı soru sorulmuş mu kontrol et
            from run_query import clear_query_cache, clear_rag_cache, clear_collection_cache

            # Gelişmiş ayarlara bakalım
            with st.expander(t("cache_settings"), expanded=False):
                use_cache = st.checkbox(t("use_cache"), value=True, 
                                      help=t("cache_help"))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(t("clear_collection_cache")):
                        # Sadece mevcut koleksiyonun önbelleğini temizle
                        collection_name = st.session_state.get('selected_collection', 'documents')
                        clear_collection_cache(collection_name)
                        
                        # Session state'ten de yanıtları temizle
                        if 'answer' in st.session_state:
                            del st.session_state['answer']
                        if 'source_docs' in st.session_state:
                            del st.session_state['source_docs']
                        st.success(f"'{collection_name}' {t('collection_cache_cleared')}")
                        st.info(t("answers_cleared"))
                
                with col2:
                    if st.button(t("clear_all_caches")):
                        clear_query_cache()
                        clear_rag_cache()
                        # Session state'ten de yanıtları temizle
                        if 'answer' in st.session_state:
                            del st.session_state['answer']
                        if 'source_docs' in st.session_state:
                            del st.session_state['source_docs']
                        st.success(t("all_caches_cleared"))
                        st.info(t("answers_cleared"))
            
            # İlerleme göstergesi başlat
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
            
            # Zamanlayıcı başlat
            start_time = time.time()
            
            with st.spinner(t("generating_answer")):
                try:
                    # İlerleme göstergesi kademeli olarak ilerleyecek
                    for percent_complete in range(0, 101, 5):
                        progress_bar.progress(percent_complete)
                        # Son %20'de daha yavaş ilerlesin ki kullanıcı beklerken daha iyi bir deneyim yaşasın
                        if percent_complete > 80:
                            time.sleep(0.3)
                        elif percent_complete > 40:
                            time.sleep(0.1)
                        else:
                            time.sleep(0.05)
                    
                    # Mevcut dil tercihini al
                    current_language = get_current_language()
                    logger.info(f"Sorgu dili: {current_language}")
                    
                    # Sorguyu çalıştır
                    answer, source_docs = run_query(
                        query=current_query,
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        temperature=temperature,
                        top_k=top_k,
                        collection_name=collection_name,
                        openai_api_key=openai_api_key if llm_provider == "openai" else None,
                        use_cache=use_cache,
                        language=current_language  # Dil bilgisini aktarıyoruz
                    )
                    
                    # İşlem süresini hesapla
                    elapsed_time = time.time() - start_time
                    
                    # Progress barı kaldır
                    progress_bar.empty()
                    
                    # Session state'e kaydet
                    st.session_state['answer'] = answer
                    st.session_state['source_docs'] = source_docs
                    
                    # Süre bilgisini göster
                    st.info(f"{t('answer_generated_in')} {elapsed_time:.2f} {t('seconds')}" + 
                           (f" ({t('from_cache')})" if elapsed_time < 1.0 and use_cache else ""))
                    
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"{t('error')}: {str(e)}")
                    logger.error(f"{t('query_error')}: {str(e)}")

    with query_col2:
        # Koleksiyon seçimi
        with st.expander(t("query_settings"), expanded=False):
            collections = vector_db.list_collections()
            if collections:
                selected_collection = st.selectbox(
                    t("collection_to_query"), 
                    collections,
                    index=collections.index(collection_name) if collection_name in collections else 0,
                    key="sorgu_koleksiyon_secim"
                )
                
                # Koleksiyon değiştiyse, session state'i güncelle
                if selected_collection != collection_name:
                    collection_name = selected_collection
                
                # Koleksiyon metadata bilgilerini göster
                try:
                    metadata = load_collection_metadata(collection_name)
                    if metadata:
                        embedding_type = metadata.get("embedding_type", "")
                        embedding_model = metadata.get("embedding_model", "")
                        if embedding_type and embedding_model:
                            st.info(f"{t('collection_created_with')} {embedding_type} - {embedding_model}. "\
                                    f"{t('using_same_model')}")
                except:
                    pass
            else:
                st.warning(t("no_collections"))
        
        # Bilgi kutusu - eğer henüz soru sorulmadıysa
        if 'answer' not in st.session_state or not st.session_state['answer']:
            st.markdown(f"""
            <div style="background-color: #1a2340; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #3F51B5;">
                <h4 style="color: #8c9eff; margin-top: 0;">{t('query_tips_title')}</h4>
                <p>• {t('query_tip_1')}</p>
                <p>• {t('query_tip_2')}</p>
                <p>• {t('query_tip_3')}</p>
                <p>• {t('query_tip_4')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Yanıt alanı - tüm genişlikte
    st.markdown('<div style="height: 20px"></div>', unsafe_allow_html=True)  # Boşluk ekle
    
    # Yanıt gösterimi
    if 'answer' in st.session_state and st.session_state['answer']:
        # Yanıtı göster
        st.markdown(f'<h3 style="color:#2196F3;">{t("answer_title")}</h3>', unsafe_allow_html=True)
        st.markdown(f'<div style="background-color: #1a1a1a; padding: 20px; border-radius: 5px; border-left: 4px solid #2196F3;">{st.session_state["answer"]}</div>', unsafe_allow_html=True)
        
        # Kaynak belgeleri göster
        st.markdown(f'<h3 style="color:#2196F3; margin-top: 20px;">{t("source_documents")}</h3>', unsafe_allow_html=True)
        
        if 'source_docs' in st.session_state and st.session_state['source_docs']:
            for i, doc in enumerate(st.session_state['source_docs']):
                with st.expander(f"{t('source')} {i+1}: {doc.metadata.get('filename', t('unknown'))} - {t('page')} {doc.metadata.get('page', t('unknown'))}"):
                    # Sol ve sağ sütunları oluştur
                    source_col1, source_col2 = st.columns([3, 1])
                    
                    with source_col1:
                        st.markdown(f"**{t('content')}:**")
                        st.markdown(f"{doc.page_content}")
                    
                    with source_col2:
                        st.markdown("**Metadata:**")
                        for key, value in doc.metadata.items():
                            st.markdown(f"**{key}:** {value}")
        else:
            st.info(t("no_source_docs"))

# 4. Koleksiyon Yönetimi Sekmesi
with tab4:
    st.header(t("collections_title"))
    st.markdown(t("collections_description"))
    
    # Koleksiyonlari listele ve yönet
    collections = vector_db.list_collections()
    
    if collections:
        # Koleksiyon bilgilerini grid şeklinde görüntüle
        st.markdown(f"### {t('available_collections')} ({len(collections)})")
        
        # Koleksiyonları 3 sütunlu düzende göster
        cols = st.columns(3)
        
        for i, collection_name in enumerate(collections):
            col = cols[i % 3]
            
            with col:
                # Kart tarzında koleksiyon gösterimi
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #1a273a; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #2196F3;">
                        <h4 style="color: #42a5f5; margin-top: 0;">{collection_name}</h4>
                    """, unsafe_allow_html=True)
                    
                    # Metadata bilgilerini al (varsa)
                    try:
                        metadata = load_collection_metadata(collection_name)
                        if metadata:
                            num_docs = metadata.get("num_documents", "?")
                            embedding_type = metadata.get("embedding_type", "?")
                            embedding_model = metadata.get("embedding_model", "?")
                            
                            # Metadata bilgilerini göster
                            st.markdown(f"""
                            <p><strong>{t('document_count')}:</strong> {num_docs}</p>
                            <p><strong>{t('embedding')}:</strong> {embedding_type} - {embedding_model}</p>
                            """, unsafe_allow_html=True)
                    except:
                        st.markdown(f"<p>{t('no_metadata')}</p>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Koleksiyon işlemleri
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # İçeriğini sorgula butonu
                        if st.button(f"🔍 {t('query_button')}", key=f"query_{collection_name}", use_container_width=True):
                            # Sorgu sekmesine geç ve bu koleksiyonu seç
                            st.session_state["selected_collection"] = collection_name
                            st.session_state["redirect_to_query"] = True
                            # Sayfayı yeniden yükle
                            st.rerun()
                    
                    with col2:
                        # Sil butonu
                        if st.button(f"🗑️ {t('delete_button')}", key=f"delete_{collection_name}", use_container_width=True):
                            # Silme onayını bir dialog olarak göster
                            st.session_state["delete_confirm"] = collection_name
                            st.rerun()
        
        # Silme onayı göster (varsa)
        if "delete_confirm" in st.session_state and st.session_state["delete_confirm"]:
            collection_to_delete = st.session_state["delete_confirm"]
            
            # Silme onay dialogu
            with st.expander(f"⚠️ {t('delete_confirmation')}: {collection_to_delete}", expanded=True):
                st.warning(f"{t('delete_warning')} '{collection_to_delete}'?")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ {t('confirm_delete')}", use_container_width=True):
                        # Koleksiyonu sil
                        try:
                            vector_db.delete_collection(collection_to_delete)
                            st.session_state["delete_confirm"] = None
                            # Koleksiyon silindi mesajı
                            st.success(f"'{collection_to_delete}' {t('collection_deleted')}")
                            # Sayfa yeniden yükle
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"{t('delete_error')}: {str(e)}")
                
                with col2:
                    if st.button(f"❌ {t('cancel_delete')}", use_container_width=True):
                        # İptal et
                        st.session_state["delete_confirm"] = None
                        st.rerun()
    else:
        # Koleksiyon yoksa, bilgi mesajı göster
        st.warning(t("no_collections_warning"))
        # Yükleme bilgisi
        st.markdown(f"""
        <div style="background-color: #1a2340; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #3F51B5;">
            <h4 style="color: #8c9eff; margin-top: 0;">{t('add_document_title')}</h4>
            <p>{t('add_document_instruction')}</p>
        </div>
        """, unsafe_allow_html=True)

# 5. İstatistikler Sekmesi
with tab5:
    st.header(t("stats_title"))
    st.markdown(t("stats_description"))
    
    # İstatistikleri iki sütunlu göster
    col1, col2 = st.columns(2)
    
    # Sol taraf: Genel istatistikler
    with col1:
        st.subheader(t("general_stats"))
        
        # Koleksiyon sayısı
        collections = vector_db.list_collections()
        total_collections = len(collections)
        
        # Toplam doküman sayısını ve vektör boyutunu hesapla
        total_documents = 0
        total_vectors = 0
        embedding_models = set()
        
        for collection_name in collections:
            try:
                metadata = load_collection_metadata(collection_name)
                if metadata:
                    total_documents += metadata.get("num_documents", 0)
                    total_vectors += metadata.get("num_vectors", 0)
                    if "embedding_model" in metadata:
                        embedding_models.add(metadata["embedding_model"])
            except:
                pass
                
        # İstatistikleri metriklerde göster
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric(t("collection_count"), total_collections)
            st.metric(t("document_count"), total_documents)
        
        with metrics_col2:
            st.metric(t("vector_count"), total_vectors)
            st.metric(t("embedding_model_count"), len(embedding_models))
        
        # Kullanılan embedding modellerini göster
        if embedding_models:
            st.subheader(t("embedding_models_used"))
            for model in embedding_models:
                st.markdown(f"- `{model}`")
        
    # Sağ taraf: Koleksiyon detayları
    with col2:
        st.subheader(t("collection_details"))
        
        if collections:
            # Koleksiyon seçimi
            selected_collection = st.selectbox(
                t("select_collection_for_stats"), 
                collections,
                key="collection_stats_select"
            )
            
            # Seçilen koleksiyonun detayları
            try:
                metadata = load_collection_metadata(selected_collection)
                if metadata:
                    # Expandable detaylar
                    with st.expander(f"{selected_collection} {t('details')}", expanded=True):
                        # Temel bilgiler
                        created_date = metadata.get('created_date', t('unknown'))
                        num_docs = metadata.get('num_documents', 0)
                        num_vectors = metadata.get('num_vectors', 0)
                        chunk_size = metadata.get('chunk_size', t('unknown'))
                        chunk_overlap = metadata.get('chunk_overlap', t('unknown'))
                        embedding_type = metadata.get('embedding_type', t('unknown'))
                        embedding_model = metadata.get('embedding_model', t('unknown'))
                        
                        # Metadata bilgilerini göster
                        st.markdown(f"**{t('created_date')}:** {created_date}")
                        st.markdown(f"**{t('document_count')}:** {num_docs}")
                        st.markdown(f"**{t('vector_count')}:** {num_vectors}")
                        
                        # Embedding bilgileri
                        st.markdown(f"**{t('embedding_provider')}:** {embedding_type}")
                        st.markdown(f"**{t('embedding_model')}:** {embedding_model}")
                        
                        # Bölümleme bilgileri
                        st.markdown(f"**{t('chunk_size')}:** {chunk_size}")
                        st.markdown(f"**{t('chunk_overlap')}:** {chunk_overlap}")
                        
                        # Doküman bilgileri (varsa)
                        if "documents" in metadata:
                            st.subheader(t("documents_in_collection"))
                            for doc in metadata["documents"]:
                                st.markdown(f"- {doc.get('filename', t('unknown'))}")
            except Exception as e:
                st.error(f"{t('error_loading_metadata')}: {str(e)}")
        else:
            st.info(t("no_collections_for_stats"))

# Footer
st.markdown("---")
st.markdown("BilgiÇekirdeği © 2025") 