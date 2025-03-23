"""
BilgiÇekirdeği Streamlit Arayüzü
-------------------------------
Bu modül, BilgiÇekirdeği uygulaması için bir web arayüzü sunar.
PDF dokümanlarını yükleme, indeksleme ve sorgu yapma işlevleri içerir.
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

# BilgiÇekirdeği modülleri
from vectorstore.vector_db import VectorDatabase
from ingestion.load_pdf import load_pdf
from qa.rag_chain import RAGChain
from utils.logging_config import setup_logging, get_logger
from load_pdf import load_pdf_document
from run_query import run_query, clear_query_cache, clear_rag_cache

# Loglama yapılandırmasını etkinleştir
setup_logging()
logger = get_logger(__name__)

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
    Koleksiyon metadata bilgilerini yükler.
    
    Args:
        collection_name: Koleksiyon adı
        
    Returns:
        dict: Metadata bilgileri veya boş sözlük
    """
    try:
        metadata_file = os.path.join("./indices", collection_name, "metadata", "collection_info.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Metadata yüklenirken hata: {str(e)}")
        return {}

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

# Başlık ve açıklama
st.markdown('<h1 class="main-header">BilgiÇekirdeği</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">PDF dokümanlarını yükleyin, indeksleyin ve sorular sorun. BilgiÇekirdeği, yapay zeka ile dokümanlarınızdaki bilgiye erişmenizi sağlar.</p>', unsafe_allow_html=True)

# Arka planda Ollama modellerini yükle
if 'ollama_models' not in st.session_state:
    st.session_state['ollama_models'] = get_ollama_models()

# Yan menü
with st.sidebar:
    st.markdown('<h2 style="color:#2196F3; font-weight:600;">BilgiÇekirdeği</h2>', unsafe_allow_html=True)
    
    # Logo için geçici bir çözüm
    try:
        st.image("./static/logo.png", width=150)
    except:
        st.markdown('<div style="text-align:center; font-size:3.5rem; margin-bottom:20px;">🧠</div>', unsafe_allow_html=True)
    
    st.markdown('<hr style="margin-top:0;">', unsafe_allow_html=True)
    
    # Ayarlar bölümü
    st.markdown('<h3 style="color:#e0e0e0; font-weight:600; font-size:1.3rem;">⚙️ Ayarlar</h3>', unsafe_allow_html=True)
    
    # Koleksiyon adı
    st.markdown('<p style="font-weight:500; margin-bottom:5px;">Koleksiyon Adı</p>', unsafe_allow_html=True)
    collection_name = st.text_input(
        label="Koleksiyon Adı",
        value=DEFAULT_COLLECTION, 
        key="collection_name_input", 
        label_visibility="collapsed"
    )
    
    # LLM Ayarları
    with st.expander("🤖 LLM Ayarları", expanded=True):
        llm_provider = st.selectbox(
            "LLM Sağlayıcı", 
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
                    "Ollama Model", 
                    options=ollama_models,
                    index=ollama_models.index(LLM_MODEL) if LLM_MODEL in ollama_models else 0,
                    help="Ollama üzerinde yüklü olan modellerden birini seçin",
                    key="ollama_model_select"
                )
            else:
                llm_model = st.text_input(
                    "Ollama Model", 
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
                "OpenAI Model", 
                ["gpt-3.5-turbo", "gpt-4"], 
                index=0,
                help="OpenAI'nin üretimde olan modellerinden birini seçin",
                key="openai_model_select"
            )
            openai_api_key = st.text_input(
                "OpenAI API Anahtarı", 
                type="password",
                key="openai_api_key_input"
            )
    
    # Embedding Ayarları
    with st.expander("🧬 Embedding Ayarları", expanded=True):
        embedding_provider = st.selectbox(
            "Embedding Sağlayıcı", 
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
    st.markdown('<div class="info-box"><strong>⚠️ Önemli Not:</strong> Vektör veritabanı oluştururken kullandığınız embedding modeli ile sorgu yaparken aynı modeli kullanmanız gerekir. Aksi halde boyut uyuşmazlığı hatası alırsınız.</div>', unsafe_allow_html=True)
    
    # Koleksiyon bilgileri
    st.markdown('<h3 style="color:#424242; font-weight:600; font-size:1.3rem; margin-top:20px;">📚 Koleksiyonlar</h3>', unsafe_allow_html=True)
    
    # İndeks bilgilerini göster
    vector_db = VectorDatabase()
    try:
        collections = vector_db.list_collections()
        if collections:
            st.success(f"{len(collections)} koleksiyon bulundu")
            for collection in collections:
                # Metadata'yı yükle
                metadata = load_collection_metadata(collection)
                if metadata:
                    model_info = f"({metadata.get('embedding_provider', '')}/{metadata.get('embedding_model', '')})"
                    st.markdown(f'<div class="collection-card"><strong>{collection}</strong><br/><small>{model_info}</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="collection-card"><strong>{collection}</strong></div>', unsafe_allow_html=True)
        else:
            st.warning("Henüz bir koleksiyon oluşturulmamış")
    except Exception as e:
        st.error(f"Koleksiyonlar yüklenirken hata: {str(e)}")

# Ana sekmeleri oluştur
tab1, tab2, tab3, tab4 = st.tabs(["📄 Doküman Yükleme", "❓ Soru Sorma", "📋 Koleksiyon İçeriği", "📊 Koleksiyon İstatistikleri"])

# 1. Doküman Yükleme Sekmesi
with tab1:
    st.header("PDF Dokümanı Yükle")
    
    # İki sütunlu düzen
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Kullanıcı bilgi kartı
        st.markdown("""
        <div style="background-color: #1a273a; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #2196F3;">
            <h4 style="color: #42a5f5; margin-top: 0;">PDF Dokümanı Nasıl Yüklenir?</h4>
            <p>1. Yüklemek istediğiniz PDF dosyasını seçin</p>
            <p>2. "Yükle" butonuna tıklayın</p>
            <p>3. Yüklenen dosyayı vektör veritabanına ekleyin</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "PDF Yükle", 
            type="pdf",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            # Geçici dosyayı kaydet ve işle
            file_path = f"temp_{int(time.time())}.pdf"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"'{uploaded_file.name}' başarıyla yüklendi! Şimdi vektör veritabanına ekleyebilirsiniz.")
            
            process_col1, process_col2 = st.columns([1, 1])
            
            with process_col1:
                # Yükleme düğmesi
                process_button = st.button(
                    "PDF'yi İşle ve Ekle", 
                    use_container_width=True,
                    key="isle_button"
                )
            with process_col2:
                # İptal düğmesi
                cancel_button = st.button(
                    "İptal", 
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
                            # Temporary file temizliği
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        else:
                            st.error("PDF işlenirken bir hata oluştu.")
                    except Exception as e:
                        st.error(f"Hata: {str(e)}")
                        logger.error(f"PDF yüklenirken hata: {str(e)}")
            
            elif cancel_button:
                # Temporary file temizliği
                if os.path.exists(file_path):
                    os.remove(file_path)
                st.info("İşlem iptal edildi.")
                st.rerun()
    
    with col2:
        # İşlem sonuçları ve bilgiler burada gösterilecek
        if uploaded_file is None:
            st.markdown("""
            <div style="background-color: #2c2c00; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #FFD600;">
                <h4 style="color: #FFD600; margin-top: 0;">BilgiÇekirdeği Nasıl Çalışır?</h4>
                <p>1. Yüklenen PDF dosyaları küçük parçalara bölünür</p>
                <p>2. Her parça vektör temsillere dönüştürülür</p>
                <p>3. Vektörler veritabanında saklanır</p>
                <p>4. Sorularınız benzer şekilde vektörlere dönüştürülür</p>
                <p>5. En ilgili doküman parçaları bulunur</p>
                <p>6. Yapay zeka doküman parçalarını kullanarak yanıt üretir</p>
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

# 2. Soru Sorma Sekmesi
with tab2:
    st.header("Dokümanlara Soru Sor")
    
    # İki sütunlu düzen
    query_col1, query_col2 = st.columns([3, 2])
    
    with query_col1:
        # Sorgu alanı
        st.markdown('<p style="font-weight:500; margin-bottom:5px;">Sorunuzu Girin</p>', unsafe_allow_html=True)
        
        # Session state'teki sorguyu başlangıçta al
        if 'query' not in st.session_state:
            st.session_state['query'] = ""
            
        # Sorgu değiştiğinde bu fonksiyon çalışacak
        def on_query_change():
            st.session_state['current_query'] = st.session_state.soru_input
        
        # Sorgu alanı
        query = st.text_area(
            label="Soru",
            value=st.session_state.get('query', ''),
            height=120, 
            placeholder="Dokümanlarınıza sormak istediğiniz soruyu buraya yazın...", 
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
                "🔍 Soru Sor", 
                use_container_width=True, 
                type="primary",
                key="soru_sor_button"
            )
        
        with button_col2:
            # Temizle düğmesi
            clear_button = st.button(
                "🧹 Temizle", 
                use_container_width=True,
                key="temizle_button"
            )
            
        with button_col3:
            # Örnek soru düğmesi
            example_button = st.button(
                "📝 Örnek Soru", 
                use_container_width=True,
                key="ornek_soru_button"
            )
            
        # Eğer örnek soru istenirse
        if example_button:
            example_questions = [
                "Bu dokümanda bahsedilen en önemli konular nelerdir?",
                "İK süreçlerinin online olarak yönetilmesi için neler yapılmalıdır?",
                "Doğru adayı bulmak için hangi stratejiler önerilmiştir?",
                "GDPR düzenlemeleri hakkında ne söyleniyor?",
                "Matrisler ve yetkinlik matrisleri nasıl hazırlanır?"
            ]
            import random
            selected_query = random.choice(example_questions)
            # Burada doğrudan session state'e atıyoruz ve sayfayı yeniliyoruz
            st.session_state['query'] = selected_query
            st.rerun()
            
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
            from run_query import clear_query_cache, clear_rag_cache

            # Gelişmiş ayarlara bakalım
            with st.expander("Önbellek Ayarları", expanded=False):
                use_cache = st.checkbox("Önbelleklemeyi Kullan", value=True, 
                                        help="Aynı soruların daha hızlı yanıtlanması için önbellek kullan")
                if st.button("Önbelleği Temizle"):
                    clear_query_cache()
                    clear_rag_cache()
                    # Session state'ten de yanıtları temizle
                    if 'answer' in st.session_state:
                        del st.session_state['answer']
                    if 'source_docs' in st.session_state:
                        del st.session_state['source_docs']
                    st.success("Tüm önbellekler temizlendi!")
                    st.info("Yanıtlar temizlendi. Yeni bir sorgu yapabilirsiniz.")
            
            # İlerleme göstergesi başlat
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
            
            # Zamanlayıcı başlat
            start_time = time.time()
            
            with st.spinner("Yanıt oluşturuluyor... Bu işlem sistemin yüküne bağlı olarak 10-30 saniye sürebilir."):
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
                    
                    # Sorgunun en güncel halini kullan
                    final_query = st.session_state.get('current_query', current_query)
                    
                    # Sorguyu çalıştır
                    answer, source_docs = run_query(
                        query=final_query,
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        temperature=temperature,
                        top_k=top_k,
                        collection_name=collection_name,
                        openai_api_key=openai_api_key if embedding_provider == "openai" or llm_provider == "openai" else None,
                        use_cache=use_cache
                    )
                    
                    # İşlem süresini hesapla
                    elapsed_time = time.time() - start_time
                    
                    # Progress barı kaldır
                    progress_placeholder.empty()
                    
                    # Session state'e kaydet
                    st.session_state['answer'] = answer
                    st.session_state['source_docs'] = source_docs
                    
                    # Süre bilgisini göster
                    st.info(f"Yanıt {elapsed_time:.2f} saniyede oluşturuldu" + 
                           (" (önbellekten)" if elapsed_time < 1.0 and use_cache else ""))
                    
                except Exception as e:
                    progress_placeholder.empty()
                    st.error(f"Hata: {str(e)}")
                    logger.error(f"Sorgu çalıştırılırken hata: {str(e)}")

    with query_col2:
        # Koleksiyon seçimi
        with st.expander("Sorgu Ayarları", expanded=False):
            collections = vector_db.list_collections()
            if collections:
                selected_collection = st.selectbox(
                    "Sorgulanacak Koleksiyon", 
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
                            st.info(f"Bu koleksiyon {embedding_type} - {embedding_model} ile oluşturulmuş. "\
                                    f"Sorgu yaparken uyumsuzluk hatalarını önlemek için otomatik olarak aynı model kullanılacak.")
                except:
                    pass
            else:
                st.warning("Henüz hiç koleksiyon bulunmuyor. Lütfen önce bir doküman yükleyin.")
        
        # Bilgi kutusu - eğer henüz soru sorulmadıysa
        if 'answer' not in st.session_state or not st.session_state['answer']:
            st.markdown("""
            <div style="background-color: #1a2340; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #3F51B5;">
                <h4 style="color: #8c9eff; margin-top: 0;">Sorgu İpuçları</h4>
                <p>• Sorunuzu açık ve net bir şekilde ifade edin</p>
                <p>• Sorularınızı tam cümleler halinde sorun</p>
                <p>• Aşırı uzun sorular yerine birden fazla kısa soru sorun</p>
                <p>• Yanıtın belirli bir formatta olmasını istiyorsanız belirtin</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Yanıt alanı - tüm genişlikte
    st.markdown('<div style="height: 20px"></div>', unsafe_allow_html=True)  # Boşluk ekle
    
    # Yanıt gösterimi
    if 'answer' in st.session_state and st.session_state['answer']:
        # Yanıtı göster
        st.markdown('<h3 style="color:#2196F3;">Yanıt</h3>', unsafe_allow_html=True)
        st.markdown(f'<div style="background-color: #1a1a1a; padding: 20px; border-radius: 5px; border-left: 4px solid #2196F3;">{st.session_state["answer"]}</div>', unsafe_allow_html=True)
        
        # Kaynak belgeleri göster
        st.markdown('<h3 style="color:#2196F3; margin-top: 20px;">Kaynak Belgeler</h3>', unsafe_allow_html=True)
        
        if 'source_docs' in st.session_state and st.session_state['source_docs']:
            for i, doc in enumerate(st.session_state['source_docs']):
                with st.expander(f"Kaynak {i+1}: {doc.metadata.get('filename', 'Bilinmeyen')} - Sayfa {doc.metadata.get('page', 'Bilinmeyen')}"):
                    # Sol ve sağ sütunları oluştur
                    source_col1, source_col2 = st.columns([3, 1])
                    
                    with source_col1:
                        st.markdown(f"**İçerik:**")
                        st.markdown(f"{doc.page_content}")
                    
                    with source_col2:
                        st.markdown("**Metadata:**")
                        for key, value in doc.metadata.items():
                            st.markdown(f"**{key}:** {value}")
        else:
            st.info("Bu sorgu için kaynak belge bulunamadı.")

# 3. Koleksiyon İçeriği Sekmesi
with tab3:
    st.header("Koleksiyon İçeriği")
    
    # Koleksiyon seçimi
    try:
        vector_db = VectorDatabase()
        collections = vector_db.list_collections()
        
        if not collections:
            st.info("Henüz hiç koleksiyon bulunamadı.")
        else:
            selected_collection = st.selectbox(
                "Koleksiyon Seçiniz", 
                collections,
                index=collections.index(collection_name) if collection_name in collections else 0,
                key="koleksiyon_secim"
            )
            
            if st.button("Koleksiyon İçeriğini Göster", key="koleksiyon_goster"):
                with st.spinner("Koleksiyon içeriği alınıyor..."):
                    try:
                        # Koleksiyonu yükle
                        vector_db.load_collection(selected_collection)
                        
                        if vector_db.vector_store is None:
                            st.error(f"'{selected_collection}' koleksiyonu yüklenirken hata oluştu.")
                        else:
                            try:
                                # Koleksiyondaki tüm dokümanları al (en fazla 100 doküman)
                                # Dummy embeddings sorunu için geçici çözüm
                                from langchain_core.embeddings import Embeddings
                                class FixedDummyEmbeddings(Embeddings):
                                    def __init__(self, dim: int = 1536):
                                        self.dim = dim
                                    
                                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                                        return [[0.1] * self.dim for _ in texts]
                                    
                                    def embed_query(self, text: str) -> List[float]:
                                        return [0.1] * self.dim
                                
                                # MetadataFiltering kullanarak dokümanları getirme
                                if hasattr(vector_db.vector_store, "metadata_field_info"):
                                    # Koleksiyon bilgilerini al
                                    collection_metadata = load_collection_metadata(selected_collection)
                                    embedding_dim = collection_metadata.get("embedding_dimension", 1536)
                                    
                                    # Mevcut embeddings modelini değiştir
                                    if hasattr(vector_db.vector_store, 'embedding_function'):
                                        original_embed_function = vector_db.vector_store.embedding_function
                                        # Embeddings fonksiyonu çalışmıyorsa sabit bir fonksiyon kullan
                                        vector_db.vector_store.embedding_function = FixedDummyEmbeddings(dim=embedding_dim)
                                
                                # Tüm dokümanları getirmeye çalış
                                try:
                                    docs = vector_db.vector_store.similarity_search("", k=100)
                                except Exception as search_error:
                                    logger.warning(f"Benzerlik aramasında hata: {str(search_error)}")
                                    # Alternatif yöntem dene
                                    try:
                                        # Direkt VectorStore'un içindeki dokümanları al
                                        if hasattr(vector_db.vector_store, "docstore"):
                                            docstore_docs = list(vector_db.vector_store.docstore._dict.values())
                                            if docstore_docs:
                                                docs = docstore_docs[:100]  # En fazla 100 doküman
                                            else:
                                                docs = []
                                        else:
                                            # Boş bir query ile getir
                                            docs = vector_db.vector_store.similarity_search(" ", k=100)
                                    except Exception as alt_error:
                                        logger.error(f"Alternatif doküman almada da hata: {str(alt_error)}")
                                        st.error("Dokümanlar alınamadı. Embedding modeli uyumsuzluğu olabilir.")
                                        st.info("Koleksiyon bilgileri başarıyla yüklendi, ancak içeriği görmek mümkün olmadı.")
                                        
                                        # Koleksiyon metadata dosyasını göster
                                        st.subheader("Koleksiyon Metadata Dosyaları")
                                        metadata_path = os.path.join("./indices", selected_collection, "metadata", "collection_info.json")
                                        if os.path.exists(metadata_path):
                                            with open(metadata_path, 'r') as f:
                                                metadata_json = json.load(f)
                                                st.json(metadata_json)
                                        else:
                                            st.warning("Metadata dosyası bulunamadı.")
                                        
                                        # Orijinal embedding fonksiyonunu geri yükle
                                        if 'original_embed_function' in locals() and hasattr(vector_db.vector_store, 'embedding_function'):
                                            vector_db.vector_store.embedding_function = original_embed_function
                                        
                                        # Koleksiyon metadata bilgilerini göster
                                        docs = []  # Boş liste döndür
                                
                                # Orijinal embedding fonksiyonunu geri yükle
                                if 'original_embed_function' in locals() and hasattr(vector_db.vector_store, 'embedding_function'):
                                    vector_db.vector_store.embedding_function = original_embed_function
                                
                                if not docs:
                                    st.warning(f"'{selected_collection}' koleksiyonunda doküman bulunamadı.")
                                else:
                                    st.success(f"{len(docs)} doküman bulundu.")
                                    
                                    # Dokümanları göster
                                    doc_data = []
                                    for i, doc in enumerate(docs):
                                        filename = doc.metadata.get('filename', 'Bilinmiyor')
                                        page = doc.metadata.get('page', 'Bilinmiyor')
                                        filetype = doc.metadata.get('filetype', 'Bilinmiyor')
                                        source = doc.metadata.get('source', 'Bilinmiyor')
                                        
                                        # Doküman içeriğinin ilk 100 karakteri
                                        content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                                        
                                        doc_data.append({
                                            "ID": i+1,
                                            "Dosya": filename,
                                            "Sayfa": page,
                                            "Tür": filetype,
                                            "İçerik Önizleme": content
                                        })
                                    
                                    # DataFrame oluştur ve göster
                                    df = pd.DataFrame(doc_data)
                                    st.dataframe(df, use_container_width=True)
                                    
                                    # CSV olarak indirme butonu
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        "CSV Olarak İndir",
                                        csv,
                                        f"{selected_collection}_içerik.csv",
                                        "text/csv",
                                        key='download-csv'
                                    )
                            except Exception as e:
                                st.error(f"Dokümanlar alınırken hata: {str(e)}")
                                logger.error(f"Dokümanlar alınırken hata: {str(e)}")
                                st.info("Koleksiyon yüklendi ancak içeriği görüntülenemiyor. Embedding modeli uyumsuzluğu olabilir.")
                                
                                # Koleksiyon metadata bilgilerini göster
                                st.subheader("Koleksiyon Metadata Dosyaları")
                                try:
                                    metadata_path = os.path.join("./indices", selected_collection, "metadata", "collection_info.json")
                                    if os.path.exists(metadata_path):
                                        with open(metadata_path, 'r') as f:
                                            metadata_json = json.load(f)
                                            st.json(metadata_json)
                                    else:
                                        st.warning("Metadata dosyası bulunamadı.")
                                except Exception as meta_error:
                                    st.warning(f"Metadata okuma hatası: {str(meta_error)}")
                    except Exception as e:
                        st.error(f"Koleksiyon yüklenirken hata: {str(e)}")
                        logger.error(f"Koleksiyon yüklenirken hata: {str(e)}")
    except Exception as e:
        st.error(f"Koleksiyon listesi alınırken hata: {str(e)}")
        logger.error(f"Koleksiyon listesi alınırken hata: {str(e)}")
        st.info("Bu hata genellikle henüz hiç koleksiyon oluşturulmadığında görülür. Lütfen önce bir PDF yükleyin.")
    
    # Koleksiyon Metadata Bölümü
    st.subheader("Koleksiyon Metadata Dosyaları")
    
    try:
        metadata = load_collection_metadata(selected_collection if 'selected_collection' in locals() else collection_name)
        if metadata:
            st.json(metadata)
        else:
            st.info("Bu koleksiyon için metadata bilgisi bulunamadı.")
    except:
        st.info("Koleksiyon metadata bilgisi yüklenemedi.")

# 4. Koleksiyon İstatistikleri Sekmesi
with tab4:
    st.header("Koleksiyon İstatistikleri")
    
    # Yenileme düğmesi
    if st.button("İstatistikleri Yenile"):
        with st.spinner("Koleksiyon istatistikleri alınıyor..."):
            try:
                # VectorDatabase örneği oluştur
                vector_db = VectorDatabase()
                
                # Koleksiyonları listele
                collections = []
                for item in os.listdir(vector_db.base_dir):
                    item_path = os.path.join(vector_db.base_dir, item)
                    if os.path.isdir(item_path):
                        collections.append(item)
                
                if not collections:
                    st.info("Henüz hiç koleksiyon bulunamadı.")
                else:
                    # Her koleksiyon için istatistikleri göster
                    for collection in collections:
                        st.subheader(f"Koleksiyon: {collection}")
                        
                        try:
                            # İstatistikleri al
                            stats = vector_db.get_collection_stats(collection)
                            
                            # İki sütunlu düzen
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Temel Bilgiler**")
                                st.write(f"Doküman Sayısı: {stats.get('document_count', 'Bilinmiyor')}")
                                st.write(f"Depolama Tipi: {stats.get('storage_type', 'Bilinmiyor')}")
                                st.write(f"İndeks Yolu: {stats.get('index_path', 'Bilinmiyor')}")
                            
                            # Metadata varsa göster
                            if "metadata" in stats:
                                with col2:
                                    st.write("**Metadata Bilgileri**")
                                    metadata = stats["metadata"]
                                    
                                    if "embedding_type" in metadata:
                                        st.write(f"Embedding Tipi: {metadata['embedding_type']}")
                                    
                                    if "embedding_model" in metadata:
                                        st.write(f"Embedding Modeli: {metadata['embedding_model']}")
                                    
                                    if "embedding_dimension" in metadata and metadata["embedding_dimension"]:
                                        st.write(f"Embedding Boyutu: {metadata['embedding_dimension']}")
                                    
                                    if "created_at" in metadata:
                                        st.write(f"Oluşturulma Tarihi: {metadata['created_at']}")
                                
                            # JSON görünümü
                            with st.expander("Tüm İstatistikleri JSON Olarak Göster"):
                                st.json(stats)
                        
                        except Exception as e:
                            st.error(f"'{collection}' koleksiyonu için istatistikler alınırken hata: {str(e)}")
                            logger.error(f"Koleksiyon istatistikleri alınırken hata: {str(e)}")
            
            except Exception as e:
                st.error(f"Koleksiyon istatistikleri alınırken hata: {str(e)}")
                logger.error(f"Koleksiyon istatistikleri alınırken hata: {str(e)}")

# Footer
st.markdown("---")
st.markdown("BilgiÇekirdeği © 2025") 