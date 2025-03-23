# BilgiÇekirdeği (Knowledge Kernel) - Ürün Gereksinimleri Dokümanı

## 1. Giriş

### 1.1 Amaç

BilgiÇekirdeği projesi, kullanıcıların kişisel veya kurumsal dokümanlarını (PDF, notlar, CV'ler, vb.) vektör veritabanına yükleyerek yapay zeka ile sorgulanabilir hale getirmeyi amaçlamaktadır. Proje, kullanıcıların dokümanlarındaki bilgilere doğal dil ile sorgular yaparak hızlıca erişmelerini sağlayacaktır.

### 1.2 Kapsam

Bu PRD, BilgiÇekirdeği projesinin geliştirilmesi için gerekli gereksinimleri, özellikleri ve teknik detayları içermektedir. Belge, ürünün geliştirilmesi sürecinde bir rehber olarak kullanılacaktır.

### 1.3 Motivasyon

Modern dünyada, kişiler ve kurumlar giderek artan miktarda dijital dokümana sahiptir. Bu dokümanlar içindeki bilgilere erişmek, özellikle çok sayıda doküman söz konusu olduğunda, zor ve zaman alıcı olabilir. Bu proje:

- Dokümanlar arasında hızlı ve doğru bilgi erişimi sağlayacak
- Kullanıcıların bilgi tabanlarını daha etkili kullanmalarına olanak tanıyacak
- Yapay zeka teknolojileri ile bilgi erişimini demokratikleştirecek
- Kişisel veya kurumsal bilgi yönetimini kolaylaştıracak

## 2. Ürün Özellikleri

### 2.1 Temel Özellikler

#### 2.1.1 Doküman Yükleme ve İşleme
- PDF, TXT, DOCX formatlarında doküman desteği
- Dokümanları anlamlı parçalara bölme (chunking)
- Metadata çıkarımı ve saklama

#### 2.1.2 Vektörleştirme ve Depolama
- Doküman parçalarını vektörleştirme (embedding)
- Vektörleri FAISS veya Pinecone veritabanında saklama
- Vektör indeksi oluşturma ve yönetme

#### 2.1.3 Sorgulama ve Yanıt Üretimi
- Doğal dil ile sorgulama arayüzü
- İlgili doküman parçalarını belirleme ve çıkarma
- Yanıt sentezleme ve kaynak bilgisi sağlama

### 2.2 İleri Aşama Özellikler (Sprint 2+)

#### 2.2.1 Kullanıcı Arayüzü
- Web tabanlı kullanıcı arayüzü (Streamlit veya Gradio ile)
- Doküman yükleme, görüntüleme ve sorgulama için görsel arayüz
- Kullanıcı oturumları ve ayarlar

#### 2.2.2 Gelişmiş İşleme Özellikleri
- Çoklu dil desteği
- Görsel içerikten metin çıkarımı (OCR)
- Ses dosyalarından metin çıkarımı

#### 2.2.3 Kategorilendirme ve Organizasyon
- Dokümanları otomatik kategorilendirme
- Etiketleme ve organizasyon araçları
- İlişkili dokümanları önerme

## 3. Teknik Gereksinimler

### 3.1 Mimari

BilgiÇekirdeği, modüler bir mimariye sahip olacak ve aşağıdaki ana bileşenleri içerecektir:

1. **Doküman İşleme Modülü**: Dokümanları yükleme, işleme ve parçalara ayırma
2. **Vektörleştirme Modülü**: Doküman parçalarını vektörlere dönüştürme
3. **Vektör Veritabanı Modülü**: Vektörleri depolama ve sorgulama
4. **QA Zinciri Modülü**: Sorguları işleme ve yanıtları üretme
5. **CLI/API Arayüzü**: Kullanıcı etkileşimi için arayüz

### 3.2 Teknoloji Yığını

#### 3.2.1 Programlama Dili ve Çerçeveler
- Python 3.8+
- LangChain (RAG uygulamaları için)
- Streamlit/Gradio (UI için, ileri aşamalarda)

#### 3.2.2 Yapay Zeka ve Vektör İşleme
- OpenAI API (veya alternatif LLM'ler)
- OpenAI Embeddings veya HuggingFace'in InstructorEmbedding
- FAISS (yerel vektör veritabanı) veya Pinecone (bulut tabanlı)

#### 3.2.3 Depolama ve Veri Yönetimi
- Yerel dosya sistemi (vektör indeksleri için)
- SQLite veya JSON (metadata depolama için)

### 3.3 Güvenlik Gereksinimleri
- API anahtarlarının güvenli yönetimi (.env dosyaları)
- Yerel depolama için dosya güvenliği
- Hassas bilgilerin korunması (gelecek aşamalarda)

## 4. Geliştirme Yol Haritası

### 4.1 Sprint 1: Temel Altyapı
- Doküman yükleme ve işleme altyapısı
- Vektörleştirme sistemi
- Yerel FAISS veritabanı entegrasyonu
- Basit QA zinciri
- Komut satırı arayüzü

### 4.2 Sprint 2: Kullanıcı Deneyimi ve Gelişmiş Özellikler
- Web tabanlı kullanıcı arayüzü
- Çoklu doküman yönetimi
- Dokümanlara metadata ekleme
- Gelişmiş sorgulama seçenekleri

### 4.3 Sprint 3: Entegrasyon ve Optimizasyon
- Kategorilendirme özellikleri
- Bulut tabanlı vektör veritabanı seçeneği
- Performans optimizasyonları
- Dışa/içe aktarma özellikleri

## 5. Değerlendirme Kriterleri

### 5.1 Performans Metrikleri
- Sorgu yanıt süresi (< 2 saniye hedefi)
- Yanıt doğruluğu ve ilgisi
- Doküman işleme hızı

### 5.2 Kullanıcı Deneyimi Metrikleri
- Kullanım kolaylığı
- Kurulum süreci basitliği
- Dokümantasyon kalitesi

## 6. Kısıtlamalar ve Varsayımlar

### 6.1 Kısıtlamalar
- API kullanımı için maliyetler
- Büyük doküman koleksiyonları için performans limitleri
- Dil modeli ve embedding kapasitesi kısıtlamaları

### 6.2 Varsayımlar
- Kullanıcıların geçerli API anahtarlarına erişimi olacak
- Dokümanların çoğunlukla metin içerik taşıyacağı
- Temel Python bilgisine sahip kullanıcılar hedefleniyor (ilk aşamada)

## 7. Ek Bilgiler

### 7.1 Referanslar
- LangChain dokümantasyonu
- OpenAI API dokümantasyonu
- FAISS dokümantasyonu
- RAG (Retrieval Augmented Generation) akademik makaleleri

### 7.2 Sözlük
- **RAG**: Retrieval Augmented Generation
- **Embedding**: Dokümanları vektörlere dönüştürme işlemi
- **Chunking**: Dokümanları anlamlı parçalara bölme
- **LLM**: Large Language Model (Büyük Dil Modeli) 