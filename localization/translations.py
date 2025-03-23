"""
BilgiÇekirdeği Dil Desteği Modülü
--------------------------------
Bu modül, uygulama içindeki metinlerin farklı dillerde görüntülenmesini sağlar.
Şu anda desteklenen diller: Türkçe ve İngilizce.

Knowledge Kernel Localization Module
-----------------------------------
This module provides support for displaying texts in different languages throughout the application.
Currently supported languages: Turkish and English.
"""

# Türkçe metinler sözlüğü (varsayılan)
# Turkish texts dictionary (default)
TURKISH = {
    # Genel / General
    "app_title": "BilgiÇekirdeği - Kişisel Doküman Asistanı",
    "welcome_title": "Hoş Geldiniz!",
    "welcome_text": "BilgiÇekirdeği, dokümanlarınızı yapay zeka ile sorgulamanızı sağlayan açık kaynaklı bir bilgi erişim sistemidir.",
    
    # Menü / Menu
    "menu_home": "Ana Sayfa",
    "menu_pdf_upload": "Doküman Yükleme",
    "menu_ask": "Soru Sorma",
    "menu_collections": "Koleksiyonlar",
    "menu_settings": "Ayarlar",
    "menu_about": "Hakkında",
    
    # Ana sayfa / Home
    "home_subtitle": "Dokümanlarınızı Yapay Zeka İle Sorgulayın",
    "home_description": "PDF belgelerinizi vektör veritabanına yükleyin ve doğal dil sorguları ile bilgiye ulaşın!",
    "home_get_started": "Başlamak için bir PDF belgesi yükleyin ya da var olan bir koleksiyona soru sorun.",
    "app_usage_title": "BilgiÇekirdeği Nasıl Kullanılır?",
    "app_usage_step1": "sekmesinden PDF dokümanlarınızı yükleyin",
    "app_usage_step2": "sekmesinden dokümanlarınıza soru sorun",
    "app_usage_step3": "sekmesinden koleksiyonlarınızı yönetin",
    "app_version": "BilgiÇekirdeği",
    
    # PDF Yükleme / PDF Upload
    "upload_title": "PDF Doküman Yükleme",
    "upload_description": "PDF belgelerinizi koleksiyona ekleyin ve vektör veritabanına kaydedin.",
    "upload_button": "Dosya Seç",
    "upload_processing": "İşleniyor...",
    "upload_success": "Dosya başarıyla yüklendi",
    "upload_error": "Dosya yüklenirken bir hata oluştu",
    "upload_drag_drop": "Dosyayı sürükleyip bırakın veya tıklayarak seçin",
    "upload_supported": "Desteklenen format: PDF",
    "upload_collection_label": "Koleksiyon Adı",
    "upload_collection_help": "Dokümanın kaydedileceği koleksiyon adı",
    "upload_embedding_label": "Embedding Modeli",
    "upload_embedding_help": "Vektörleştirme için kullanılacak embedding modeli",
    "upload_chunk_size_label": "Bölüm Boyutu",
    "upload_chunk_size_help": "Her bir metin parçasının karakter sayısı",
    "upload_chunk_overlap_label": "Örtüşme Boyutu",
    "upload_chunk_overlap_help": "Ardışık parçalar arasındaki örtüşen karakter sayısı",
    "upload_process_button": "Dokümanı İşle",
    "upload_cancel_button": "İptal Et",
    "pdf_upload_instructions_title": "BilgiÇekirdeği Nasıl Çalışır?",
    "upload_button_click": "butonuna tıklayın",
    "upload_add_to_vectordb": "Yüklenen dosyayı vektör veritabanına ekleyin",
    
    # PDF işleme adımları
    "process_step_1": "Yüklenen PDF dosyaları küçük parçalara bölünür",
    "process_step_2": "Her parça vektör temsillere dönüştürülür",
    "process_step_3": "Vektörler veritabanında saklanır",
    "process_step_4": "Sorularınız benzer şekilde vektörlere dönüştürülür",
    "process_step_5": "En ilgili doküman parçaları bulunur",
    "process_step_6": "Yapay zeka doküman parçalarını kullanarak yanıt üretir",
    
    # Soru Sorma / Ask Question
    "ask_title": "Dokümanlarınıza Soru Sorun",
    "ask_description": "Dokümanlarınızla ilgili sorular sorun ve yapay zeka destekli yanıtlar alın.",
    "ask_collection_label": "Koleksiyon",
    "ask_collection_help": "Sorgulamak istediğiniz koleksiyon",
    "ask_placeholder": "Dokümanlarınıza sormak istediğiniz soruyu buraya yazın...",
    "ask_button": "Soru Sor",
    "ask_clear_button": "Temizle",
    "ask_example_button": "Örnek Soru",
    "ask_processing": "Yanıt oluşturuluyor...",
    "ask_error": "Soru yanıtlanırken bir hata oluştu",
    "ask_sources": "Kaynaklar",
    "ask_no_sources": "Kaynak doküman bulunamadı",
    
    # Önbellek Ayarları / Cache Settings
    "cache_settings": "Önbellek Ayarları",
    "use_cache": "Önbelleklemeyi Kullan",
    "use_cache_help": "Aynı soruların daha hızlı yanıtlanması için önbellek kullan",
    "clear_collection_cache": "Bu Koleksiyonun Önbelleğini Temizle",
    "clear_all_caches": "Tüm Önbellekleri Temizle",
    "clear_all_cache": "Tüm Önbellekleri Temizle",
    "cache_cleared": "Önbellek temizlendi!",
    "answers_cleared": "Yanıtlar temizlendi. Yeni bir sorgu yapabilirsiniz.",
    
    # Koleksiyonlar / Collections
    "collections_title": "Koleksiyonlar",
    "collections_description": "Vektör veritabanındaki koleksiyonları yönetin.",
    "collections_name": "Koleksiyon Adı",
    "collections_documents": "Doküman Sayısı",
    "collections_size": "Boyut",
    "collections_embedding": "Embedding",
    "collections_actions": "İşlemler",
    "collections_delete": "Sil",
    "collections_view": "Görüntüle",
    "collections_empty": "Henüz hiç koleksiyon oluşturulmadı.",
    "collection_found": "koleksiyon bulundu",
    "collections_load_error": "Koleksiyonlar yüklenirken hata",
    
    # Ayarlar / Settings
    "settings_title": "Uygulama Ayarları",
    "settings_description": "BilgiÇekirdeği uygulaması için ayarları yapılandırın.",
    "settings_language": "Dil Seçimi",
    "settings_language_turkish": "Türkçe",
    "settings_language_english": "İngilizce",
    "settings_language_change": "Dil değiştirildi. Değişikliklerin tam olarak uygulanması için sayfayı yenileyin.",
    "settings_llm_section": "LLM Ayarları",
    "settings_llm_provider": "LLM Sağlayıcısı",
    "settings_llm_model": "LLM Modeli",
    "settings_embedding_section": "Embedding Ayarları",
    "settings_embedding_provider": "Embedding Sağlayıcısı",
    "settings_embedding_model": "Embedding Modeli",
    "settings_openai_api_key": "OpenAI API Anahtarı",
    "settings_save": "Ayarları Kaydet",
    "settings_reset": "Varsayılana Sıfırla",
    
    # Bilgi Kutuları / Information Boxes
    "important_note": "Önemli Not:",
    "embedding_model_warning": "Vektör veritabanı oluştururken kullandığınız embedding modeli ile sorgu yaparken aynı modeli kullanmanız gerekir. Aksi halde boyut uyuşmazlığı hatası alırsınız.",
    
    # Hakkında / About
    "about_title": "BilgiÇekirdeği Hakkında",
    "about_version": "Versiyon",
    "about_description": "BilgiÇekirdeği, dokümanlarınızı yapay zeka ile sorgulamanızı sağlayan açık kaynaklı bir bilgi erişim sistemidir.",
    "about_features": "Özellikler",
    "about_feature_1": "PDF belgelerini vektör veritabanına indeksleme",
    "about_feature_2": "Dokümanları doğal dil ile sorgulama",
    "about_feature_3": "OpenAI veya Ollama LLM modelleri desteği",
    "about_feature_4": "Kullanıcı dostu web arayüzü",
    "about_feature_5": "Yanıtlarınız için kaynak belgeleri görüntüleme",
    "about_github": "GitHub'da Görüntüle",
    
    # Örnek Sorular / Example Questions
    "example_question_select_prompt": "Aşağıdaki örnek sorulardan birini seçebilir veya kendiniz bir soru yazabilirsiniz:",
    "example_question_select_placeholder": "Bir örnek soru seçin...",
    "selected_question": "Seçilen soru",
    "example_question_1": "Bu dokümanda neler anlatılıyor?",
    "example_question_2": "Bu dokümanın bir özetini çıkarır mısın?",
    "example_question_3": "Bu dokümandaki en önemli kısımlar hangileridir?",
    "example_question_4": "Bir sunum yapmak istesem bu dokümanda hangi konulara odaklanmalıyım?",
    "example_question_5": "Bu dokümandaki bilgileri madde madde listeleyebilir misin?",
    "example_question_6": "Bu dokümanın ana temalarını açıklar mısın?",
    "example_question_7": "Bu dokümanda bahsedilen en önemli konular nelerdir?",
    "example_question_8": "Bu dokümanda geçen temel tanımlar nelerdir?",
    "example_question_9": "Bu dokümanda önerilen çözümler nelerdir?",
    "example_question_10": "Bu dokümanda anlatılan metodolojiler nelerdir?",
    "example_question_11": "Bu dokümanı bir yönetici özetine dönüştürebilir misin?",
    "example_question_12": "Bu dokümanı 3 dakikada anlatmak istesem hangi kısımlara odaklanmalıyım?",
    "example_question_13": "Bu dokümanda anlatılan konuları bir uzman bakış açısıyla değerlendirebilir misin?",
    
    # Genel Sorular / General Questions
    "example_doc_content": "Bu dokümanda neler anlatılıyor?",
    "example_doc_summary": "Bu dokümanın bir özetini çıkarır mısın?",
    "example_doc_important": "Bu dokümandaki en önemli kısımlar hangileridir?",
    "example_doc_presentation": "Bir sunum yapmak istesem bu dokümanda hangi konulara odaklanmalıyım?",
    "example_doc_list": "Bu dokümandaki bilgileri madde madde listeleyebilir misin?",
    "example_doc_themes": "Bu dokümanın ana temalarını açıklar mısın?",
    
    # Alan Bazlı Spesifik Sorular / Domain Specific Questions
    "example_domain_important": "Bu dokümanda bahsedilen en önemli konular nelerdir?",
    "example_domain_definitions": "Bu dokümanda geçen temel tanımlar nelerdir?",
    "example_domain_solutions": "Bu dokümanda önerilen çözümler nelerdir?",
    "example_domain_methods": "Bu dokümanda anlatılan metodolojiler nelerdir?",
    
    # Kullanım Senaryoları / Usage Scenarios
    "example_usage_executive": "Bu dokümanı bir yönetici özetine dönüştürebilir misin?",
    "example_usage_quick": "Bu dokümanı 3 dakikada anlatmak istesem hangi kısımlara odaklanmalıyım?",
    "example_usage_expert": "Bu dokümanda anlatılan konuları bir uzman bakış açısıyla değerlendirebilir misin?",
    
    # Çeşitli / Miscellaneous
    "load_models_error": "Modeller yüklenirken bir hata oluştu",
    "invalid_api_key": "Geçersiz API anahtarı",
    "no_documents": "Henüz yüklenmiş bir doküman yok",
    "no_questions": "Henüz sorulmuş bir soru yok",
    "empty_input": "Lütfen bir soru girin",
    "query_time": "Sorgu süresi",
    "cached_response": "Önbellekten yanıt verildi",
    "seconds": "saniye",
    "process_success": "İşlem başarılı",
    "process_error": "İşlem sırasında bir hata oluştu",
    "file_upload_prompt": "Bu dosyayı vektör veritabanına eklemek ister misiniz?",
    "file_too_large": "Dosya boyutu çok büyük",
    
    # Soru sekmesi ek çeviriler / Ask tab additional translations
    "ask_enter_question": "Sorunuzu Girin",
    "ask_question": "Soru",
    "clear_button": "Temizle",
    "query_settings": "Sorgu Ayarları",
    "collection_to_query": "Sorgulanacak Koleksiyon",
    "collection_created_with": "Bu koleksiyon şu model ile oluşturulmuş:",
    "using_same_model": "Sorgu yaparken uyumsuzluk hatalarını önlemek için otomatik olarak aynı model kullanılacak.",
    "no_collections": "Henüz hiç koleksiyon bulunmuyor. Lütfen önce bir doküman yükleyin.",
    "query_tips_title": "Sorgu İpuçları",
    "query_tip_1": "Sorunuzu açık ve net bir şekilde ifade edin",
    "query_tip_2": "Sorularınızı tam cümleler halinde sorun",
    "query_tip_3": "Aşırı uzun sorular yerine birden fazla kısa soru sorun",
    "query_tip_4": "Yanıtın belirli bir formatta olmasını istiyorsanız belirtin",
    "answer_title": "Yanıt",
    "answer_generated_in": "Yanıt şu sürede oluşturuldu:",
    "from_cache": "önbellekten",
    "source_documents": "Kaynak Belgeler",
    "source": "Kaynak",
    "page": "Sayfa",
    "unknown": "Bilinmeyen",
    "content": "İçerik",
    "no_source_docs": "Bu sorgu için kaynak belge bulunamadı.",
    "generating_answer": "Yanıt oluşturuluyor... Bu işlem sistemin yüküne bağlı olarak 10-30 saniye sürebilir.",
    "error": "Hata",
    "query_error": "Sorgu çalıştırılırken hata",
    "collection_cache_cleared": "koleksiyonunun önbelleği temizlendi!",
    "all_caches_cleared": "Tüm önbellekler temizlendi!",
    
    # Koleksiyonlar sekmesi / Collections tab
    "collections_stats": "İstatistikler",
    "available_collections": "Mevcut Koleksiyonlar",
    "document_count": "Doküman Sayısı",
    "embedding": "Embedding",
    "no_metadata": "Metadata bilgisi bulunamadı",
    "query_button": "Sorgula",
    "delete_button": "Sil",
    "delete_confirmation": "Koleksiyon Silme Onayı",
    "delete_warning": "Bu koleksiyonu silmek istediğinizden emin misiniz",
    "confirm_delete": "Sil",
    "cancel_delete": "İptal",
    "collection_deleted": "koleksiyonu başarıyla silindi!",
    "delete_error": "Silme hatası",
    "no_collections_warning": "Henüz hiç koleksiyon bulunmuyor.",
    "add_document_title": "İlk Dokümanınızı Ekleyin",
    "add_document_instruction": "Bir koleksiyon oluşturmak için önce 'Doküman Yükleme' sekmesinden bir PDF belgesi yükleyin.",
    
    # İstatistikler sekmesi / Stats tab
    "stats_title": "Koleksiyon İstatistikleri",
    "stats_description": "Vektör veritabanındaki koleksiyonlara ait istatistikleri görüntüleyin.",
    "general_stats": "Genel İstatistikler",
    "collection_count": "Koleksiyon Sayısı",
    "vector_count": "Vektör Sayısı",
    "embedding_model_count": "Embedding Model Sayısı",
    "embedding_models_used": "Kullanılan Embedding Modelleri",
    "collection_details": "Koleksiyon Detayları",
    "select_collection_for_stats": "İstatistikleri Görüntülenecek Koleksiyon",
    "details": "Detayları",
    "created_date": "Oluşturulma Tarihi",
    "embedding_provider": "Embedding Sağlayıcısı",
    "embedding_model": "Embedding Modeli",
    "chunk_size": "Bölüm Boyutu",
    "chunk_overlap": "Örtüşme Boyutu",
    "documents_in_collection": "Koleksiyondaki Dokümanlar",
    "error_loading_metadata": "Metadata yüklenirken hata",
    "no_collections_for_stats": "İstatistik görüntülemek için henüz bir koleksiyon bulunmuyor"
}

# İngilizce metinler sözlüğü
# English texts dictionary
ENGLISH = {
    # Genel / General
    "app_title": "Knowledge Kernel - Personal Document Assistant",
    "welcome_title": "Welcome!",
    "welcome_text": "Knowledge Kernel is an open-source information retrieval system that allows you to query your documents using artificial intelligence.",
    
    # Menü / Menu
    "menu_home": "Home",
    "menu_pdf_upload": "Document Upload",
    "menu_ask": "Ask Question",
    "menu_collections": "Collections",
    "menu_settings": "Settings",
    "menu_about": "About",
    
    # Ana sayfa / Home
    "home_subtitle": "Query Your Documents with Artificial Intelligence",
    "home_description": "Upload your PDF documents to the vector database and access information with natural language queries!",
    "home_get_started": "To get started, upload a PDF document or ask questions to an existing collection.",
    "app_usage_title": "How to Use Knowledge Kernel?",
    "app_usage_step1": "tab to upload your PDF documents",
    "app_usage_step2": "tab to ask questions about your documents",
    "app_usage_step3": "tab to manage your collections",
    "app_version": "Knowledge Kernel",
    
    # PDF Yükleme / PDF Upload
    "upload_title": "PDF Document Upload",
    "upload_description": "Add your PDF documents to the collection and save them to the vector database.",
    "upload_button": "Choose File",
    "upload_processing": "Processing...",
    "upload_success": "File uploaded successfully",
    "upload_error": "An error occurred while uploading the file",
    "upload_drag_drop": "Drag and drop a file or click to select",
    "upload_supported": "Supported format: PDF",
    "upload_collection_label": "Collection Name",
    "upload_collection_help": "Collection name where the document will be saved",
    "upload_embedding_label": "Embedding Model",
    "upload_embedding_help": "Embedding model to be used for vectorization",
    "upload_chunk_size_label": "Chunk Size",
    "upload_chunk_size_help": "Number of characters in each text segment",
    "upload_chunk_overlap_label": "Overlap Size",
    "upload_chunk_overlap_help": "Number of overlapping characters between consecutive segments",
    "upload_process_button": "Process Document",
    "upload_cancel_button": "Cancel",
    "pdf_upload_instructions_title": "How to Upload a PDF Document?",
    "upload_button_click": "button",
    "upload_add_to_vectordb": "Add the uploaded file to the vector database",
    
    # PDF işleme adımları
    "process_step_1": "Uploaded PDF files are divided into small chunks",
    "process_step_2": "Each chunk is converted to vector representations",
    "process_step_3": "Vectors are stored in the database",
    "process_step_4": "Your questions are similarly converted to vectors",
    "process_step_5": "The most relevant document chunks are found",
    "process_step_6": "AI uses document chunks to generate an answer",
    
    # Soru Sorma / Ask Question
    "ask_title": "Ask Questions About Your Documents",
    "ask_description": "Ask questions about your documents and get AI-powered answers.",
    "ask_collection_label": "Collection",
    "ask_collection_help": "Collection you want to query",
    "ask_placeholder": "Type your question about your documents here...",
    "ask_button": "Ask Question",
    "ask_clear_button": "Clear",
    "ask_example_button": "Example Question",
    "ask_processing": "Generating answer...",
    "ask_error": "An error occurred while answering the question",
    "ask_sources": "Sources",
    "ask_no_sources": "No source documents found",
    
    # Önbellek Ayarları / Cache Settings
    "cache_settings": "Cache Settings",
    "use_cache": "Use Caching",
    "use_cache_help": "Use cache for faster responses to repeated questions",
    "clear_collection_cache": "Clear This Collection's Cache",
    "clear_all_caches": "Clear All Caches",
    "clear_all_cache": "Clear All Caches",
    "cache_cleared": "Cache cleared!",
    "answers_cleared": "Answers cleared. You can make a new query.",
    
    # Koleksiyonlar / Collections
    "collections_title": "Collections",
    "collections_description": "Manage collections in the vector database.",
    "collections_name": "Collection Name",
    "collections_documents": "Number of Documents",
    "collections_size": "Size",
    "collections_embedding": "Embedding",
    "collections_actions": "Actions",
    "collections_delete": "Delete",
    "collections_view": "View",
    "collections_empty": "No collections have been created yet.",
    "collection_found": "collections found",
    "collections_load_error": "Error loading collections",
    
    # Ayarlar / Settings
    "settings_title": "Application Settings",
    "settings_description": "Configure settings for the Knowledge Kernel application.",
    "settings_language": "Language Selection",
    "settings_language_turkish": "Turkish",
    "settings_language_english": "English",
    "settings_language_change": "Language changed. Refresh the page for the changes to fully take effect.",
    "settings_llm_section": "LLM Settings",
    "settings_llm_provider": "LLM Provider",
    "settings_llm_model": "LLM Model",
    "settings_embedding_section": "Embedding Settings",
    "settings_embedding_provider": "Embedding Provider",
    "settings_embedding_model": "Embedding Model",
    "settings_openai_api_key": "OpenAI API Key",
    "settings_save": "Save Settings",
    "settings_reset": "Reset to Default",
    
    # Bilgi Kutuları / Information Boxes
    "important_note": "Important Note:",
    "embedding_model_warning": "You must use the same embedding model when querying as you used when creating the vector database. Otherwise, you will get a dimension mismatch error.",
    
    # Hakkında / About
    "about_title": "About Knowledge Kernel",
    "about_version": "Version",
    "about_description": "Knowledge Kernel is an open-source information retrieval system that allows you to query your documents using artificial intelligence.",
    "about_features": "Features",
    "about_feature_1": "Index PDF documents to a vector database",
    "about_feature_2": "Query documents using natural language",
    "about_feature_3": "Support for OpenAI or Ollama LLM models",
    "about_feature_4": "User-friendly web interface",
    "about_feature_5": "View source documents for your answers",
    "about_github": "View on GitHub",
    
    # Örnek Sorular / Example Questions
    "example_question_select_prompt": "Select one of the following example questions or write your own question:",
    "example_question_select_placeholder": "Select an example question...",
    "selected_question": "Selected question",
    "example_question_1": "What is discussed in this document?",
    "example_question_2": "Can you summarize this document?",
    "example_question_3": "What are the most important parts of this document?",
    "example_question_4": "If I were to make a presentation, which topics in this document should I focus on?",
    "example_question_5": "Can you list the information in this document in bullet points?",
    "example_question_6": "Can you explain the main themes of this document?",
    "example_question_7": "What are the most important topics mentioned in this document?",
    "example_question_8": "What are the key definitions in this document?",
    "example_question_9": "What solutions are proposed in this document?",
    "example_question_10": "What methodologies are described in this document?",
    "example_question_11": "Can you convert this document into an executive summary?",
    "example_question_12": "If I had to present this document in 3 minutes, which parts should I focus on?",
    "example_question_13": "Can you evaluate the topics discussed in this document from an expert perspective?",
    
    # Genel Sorular / General Questions
    "example_doc_content": "What is discussed in this document?",
    "example_doc_summary": "Can you summarize this document?",
    "example_doc_important": "What are the most important parts of this document?",
    "example_doc_presentation": "If I were to make a presentation, which topics in this document should I focus on?",
    "example_doc_list": "Can you list the information in this document in bullet points?",
    "example_doc_themes": "Can you explain the main themes of this document?",
    
    # Alan Bazlı Spesifik Sorular / Domain Specific Questions
    "example_domain_important": "What are the most important topics mentioned in this document?",
    "example_domain_definitions": "What are the key definitions in this document?",
    "example_domain_solutions": "What solutions are proposed in this document?",
    "example_domain_methods": "What methodologies are described in this document?",
    
    # Kullanım Senaryoları / Usage Scenarios
    "example_usage_executive": "Can you convert this document into an executive summary?",
    "example_usage_quick": "If I had to present this document in 3 minutes, which parts should I focus on?",
    "example_usage_expert": "Can you evaluate the topics discussed in this document from an expert perspective?",
    
    # Çeşitli / Miscellaneous
    "load_models_error": "An error occurred while loading models",
    "invalid_api_key": "Invalid API key",
    "no_documents": "No documents have been uploaded yet",
    "no_questions": "No questions have been asked yet",
    "empty_input": "Please enter a question",
    "query_time": "Query time",
    "cached_response": "Answered from cache",
    "seconds": "seconds",
    "process_success": "Process successful",
    "process_error": "An error occurred during the process",
    "file_upload_prompt": "Would you like to add this file to the vector database?",
    "file_too_large": "File size is too large",
    
    # Soru sekmesi ek çeviriler / Ask tab additional translations
    "ask_enter_question": "Enter Your Question",
    "ask_question": "Question",
    "clear_button": "Clear",
    "query_settings": "Query Settings",
    "collection_to_query": "Collection to Query",
    "collection_created_with": "This collection was created with:",
    "using_same_model": "The same model will be used automatically to prevent incompatibility errors when querying.",
    "no_collections": "No collections yet. Please upload a document first.",
    "query_tips_title": "Query Tips",
    "query_tip_1": "Express your question clearly and concisely",
    "query_tip_2": "Ask questions in complete sentences",
    "query_tip_3": "Ask multiple short questions instead of one very long question",
    "query_tip_4": "Specify if you want the answer in a particular format",
    "answer_title": "Answer",
    "answer_generated_in": "Answer generated in",
    "from_cache": "from cache",
    "source_documents": "Source Documents",
    "source": "Source",
    "page": "Page",
    "unknown": "Unknown",
    "content": "Content",
    "no_source_docs": "No source documents found for this query.",
    "generating_answer": "Generating answer... This process may take 10-30 seconds depending on system load.",
    "error": "Error",
    "query_error": "Error running query",
    "collection_cache_cleared": "collection cache cleared!",
    "all_caches_cleared": "All caches cleared!",
    
    # Koleksiyonlar sekmesi / Collections tab
    "collections_stats": "Statistics",
    "available_collections": "Available Collections",
    "document_count": "Document Count",
    "embedding": "Embedding",
    "no_metadata": "No metadata information found",
    "query_button": "Query",
    "delete_button": "Delete",
    "delete_confirmation": "Collection Deletion Confirmation",
    "delete_warning": "Are you sure you want to delete this collection",
    "confirm_delete": "Delete",
    "cancel_delete": "Cancel",
    "collection_deleted": "collection successfully deleted!",
    "delete_error": "Delete error",
    "no_collections_warning": "No collections found yet.",
    "add_document_title": "Add Your First Document",
    "add_document_instruction": "To create a collection, first upload a PDF document from the 'Document Upload' tab.",
    
    # İstatistikler sekmesi / Stats tab
    "stats_title": "Collection Statistics",
    "stats_description": "View statistics for collections in the vector database.",
    "general_stats": "General Statistics",
    "collection_count": "Collection Count",
    "vector_count": "Vector Count",
    "embedding_model_count": "Embedding Model Count",
    "embedding_models_used": "Embedding Models Used",
    "collection_details": "Collection Details",
    "select_collection_for_stats": "Collection to Display Statistics",
    "details": "Details",
    "created_date": "Creation Date",
    "embedding_provider": "Embedding Provider",
    "embedding_model": "Embedding Model",
    "chunk_size": "Chunk Size",
    "chunk_overlap": "Chunk Overlap",
    "documents_in_collection": "Documents in Collection",
    "error_loading_metadata": "Error loading metadata",
    "no_collections_for_stats": "No collections available for statistics yet"
}

# Varsayılan dil
DEFAULT_LANGUAGE = "tr"

def get_text(key, language=DEFAULT_LANGUAGE):
    """
    Belirli bir dil için metin çevirisini döndürür.
    
    Args:
        key: Metin anahtarı
        language: Dil kodu ('tr' veya 'en')
        
    Returns:
        str: Çevrilmiş metin. Eğer çeviri bulunamazsa, anahtar değerinin kendisi döndürülür.
    
    Returns the text translation for a specific language.
    
    Args:
        key: Text key
        language: Language code ('tr' or 'en')
        
    Returns:
        str: Translated text. If the translation is not found, the key value itself is returned.
    """
    if language == "en":
        return ENGLISH.get(key, key)
    return TURKISH.get(key, key) 