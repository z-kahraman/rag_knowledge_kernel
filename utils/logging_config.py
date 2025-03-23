"""
BilgiÇekirdeği Loglama Yapılandırması
------------------------------------
Bu modül, tüm proje için merkezi bir loglama yapılandırması sağlar.
Loglar hem konsola hem de dosyaya yazılır.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys

# Log dosyalarının kaydedileceği dizin
LOG_DIRECTORY = "./logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Varsayılan log dosyası
DEFAULT_LOG_FILE = os.path.join(LOG_DIRECTORY, "bilgicekirdegi.log")
HTTP_LOG_FILE = os.path.join(LOG_DIRECTORY, "http_requests.log")

def setup_logging(
    log_file: str = None, 
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    enable_http_logging: bool = True
) -> None:
    """
    Proje genelinde kullanılacak loglama yapılandırmasını ayarlar.
    
    Args:
        log_file: Log dosyasının yolu (None ise DEFAULT_LOG_FILE kullanılır)
        console_level: Konsol loglarının seviyesi
        file_level: Dosya loglarının seviyesi
        max_file_size: Maksimum log dosyası boyutu (byte)
        backup_count: Tutulacak yedek log dosyası sayısı
        enable_http_logging: HTTP isteklerinin detaylı loglamasını etkinleştirir
    """
    # Log dosyasını belirle
    log_file = log_file or DEFAULT_LOG_FILE
    
    # Root logger'ı yapılandır
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # En düşük seviye (filtreler daha sonra uygulanır)
    
    # Önceki tüm handler'ları temizle
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Log formatını belirle
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Konsol log handler'ı
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # Dosya log handler'ı (RotatingFileHandler kullanılıyor)
    try:
        # Önce dosya dizininin varlığını kontrol et
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # RotatingFileHandler'ı oluştur
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            delay=True  # Dosyayı hemen açmak yerine, gerektiğinde açacak
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Log dosyası yapılandırılırken hata: {str(e)}")
        # Konsola yine de bildirelim ama uygulama çalışsın
    
    # HTTP isteklerinin detaylı loglanması için yapılandırma
    if enable_http_logging:
        setup_http_logging(max_file_size, backup_count)
    
    # Bilgi mesajı
    logging.info(f"Loglama yapılandırması tamamlandı. Log dosyası: {log_file}")

def setup_http_logging(max_file_size: int = 10 * 1024 * 1024, backup_count: int = 5) -> None:
    """
    HTTP isteklerinin detaylı loglanması için yapılandırma.
    
    Args:
        max_file_size: Maksimum log dosyası boyutu (byte)
        backup_count: Tutulacak yedek log dosyası sayısı
    """
    # HTTP related loggers
    http_loggers = [
        "httpcore", 
        "httpx", 
        "requests", 
        "urllib3", 
        "httpcore.http11", 
        "httpcore.connection"
    ]
    
    # Detaylı HTTP log formatı
    http_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # HTTP log dizininin varlığını kontrol et
        http_log_dir = os.path.dirname(HTTP_LOG_FILE)
        if not os.path.exists(http_log_dir):
            os.makedirs(http_log_dir, exist_ok=True)
            
        # HTTP log handler
        http_handler = RotatingFileHandler(
            HTTP_LOG_FILE,
            maxBytes=max_file_size,
            backupCount=backup_count,
            delay=True  # Dosyayı hemen açmak yerine, gerektiğinde açacak
        )
        http_handler.setLevel(logging.DEBUG)
        http_handler.setFormatter(http_formatter)
        
        # Set up each HTTP logger
        for logger_name in http_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(http_handler)
            logger.propagate = False  # Prevent logs from being sent to root logger
        
        logging.info(f"HTTP isteklerinin detaylı loglaması etkinleştirildi. Log dosyası: {HTTP_LOG_FILE}")
    except Exception as e:
        print(f"HTTP log dosyası yapılandırılırken hata: {str(e)}")
        # Loglama olmasa da uygulama çalışmaya devam etsin

# Özel modül logger'ı alma
def get_logger(name: str) -> logging.Logger:
    """
    Belirtilen isimde bir logger döndürür.
    
    Args:
        name: Logger adı
        
    Returns:
        Logger: Yapılandırılmış logger nesnesi
    """
    return logging.getLogger(name) 