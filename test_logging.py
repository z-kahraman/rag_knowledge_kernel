"""
Log Sistemi Testi
-----------------
Bu modül, merkezi loglama sisteminin düzgün çalıştığını doğrular.
"""

import os
import sys
import time

# Loglama yapılandırmasını içe aktar
from utils.logging_config import setup_logging, get_logger

def main():
    """
    Loglama sistemini test eder.
    """
    # Loglama sistemini başlat
    setup_logging()
    
    # Ana modül logger'ı
    logger = get_logger(__name__)
    
    # Test logları oluştur
    logger.debug("Bu bir DEBUG mesajıdır - sadece dosyaya yazılmalı")
    logger.info("Bu bir INFO mesajıdır - hem konsola hem dosyaya yazılmalı")
    logger.warning("Bu bir WARNING mesajıdır - hem konsola hem dosyaya yazılmalı")
    logger.error("Bu bir ERROR mesajıdır - hem konsola hem dosyaya yazılmalı")
    
    # Loglama başarılı oldu mu kontrol et
    log_file = "./logs/bilgicekirdegi.log"
    if os.path.exists(log_file):
        print(f"\nLog dosyası oluşturuldu: {log_file}")
        print("\nLog dosyasının içeriği:")
        
        # Log dosyasının son 5 satırını göster
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(line.strip())
    else:
        print(f"\nHATA: Log dosyası oluşturulamadı: {log_file}")

if __name__ == "__main__":
    main() 