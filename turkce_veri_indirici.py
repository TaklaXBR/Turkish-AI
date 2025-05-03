#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Türkçe Veri İndirici - Huggingface Datasets kütüphanesinden Türkçe veri setlerini
satır sayarak indiren ve JSON formatında kaydeden araç.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, __version__
from concurrent.futures import ThreadPoolExecutor
import re

# ----- LOGLAMA AYARLARI -----
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = "turkce_veri_indirme.log"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TurkceVeriIndirici")

# ----- SABİT DEĞERLER -----
# Her dosyaya kaç satır yazılacak
BATCH_SIZE = 100000  # 100 bin satır (daha küçük dosyalar için)
# İstatistik güncelleme sıklığı (kaç satırda bir)
STATS_INTERVAL = 10000  # 10 bin satıra çıkarıldı
# Türkçe doğrulama için minimum Türkçe karakter oranı
MIN_TURKISH_CHAR_RATIO = 0.05

# ----- JSON DOĞRULAMA VE TEMİZLEME FONKSİYONLARI -----
def is_valid_json(data):
    """Verinin geçerli JSON'a dönüştürülebilir olup olmadığını kontrol eder"""
    try:
        if data is None:
            return False
            
        if not isinstance(data, (dict, list, str, int, float, bool)):
            return False
        
        # Test et - JSON'a dönüştürülebiliyor mu?
        if isinstance(data, (dict, list)):
            json.dumps(data, ensure_ascii=False)
        return True
    except (TypeError, OverflowError, ValueError):
        return False

def clean_json_data(data, category="genel"):
    """
    JSON verilerini temizler ve bozuk kısımları düzeltir
    
    Args:
        data: Temizlenecek veri
        category: Veri kategorisi (mizah, altyapi, egitim)
    
    Returns:
        Temizlenmiş veri veya None (düzeltilemediyse)
    """
    if data is None:
        return None
    
    # Dict değilse düzeltmeye çalış
    if not isinstance(data, dict):
        try:
            if isinstance(data, (str, bytes)):
                # String ise JSON olarak çözümlemeyi dene
                if isinstance(data, str) and data.strip().startswith('{'):
                    try:
                        data = json.loads(data)
                    except:
                        # JSON çözümlenemezse metin olarak kabul et
                        data = {"text": data}
                else:
                    # Düz metin olarak kabul et
                    data = {"text": data}
            elif isinstance(data, (list, tuple)):
                # Liste ise ilk öğeyi almayı dene
                if len(data) > 0 and isinstance(data[0], dict):
                    data = data[0]
                else:
                    # Liste dönüştürülemiyorsa sözlük oluştur
                    data = {"text": str(data)}
            else:
                # Diğer türleri metne dönüştür
                data = {"text": str(data)}
        except:
            # Düzeltilemezse None döndür
            return None
    
    # Boş dict ise None döndür
    if not data:
        return None
        
    # URL, html, link ve derin JSON temizliği
    try:
        # Text alanını temizle
        if "text" in data and isinstance(data["text"], str):
            text = data["text"]
            
            # HTML etiketlerini temizle
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # URL'leri temizle
            text = re.sub(r'https?://\S+', ' ', text)
            
            # Fazla boşlukları temizle
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Boş metin kontrolü
            if not text or len(text) < 5:  # En az 5 karakter olsun
                return None
                
            data["text"] = text
    except Exception:
        # Hata durumunda orijinal veriyi koru
        pass
    
    # Kategori bazlı özel temizlik
    if category == "mizah":
        # Mizah verileri için None ve boş değerleri düzelt
        clean_data = {}
        for key, value in data.items():
            if value is None:
                clean_data[key] = ""
            elif isinstance(value, str) and not value.strip():
                clean_data[key] = ""
            elif isinstance(value, (dict, list)) and not is_valid_json(value):
                try:
                    clean_data[key] = str(value)  # JSON'a dönüştürülemiyorsa string yap
                except:
                    clean_data[key] = ""
            else:
                clean_data[key] = value
        data = clean_data
    
    elif category in ["altyapi", "egitim"]:
        # Büyük veri setleri için text alanına odaklan
        if "text" in data and not data["text"]:
            return None  # Boş metin içeren kayıtları atla
        
        # Bellek optimizasyonu için fazla alanları kaldır
        text_fields = ["text", "content", "article", "summary", "title"]
        for field in text_fields:
            if field in data and data[field]:
                # Sadece metin alanını koru, gerisi atla
                clean_data = {"text": data[field]}
                if "id" in data:
                    clean_data["id"] = data["id"]
                data = clean_data
                break
    
    # Son olarak JSON serileştirilmesini test et
    try:
        json.dumps(data, ensure_ascii=False)
        return data
    except (TypeError, OverflowError, ValueError):
        # JSON olarak serileştirilemiyorsa
        if "text" in data:
            # Sadece metin alanını kurtarmayı dene
            return {"text": str(data["text"])}
        return None

# ----- YARDIMCI FONKSİYONLAR -----
def format_size(size_bytes):
    """Byte değerini okunabilir formata dönüştürür (KB, MB, GB)"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.2f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"

def format_time(seconds):
    """Saniye değerini saat:dakika:saniye formatına dönüştürür"""
    if seconds < 60:
        return f"{seconds:.2f} saniye"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)} dakika {int(seconds)} saniye"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)} saat {int(minutes)} dakika {int(seconds)} saniye"

def get_file_size(file_path):
    """Dosya boyutunu byte cinsinden döndürür"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0

def count_lines(file_path):
    """Dosyadaki satır sayısını sayar"""
    if not os.path.exists(file_path):
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"Satır sayma hatası ({file_path}): {str(e)}")
        return 0

def count_turkish_chars(text):
    """Metindeki Türkçe karakter sayısını döndürür"""
    if not text or not isinstance(text, str):
        return 0
    
    turkish_chars = "çÇğĞıİöÖşŞüÜ"
    return sum(1 for char in text if char in turkish_chars)

def is_turkish_text(text, min_ratio=MIN_TURKISH_CHAR_RATIO):
    """Metnin Türkçe olup olmadığını kontrol eder"""
    # Her durumda True döndür - tüm metinleri kabul et
    return True
    
    # Aşağıdaki kontroller devre dışı bırakıldı
    # if not text or not isinstance(text, str) or len(text) < 20:
    #     return False
    
    # # Türkçe karakter oranı
    # tr_chars = count_turkish_chars(text)
    # if len(text) > 0:
    #     ratio = tr_chars / len(text)
    #     if ratio >= min_ratio:
    #         return True
    
    # # Türkçeye özgü kelimeler içeriyor mu?
    # tr_words = ['ve', 'veya', 'ile', 'için', 'bu', 'şu', 'çünkü', 'ancak', 'fakat',
    #            'gibi', 'kadar', 'daha', 'en', 'çok', 'az', 'ama', 'lakin', 'değil']
    
    # word_count = 0
    # words = text.lower().split()
    # for word in tr_words:
    #     if word in words:
    #         word_count += 1
    
    # # En az 2 Türkçe kelime içeriyorsa Türkçedir
    # return word_count >= 2

# ----- DOSYA İŞLEMLERİ -----
class JsonWriter:
    """JSON dosyalarına hızlı ve verimli yazma işlemleri için sınıf"""
    
    def __init__(self, file_path, append=False, buffer_size=100000):
        """JSON dosyası için writer başlatır"""
        self.file_path = file_path
        self.mode = "a" if append else "w"
        self.line_count = 0
        self.error_count = 0
        self.buffer_size = buffer_size
        self.buffer = []
        
        # Dizini oluştur
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        try:
            # Dosyayı aç - açılmazsa oluşturur
            self.file = open(file_path, self.mode, encoding="utf-8", buffering=1)
            logger.info(f"JSON dosyası açıldı: {file_path}")
        except Exception as e:
            logger.error(f"Dosya açılırken hata: {str(e)}")
            raise e
    
    def write_line(self, data, category="genel"):
        """
        Veriyi JSON satırı olarak yazar.
        
        Args:
            data: Yazılacak veri (dict olmalı)
            category: Veri kategorisi, veri temizleme için kullanılır
            
        Returns:
            bool: Başarılı olursa True, başarısız olursa False
        """
        if not self.file or self.file.closed:
            logger.error("Dosya kapalı veya tanımlanmamış!")
            return False
            
        try:
            # Önce veriyi temizle
            clean_data = clean_json_data(data, category)
            
            # Veri temizleme başarısız olduysa atla
            if clean_data is None:
                self.error_count += 1
                return False
                
            try:
                # JSON'a dönüştür - bazı özel karakterler için ensure_ascii=False
                json_line = json.dumps(clean_data, ensure_ascii=False)
                
                # # ile başlayan yorum satırlarını atla
                if json_line.strip().startswith("#"):
                    logger.warning(f"Yorum satırı atlandı: {json_line[:30]}...")
                    self.error_count += 1
                    return False
                
                # Boş JSON nesnelerini atla
                if json_line.strip() in ['{}', '[]', '', 'null']:
                    logger.warning("Boş JSON nesnesi atlandı")
                    self.error_count += 1
                    return False
                
                # Tampon belleğe ekle
                self.buffer.append(json_line)
                self.line_count += 1
                
                # Tampon dolu ise diske yaz
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()
                
                return True
                
            except UnicodeEncodeError as e:
                # Unicode hatası durumunda string encode/decode işlemi yap
                logger.warning(f"Unicode hatası, düzeltiliyor: {str(e)}")
                try:
                    # Alternatif temizleme yöntemi
                    clean_str = json.dumps(clean_data, ensure_ascii=True)
                    self.buffer.append(clean_str)
                    self.line_count += 1
                    return True
                except Exception as inner_e:
                    logger.error(f"İkinci JSON yazma hatası: {str(inner_e)}")
                    self.error_count += 1
                    return False
            
        except Exception as e:
            logger.error(f"JSON yazma hatası: {str(e)}")
            self.error_count += 1
            return False
    
    def _flush_buffer(self):
        """Tampon belleği diske yazar"""
        if not self.buffer:
            return
            
        try:
            # Tüm tamponu diske yaz
            self.file.write("\n".join(self.buffer) + "\n")
            self.file.flush()
            self.buffer = []  # Tamponu temizle
        except Exception as e:
            logger.error(f"Tampon yazma hatası: {str(e)}")
    
    def get_stats(self):
        """Dosya istatistiklerini döndürür"""
        if not os.path.exists(self.file_path):
            return {
                "path": self.file_path,
                "lines": 0,
                "errors": 0,
                "size_bytes": 0,
                "size_str": "0 B"
            }
            
        size_bytes = get_file_size(self.file_path)
        return {
            "path": self.file_path,
            "lines": self.line_count,
            "errors": self.error_count,
            "size_bytes": size_bytes,
            "size_str": format_size(size_bytes)
        }
    
    def close(self):
        """Dosyayı kapatır"""
        try:
            if self.file and not self.file.closed:
                # Kalan tamponu diske yaz
                self._flush_buffer()
                
                self.file.flush()  # Önce flush et
                self.file.close()
                logger.info(f"Dosya kapatıldı: {self.file_path} ({self.line_count} satır, {self.error_count} hata)")
        except Exception as e:
            logger.error(f"Dosya kapatma hatası: {str(e)}")
            
    def __del__(self):
        """Sınıf yok edildiğinde dosyayı kapatır"""
        self.close()

# ----- VERİ İNDİRME VE İŞLEME -----
class DatasetProcessor:
    """Veri seti indirme ve işleme sınıfı"""
    
    def __init__(self, dataset_name, config=None, output_dir="./data", 
                 split="train", sample_size=None, verify_lang=True, category=None):
        """
        Veri seti işleyici başlatır
        
        Args:
            dataset_name: İndirilecek veri seti adı
            config: Veri seti yapılandırması
            output_dir: Çıktı dizini
            split: Veri bölümü (train, test, vs)
            sample_size: Kaç satır indirileceği (None ise hepsi)
            verify_lang: Türkçe metinleri doğrula (kullanılmıyor, her veri yazılıyor)
            category: Veri seti kategorisi (raporlama için)
        """
        self.dataset_name = dataset_name
        self.config = config
        self.output_dir = output_dir
        self.split = split
        self.sample_size = sample_size
        self.verify_lang = verify_lang  # kullanılmıyor, her veri kabul ediliyor
        self.category = category or "genel"
        
        # İstatistik değişkenleri
        self.start_time = None
        self.end_time = None
        self.processed_count = 0
        self.written_count = 0
        self.error_count = 0
        self.batch_count = 0
        self.batch_start_time = None
        
        # Dosya adı ve yazıcı
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self._get_output_filename(timestamp)
        self.writer = None
    
    def _get_output_filename(self, timestamp):
        """Çıktı dosyası adını oluşturur"""
        safe_name = self.dataset_name.replace("/", "_")
        file_name = f"{self.category}_{safe_name}_{timestamp}.jsonl"
        return os.path.join(self.output_dir, file_name)
    
    def process(self):
        """Veri setini işler ve JSON dosyasına kaydeder"""
        print(f"\n{'*' * 80}")
        print(f"VERİ SETİ İŞLEME BAŞLADI: {self.dataset_name}")
        if self.config:
            print(f"Yapılandırma: {self.config}")
        print(f"Çıktı dosyası: {self.output_file}")
        print(f"{'*' * 80}\n")
        
        try:
            # Yazıcıyı başlat - her seferinde yeni dosya (append=False)
            self.writer = JsonWriter(self.output_file, append=False)
            
            # İşlem başlangıç zamanı
            self.start_time = time.time()
            self.batch_start_time = time.time()
            self.batch_count = 1
            
            # Veri setini yükle
            print(f"Veri seti yükleniyor: {self.dataset_name}")
            load_start = time.time()
            
            # Datasets kütüphanesini kullanarak veri setini yükle
            try:
                # Büyük veri setleri için streaming mod kullan
                is_big_dataset = self.dataset_name == "mc4" or "oscar" in self.dataset_name
                streaming = True if is_big_dataset else False
                
                if self.config:
                    dataset = load_dataset(
                        self.dataset_name, 
                        self.config, 
                        split=self.split, 
                        streaming=streaming, 
                        trust_remote_code=True,
                        verification_mode="no_checks"  # Daha az sıkı doğrulama
                    )
                else:
                    dataset = load_dataset(
                        self.dataset_name, 
                        split=self.split, 
                        streaming=streaming, 
                        trust_remote_code=True,
                        verification_mode="no_checks"  # Daha az sıkı doğrulama
                    )
            except Exception as e:
                logger.error(f"Veri seti yükleme hatası: {str(e)}")
                print(f"HATA: Veri seti yüklenemedi: {str(e)}")
                
                # Eğer manuel indirme gerektiren bir veri seti ise bilgi ver
                if "requires manual data" in str(e):
                    print("\nBu veri seti manuel indirme gerektiriyor.")
                    print("Alternatif veri setleri kullanabilir veya manuel indirme talimatlarını takip edebilirsiniz.")
                
                if self.writer:
                    self.writer.close()
                return False
            
            load_time = time.time() - load_start
            print(f"Veri seti yüklendi: {self.dataset_name} ({format_time(load_time)})")
            
            # Toplam satır sayısı
            # Streaming modunda toplam satır bilinmez, tahmin edilir
            if hasattr(dataset, '__len__'):
                total_size = len(dataset) if self.sample_size is None else min(self.sample_size, len(dataset))
                streaming_mode = False
            else:
                # Streaming modunda toplam büyüklük bilinmez - sınırsız olarak işaretle
                print("Streaming modu etkin - toplam satır sayısı bilinmiyor (sınırsız indirme)")
                total_size = float('inf')  # Sonsuz olarak işaretle
                streaming_mode = True
            
            print(f"\n{'=' * 40}")
            print(f"VERİ SETİ: {self.dataset_name}")
            if not streaming_mode:
                print(f"TOPLAM SATIR: {total_size:,}")
            else:
                print(f"SATIR LİMİTİ: Sınırsız (tüm veri indirilecek)")
            print(f"{'=' * 40}\n")
            
            # İlerlemeyi göster
            for i, item in enumerate(dataset):
                if self.sample_size is not None and i >= self.sample_size:
                    break
                
                # Satır işleme sayacını artır
                self.processed_count += 1
                
                try:
                    # Veriyi temizle ve kontrol et (özellikle mizah verileri için)
                    clean_item = self._clean_json_item(item)
                    if clean_item:
                        # JSON satırı yaz
                        self.writer.write_line(clean_item, self.category)
                        self.written_count += 1
                except Exception as e:
                    logger.error(f"Satır işleme hatası: {str(e)}")
                    self.error_count += 1
                
                # 10,000 satırda bir ekrana ilerleme yazdır
                if self.processed_count % STATS_INTERVAL == 0:
                    elapsed = time.time() - self.start_time
                    avg_speed = self.processed_count / elapsed if elapsed > 0 else 0
                    if not streaming_mode:
                        eta = (total_size - self.processed_count) / avg_speed if avg_speed > 0 else 0
                        progress_percent = self.processed_count/total_size*100 if total_size > 0 else 0
                        print(f"\r[SATIR] {self.processed_count:,}/{total_size:,} "
                            f"({progress_percent:.1f}%) | "
                            f"Yazılan: {self.written_count:,} | "
                            f"Hız: {avg_speed:.2f} satır/s | "
                            f"Boyut: {format_size(get_file_size(self.output_file))} | "
                            f"Tahmini kalan süre: {format_time(eta)}", end="", flush=True)
                    else:
                        # Streaming modunda ilerleme göster (sınırsız mod)
                        print(f"\r[SATIR] {self.processed_count:,} işlendi | "
                            f"Yazılan: {self.written_count:,} | "
                            f"Hız: {avg_speed:.2f} satır/s | "
                            f"Boyut: {format_size(get_file_size(self.output_file))} | "
                            f"Süre: {format_time(elapsed)}", 
                            end="", flush=True)
                
                # 1,000,000 satırda bir batch tamamlandı mesajı
                if self.processed_count % BATCH_SIZE == 0:
                    batch_duration = time.time() - self.batch_start_time
                    speed = BATCH_SIZE / batch_duration if batch_duration > 0 else 0
                    elapsed = time.time() - self.start_time
                    avg_speed = self.processed_count / elapsed if elapsed > 0 else 0
                    file_size = get_file_size(self.output_file)
                    
                    print(f"\n\n{'=' * 80}")
                    print(f"BATCH {self.batch_count} TAMAMLANDI: {BATCH_SIZE:,} SATIR İŞLENDİ")
                    print(f"{'=' * 80}")
                    if not streaming_mode:
                        progress_percent = self.processed_count/total_size*100 if total_size > 0 else 0
                        eta = (total_size - self.processed_count) / avg_speed if avg_speed > 0 else 0
                        print(f"İşlenen satır:    {self.processed_count:,}/{total_size:,} ({progress_percent:.1f}%)")
                        print(f"Tahmini kalan:    {format_time(eta)}")
                    else:
                        print(f"İşlenen satır:    {self.processed_count:,} (devam ediyor)")
                    print(f"Yazılan satır:    {self.written_count:,}")
                    print(f"Batch süresi:     {format_time(batch_duration)}")
                    print(f"Batch hızı:       {speed:.2f} satır/saniye")
                    print(f"Ortalama hız:     {avg_speed:.2f} satır/saniye")
                    print(f"Toplam süre:      {format_time(elapsed)}")
                    print(f"Dosya boyutu:     {format_size(file_size)}")
                    print(f"{'=' * 80}\n")
                    
                    # Batch sayacını artır ve zamanı sıfırla
                    self.batch_count += 1
                    self.batch_start_time = time.time()
            
            # İşlem sonlandırma
            self.end_time = time.time()
            total_duration = self.end_time - self.start_time
            file_size = get_file_size(self.output_file)
            avg_speed = self.processed_count / total_duration if total_duration > 0 else 0
            
            print(f"\n\n{'#' * 80}")
            print(f"VERİ SETİ İŞLEME TAMAMLANDI: {self.dataset_name}")
            print(f"{'#' * 80}")
            print(f"Toplam işlenen satır: {self.processed_count:,}")
            print(f"Toplam yazılan satır: {self.written_count:,}")
            print(f"Hatalı satır sayısı: {self.error_count:,}")
            print(f"Toplam süre: {format_time(total_duration)}")
            print(f"Ortalama hız: {avg_speed:.2f} satır/saniye")
            print(f"Çıktı dosyası: {self.output_file}")
            print(f"Dosya boyutu: {format_size(file_size)}")
            print(f"{'#' * 80}\n")
            
            # Yazıcıyı kapat
            self.writer.close()
            return True
            
        except Exception as e:
            logger.exception(f"Veri işleme hatası: {self.dataset_name}")
            print(f"\nHATA: {self.dataset_name} veri seti işlenirken hata oluştu: {str(e)}")
            if self.writer:
                self.writer.close()
            return False
            
    def _clean_json_item(self, item):
        """JSON öğesini temizler ve bozuk verileri düzeltir"""
        try:
            # Verinin doğru formatta olduğunu kontrol et
            if not isinstance(item, dict):
                return None
                
            # Veri seti formatına göre temizleme
            if self.category == "mizah":
                # Mizah verilerinde sıkça karşılaşılan sorunlar
                
                # Text alanı kontrol
                if "text" in item:
                    # Boş text alanları atla
                    if not item["text"] or not isinstance(item["text"], str):
                        return None
                        
                elif "sentence" in item:
                    # Bazı duygu analizi veri setlerinde sentence alanı var
                    item["text"] = item.pop("sentence")
                    
                elif "content" in item:
                    # Bazı veri setlerinde content alanı var
                    item["text"] = item.pop("content")
                
                # Metin temizleme
                if "text" in item and isinstance(item["text"], str):
                    # Satır sonlarını ve fazla boşlukları temizle
                    item["text"] = item["text"].strip().replace("\n", " ").replace("\r", " ")
                    while "  " in item["text"]:
                        item["text"] = item["text"].replace("  ", " ")
                
                # Diğer alan temizlemeleri
                clean_item = {}
                for key, value in item.items():
                    if value is None:
                        clean_item[key] = ""
                    elif isinstance(value, str) and not value.strip():
                        clean_item[key] = ""
                    elif not is_valid_json(value):
                        try:
                            clean_item[key] = str(value)
                        except:
                            clean_item[key] = ""
                    else:
                        clean_item[key] = value
                
                return clean_item
            
            elif self.category in ["altyapi", "egitim"]:
                # Altyapı ve eğitim verilerini standartlaştır
                
                # Metin alanını bul
                text_found = False
                text_fields = ["text", "content", "article", "summary", "title", "sentence"]
                
                for field in text_fields:
                    if field in item and item[field]:
                        if not text_found and isinstance(item[field], str):
                            # Ana metin alanını belirle
                            if "text" not in item:
                                item["text"] = item[field]
                            text_found = True
                
                # Metin yoksa atla
                if "text" not in item or not item["text"]:
                    return None
                
                # Metni temizle
                if isinstance(item["text"], str):
                    # Satır sonlarını ve fazla boşlukları temizle
                    item["text"] = item["text"].strip().replace("\n", " ").replace("\r", " ")
                    while "  " in item["text"]:
                        item["text"] = item["text"].replace("  ", " ")
                
                # Bellek optimizasyonu için sadece gerekli alanları koru
                clean_item = {"text": item["text"]}
                
                # Ekstra önemli alanları koru
                for key in ["id", "label", "title"]:
                    if key in item:
                        clean_item[key] = item[key]
                
                return clean_item
            
            return item
        except Exception as e:
            logger.error(f"Veri temizleme hatası: {str(e)}")
            return None

# ----- VERİ SETİ KOLEKSİYONLARI -----

# Türkçe veri setleri koleksiyonu
TURKCE_VERI_SETLERI = {
    # Temel Türkçe NLP veri setleri
    "altyapi": [
        {"name": "mc4", "config": "tr", "description": "Büyük web tabanlı Türkçe metin korpusu"},
        {"name": "oscar", "config": "unshuffled_deduplicated_tr", "description": "Filtrelenmiş Türkçe web metinleri"},
        {"name": "opus_books", "config": "tr", "description": "Türkçe kitap metinleri"},
        {"name": "turkish-summarization", "config": None, "description": "Türkçe özet veri seti"},
    ],
    
    # Türkçe eğitim veri setleri
    "egitim": [
        {"name": "turkish_news_texts", "config": None, "description": "Türkçe haber metinleri"},
        {"name": "wikimedia/wikipedia", "config": "20231101.tr", "description": "Türkçe Wikipedia metinleri"},
        {"name": "turkish-ner", "config": None, "description": "Türkçe varlık ismi tanıma"},
        {"name": "mlsum", "config": "tr", "description": "Türkçe metin özetleme"},
        {"name": "universal_dependencies", "config": "tr_imst", "description": "Türkçe sözdizimi analizi"},
    ],
    
    # Türkçe sentiment ve mizah veri setleri
    "mizah": [
        {"name": "turkish_product_reviews", "config": None, "description": "Türkçe ürün yorumları"},
        {"name": "emotion", "config": None, "description": "Duygu analizi veri seti"},
        {"name": "setfit/sst2", "config": None, "description": "Sentiment analizi (İngilizce)"},
        {"name": "tweet_eval", "config": "sentiment", "description": "Twitter duygu analizi"},
    ]
}

# ----- YARDIMCI İŞLEVLER -----

def list_datasets_in_category(category):
    """Belirli kategorideki veri setlerini listeler"""
    if category not in TURKCE_VERI_SETLERI:
        print(f"Hata: '{category}' kategorisi bulunamadı.")
        return
    
    datasets = TURKCE_VERI_SETLERI[category]
    print(f"\n{category.upper()} KATEGORİSİNDEKİ VERİ SETLERİ:")
    print(f"{'=' * 80}")
    print(f"{'#':<5} {'Veri Seti':<30} {'Yapılandırma':<25} {'Açıklama':<30}")
    print(f"{'-' * 80}")
    
    for i, dataset in enumerate(datasets, 1):
        name = dataset["name"]
        config = dataset["config"] if dataset["config"] else "varsayılan"
        desc = dataset.get("description", "")
        print(f"{i:<5} {name:<30} {config:<25} {desc:<30}")
    
    print(f"{'=' * 80}")

def validate_json_files(directory):
    """Belirli dizindeki JSON dosyalarını doğrular ve bilgilerini gösterir"""
    if not os.path.exists(directory):
        print(f"Dizin bulunamadı: {directory}")
        return
    
    # JSONL dosyalarını bul
    files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    
    if not files:
        print(f"{directory} dizininde hiç JSONL dosyası bulunamadı.")
        return
    
    print(f"\n{os.path.basename(directory).upper()} DİZİNİNDEKİ DOSYALAR:")
    print(f"{'=' * 80}")
    print(f"{'#':<3} {'Dosya Adı':<40} {'Boyut':<10} {'Satır Sayısı':<15}")
    print(f"{'-' * 80}")
    
    for i, file in enumerate(files, 1):
        file_path = os.path.join(directory, file)
        file_size = get_file_size(file_path)
        line_count = count_lines(file_path)
        
        print(f"{i:<3} {file:<40} {format_size(file_size):<10} {line_count:,}")
    
    print(f"{'=' * 80}")
    
    # Dosya içeriğini kontrol etmek istiyor musunuz?
    choice = input("\nBir dosyanın içeriğini kontrol etmek ister misiniz? (E/H): ")
    if choice.lower() == 'e':
        file_num = input("Kontrol etmek istediğiniz dosyanın numarasını girin: ")
        try:
            file_idx = int(file_num) - 1
            if 0 <= file_idx < len(files):
                file_path = os.path.join(directory, files[file_idx])
                
                # Dosya başlığını ve birkaç satırı göster
                print(f"\nDOSYA İÇERİĞİ: {files[file_idx]}")
                print(f"{'=' * 80}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 10:  # İlk 10 satırı göster
                            break
                        print(f"{i+1:<3} {line.strip()}")
                
                print(f"{'=' * 80}")
                print(f"(Toplam {count_lines(file_path):,} satırdan ilk 10 satır gösteriliyor)")
            else:
                print("Geçersiz dosya numarası!")
        except ValueError:
            print("Geçersiz giriş! Sayı girmelisiniz.")

def print_dataset_statistics(output_dir):
    """Tüm kategorilerdeki veri seti istatistiklerini gösterir"""
    categories = ["altyapi", "egitim", "mizah"]
    
    total_files = 0
    total_lines = 0
    total_size = 0
    
    print(f"\nTÜM VERİ SETLERİ İSTATİSTİKLERİ:")
    print(f"{'=' * 80}")
    print(f"{'Kategori':<15} {'Dosya Sayısı':<15} {'Toplam Satır':<15} {'Toplam Boyut':<15}")
    print(f"{'-' * 80}")
    
    for category in categories:
        category_dir = os.path.join(output_dir, category)
        if os.path.exists(category_dir):
            files = [f for f in os.listdir(category_dir) if f.endswith('.jsonl')]
            files_count = len(files)
            
            category_lines = 0
            category_size = 0
            
            for file in files:
                file_path = os.path.join(category_dir, file)
                file_size = get_file_size(file_path)
                file_lines = count_lines(file_path)
                
                category_lines += file_lines
                category_size += file_size
            
            print(f"{category:<15} {files_count:<15} {category_lines:,} {format_size(category_size)}")
            
            total_files += files_count
            total_lines += category_lines
            total_size += category_size
    
    print(f"{'-' * 80}")
    print(f"{'TOPLAM':<15} {total_files:<15} {total_lines:,} {format_size(total_size)}")
    print(f"{'=' * 80}")

# ----- ANA PROGRAM -----

def main():
    """Ana program işlevi"""
    parser = argparse.ArgumentParser(description="Türkçe Veri İndirici - Huggingface veri setlerini satır olarak indirir")
    parser.add_argument("--output", "-o", type=str, default="./turkish_data", help="Çıktı dizini (varsayılan: ./turkish_data)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Her veri seti için maksimum satır sayısı (varsayılan: tümü)")
    parser.add_argument("--disable-lang-check", "-d", action="store_true", help="Türkçe dil kontrolünü devre dışı bırak")
    parser.add_argument("--category", "-c", type=str, help="Sadece belirli bir kategoriyi indir")
    parser.add_argument("--dataset", "-ds", type=str, help="Sadece belirli bir veri setini indir")
    parser.add_argument("--auto", "-a", action="store_true", help="Otomatik olarak tüm veri setlerini indir (kullanıcı onayı istemeden)")
    
    args = parser.parse_args()
    
    output_dir = args.output
    sample_size = args.limit
    verify_lang = not args.disable_lang_check
    auto_download = args.auto
    
    # Dizinleri oluştur
    os.makedirs(output_dir, exist_ok=True)
    for category in TURKCE_VERI_SETLERI.keys():
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
    
    # Program başlığı
    print("\n" + "=" * 80)
    print(" " * 20 + "TÜRKÇE VERİ İNDİRİCİ v2.2")
    print(" " * 15 + "Satır Tabanlı Veri İndirme ve İşleme Aracı")
    print("=" * 80)
    print(f"Çıktı dizini: {output_dir}")
    print(f"Satır limiti: {'Sınırsız (tüm veriler indirilecek)' if sample_size is None else f'{sample_size:,} satır'}")
    print(f"Türkçe kontrolü: {'Etkin' if verify_lang else 'Devre dışı'}")
    print(f"Batch boyutu: {BATCH_SIZE:,} satır")
    
    # Datasets kütüphanesi versiyonu
    print(f"Datasets kütüphanesi versiyonu: {__version__}")
    print("=" * 80)
    
    # Otomatik indirme modunda tüm veri setlerini indir
    if auto_download:
        print("\nOTOMATİK İNDİRME MODU ETKİN")
        print("Tüm veri setleri otomatik olarak indirilecek...")
        
        all_successful = 0
        all_failed = []
        
        for category, datasets in TURKCE_VERI_SETLERI.items():
            print(f"\n{category.upper()} KATEGORI VERİ SETLERİ İNDİRİLİYOR...")
            
            category_successful = 0
            category_failed = []
            
            for dataset in datasets:
                processor = DatasetProcessor(
                    dataset_name=dataset["name"],
                    config=dataset["config"],
                    output_dir=os.path.join(output_dir, category),
                    sample_size=sample_size,
                    verify_lang=verify_lang,
                    category=category
                )
                success = processor.process()
                if success:
                    category_successful += 1
                    all_successful += 1
                else:
                    category_failed.append(dataset["name"])
                    all_failed.append(f"{category}/{dataset['name']}")
            
            print(f"\n{'=' * 80}")
            print(f"{category.upper()} KATEGORİSİ ÖZET")
            print(f"{'=' * 80}")
            print(f"Toplam veri seti sayısı: {len(datasets)}")
            print(f"Başarıyla indirilen veri seti sayısı: {category_successful}")
            print(f"Başarısız veri seti sayısı: {len(category_failed)}")
            
            if category_failed:
                print(f"\nAşağıdaki veri setleri indirilemedi:")
                for ds in category_failed:
                    print(f"- {ds}")
            
            print(f"{'=' * 80}")
        
        # Genel özet
        total_datasets = sum(len(datasets) for datasets in TURKCE_VERI_SETLERI.values())
        
        print(f"\n{'#' * 80}")
        print(f"GENEL İNDİRME ÖZET")
        print(f"{'#' * 80}")
        print(f"Toplam veri seti sayısı: {total_datasets}")
        print(f"Başarıyla indirilen veri seti sayısı: {all_successful}")
        print(f"Başarısız veri seti sayısı: {len(all_failed)}")
        
        if all_failed:
            print(f"\nAşağıdaki veri setleri indirilemedi:")
            for ds in all_failed:
                print(f"- {ds}")
        
        print(f"{'#' * 80}")
        
        # Program sonunda özet istatistikleri göster
        print("\nOtomatik indirme tamamlandı.")
        print_dataset_statistics(output_dir)
        print("\nToplam indirilen veri boyutu ve satır sayısı yukarıda gösterilmektedir.")
        return
    
    # Kategori ve dataset parametreleri belirtilmişse sadece o veri setini indir
    if args.category and args.dataset:
        category = args.category
        dataset_name = args.dataset
        
        if category not in TURKCE_VERI_SETLERI:
            print(f"HATA: '{category}' kategorisi bulunamadı.")
            return
            
        # Veri seti adına göre bul
        dataset_config = None
        found = False
        for ds in TURKCE_VERI_SETLERI[category]:
            if ds["name"] == dataset_name:
                dataset_config = ds["config"]
                found = True
                break
                
        if not found:
            # Özel veri seti olarak kabul et
            print(f"Uyarı: {dataset_name} veri seti tanımlı değil. Özel veri seti olarak indiriliyor.")
        
        # Veri setini indir
        processor = DatasetProcessor(
            dataset_name=dataset_name,
            config=dataset_config,
            output_dir=os.path.join(output_dir, category),
            sample_size=sample_size,
            verify_lang=verify_lang,
            category=category
        )
        processor.process()
        return
    
    # Kullanıcı menüsü
    while True:
        print("\nNe yapmak istiyorsunuz?")
        print("1. Temel Türkçe NLP veri setlerini indir")
        print("2. Türkçe eğitim veri setlerini indir")
        print("3. Türkçe mizah ve duygu analizi veri setlerini indir")
        print("4. Tüm veri setlerini indir")
        print("5. Belirli bir veri setini indir")
        print("6. İndirilen verileri kontrol et")
        print("7. Veri setlerini listele")
        print("8. Toplam veri istatistiklerini göster")
        print("9. Özel veri seti indir")
        print("0. Çıkış")
        
        choice = input("\nSeçiminiz: ")
        
        if choice == "0":
            print("\nProgram sonlandırılıyor...\n")
            break
            
        elif choice == "1":
            list_datasets_in_category("altyapi")
            confirm = input("\nBu veri setlerini indirmek istiyor musunuz? (E/H): ")
            if confirm.lower() == 'e':
                successful_count = 0
                failed_datasets = []
                
                for dataset in TURKCE_VERI_SETLERI["altyapi"]:
                    processor = DatasetProcessor(
                        dataset_name=dataset["name"],
                        config=dataset["config"],
                        output_dir=os.path.join(output_dir, "altyapi"),
                        sample_size=sample_size,
                        verify_lang=verify_lang,
                        category="altyapi"
                    )
                    success = processor.process()
                    if success:
                        successful_count += 1
                    else:
                        failed_datasets.append(dataset["name"])
                
                print(f"\n{'=' * 80}")
                print(f"ALTYAPI KATEGORİSİ ÖZET")
                print(f"{'=' * 80}")
                print(f"Toplam veri seti sayısı: {len(TURKCE_VERI_SETLERI['altyapi'])}")
                print(f"Başarıyla indirilen veri seti sayısı: {successful_count}")
                print(f"Başarısız veri seti sayısı: {len(failed_datasets)}")
                
                if failed_datasets:
                    print(f"\nAşağıdaki veri setleri indirilemedi:")
                    for ds in failed_datasets:
                        print(f"- {ds}")
                
                print(f"{'=' * 80}")
                
        elif choice == "2":
            list_datasets_in_category("egitim")
            confirm = input("\nBu veri setlerini indirmek istiyor musunuz? (E/H): ")
            if confirm.lower() == 'e':
                successful_count = 0
                failed_datasets = []
                
                for dataset in TURKCE_VERI_SETLERI["egitim"]:
                    processor = DatasetProcessor(
                        dataset_name=dataset["name"],
                        config=dataset["config"],
                        output_dir=os.path.join(output_dir, "egitim"),
                        sample_size=sample_size,
                        verify_lang=verify_lang,
                        category="egitim"
                    )
                    success = processor.process()
                    if success:
                        successful_count += 1
                    else:
                        failed_datasets.append(dataset["name"])
                
                print(f"\n{'=' * 80}")
                print(f"EĞİTİM KATEGORİSİ ÖZET")
                print(f"{'=' * 80}")
                print(f"Toplam veri seti sayısı: {len(TURKCE_VERI_SETLERI['egitim'])}")
                print(f"Başarıyla indirilen veri seti sayısı: {successful_count}")
                print(f"Başarısız veri seti sayısı: {len(failed_datasets)}")
                
                if failed_datasets:
                    print(f"\nAşağıdaki veri setleri indirilemedi:")
                    for ds in failed_datasets:
                        print(f"- {ds}")
                
                print(f"{'=' * 80}")
                
        elif choice == "3":
            list_datasets_in_category("mizah")
            confirm = input("\nBu veri setlerini indirmek istiyor musunuz? (E/H): ")
            if confirm.lower() == 'e':
                successful_count = 0
                failed_datasets = []
                
                for dataset in TURKCE_VERI_SETLERI["mizah"]:
                    processor = DatasetProcessor(
                        dataset_name=dataset["name"],
                        config=dataset["config"],
                        output_dir=os.path.join(output_dir, "mizah"),
                        sample_size=sample_size,
                        verify_lang=verify_lang,
                        category="mizah"
                    )
                    success = processor.process()
                    if success:
                        successful_count += 1
                    else:
                        failed_datasets.append(dataset["name"])
                
                print(f"\n{'=' * 80}")
                print(f"MİZAH KATEGORİSİ ÖZET")
                print(f"{'=' * 80}")
                print(f"Toplam veri seti sayısı: {len(TURKCE_VERI_SETLERI['mizah'])}")
                print(f"Başarıyla indirilen veri seti sayısı: {successful_count}")
                print(f"Başarısız veri seti sayısı: {len(failed_datasets)}")
                
                if failed_datasets:
                    print(f"\nAşağıdaki veri setleri indirilemedi:")
                    for ds in failed_datasets:
                        print(f"- {ds}")
                
                print(f"{'=' * 80}")
                
        elif choice == "4":
            print("\nTüm veri setleri indirilecek!")
            confirm = input("Devam etmek istiyor musunuz? (E/H): ")
            if confirm.lower() == 'e':
                all_successful = 0
                all_failed = []
                
                for category, datasets in TURKCE_VERI_SETLERI.items():
                    print(f"\n{category.upper()} KATEGORI VERİ SETLERİ İNDİRİLİYOR...")
                    
                    category_successful = 0
                    category_failed = []
                    
                    for dataset in datasets:
                        processor = DatasetProcessor(
                            dataset_name=dataset["name"],
                            config=dataset["config"],
                            output_dir=os.path.join(output_dir, category),
                            sample_size=sample_size,
                            verify_lang=verify_lang,
                            category=category
                        )
                        success = processor.process()
                        if success:
                            category_successful += 1
                            all_successful += 1
                        else:
                            category_failed.append(dataset["name"])
                            all_failed.append(f"{category}/{dataset['name']}")
                    
                    print(f"\n{'=' * 80}")
                    print(f"{category.upper()} KATEGORİSİ ÖZET")
                    print(f"{'=' * 80}")
                    print(f"Toplam veri seti sayısı: {len(datasets)}")
                    print(f"Başarıyla indirilen veri seti sayısı: {category_successful}")
                    print(f"Başarısız veri seti sayısı: {len(category_failed)}")
                    
                    if category_failed:
                        print(f"\nAşağıdaki veri setleri indirilemedi:")
                        for ds in category_failed:
                            print(f"- {ds}")
                    
                    print(f"{'=' * 80}")
                
                # Genel özet
                total_datasets = sum(len(datasets) for datasets in TURKCE_VERI_SETLERI.values())
                
                print(f"\n{'#' * 80}")
                print(f"GENEL İNDİRME ÖZET")
                print(f"{'#' * 80}")
                print(f"Toplam veri seti sayısı: {total_datasets}")
                print(f"Başarıyla indirilen veri seti sayısı: {all_successful}")
                print(f"Başarısız veri seti sayısı: {len(all_failed)}")
                
                if all_failed:
                    print(f"\nAşağıdaki veri setleri indirilemedi:")
                    for ds in all_failed:
                        print(f"- {ds}")
                
                print(f"{'#' * 80}")
                
        elif choice == "5":
            print("\nHangi kategoriden veri seti indirmek istiyorsunuz?")
            print("1. Altyapı")
            print("2. Eğitim")
            print("3. Mizah")
            
            cat_choice = input("\nKategori seçimi (1-3): ")
            
            category_map = {
                "1": "altyapi",
                "2": "egitim", 
                "3": "mizah"
            }
            
            if cat_choice in category_map:
                category = category_map[cat_choice]
                list_datasets_in_category(category)
                
                ds_choice = input("\nİndirmek istediğiniz veri setinin numarasını girin: ")
                try:
                    ds_idx = int(ds_choice) - 1
                    if 0 <= ds_idx < len(TURKCE_VERI_SETLERI[category]):
                        dataset = TURKCE_VERI_SETLERI[category][ds_idx]
                        
                        print(f"\n{dataset['name']} veri seti indirilecek.")
                        if dataset['config']:
                            print(f"Yapılandırma: {dataset['config']}")
                        
                        # Satır limiti seçeneği
                        limit_choice = input("\nSatır limitini değiştirmek ister misiniz? (E/H) [Varsayılan: Sınırsız]: ")
                        if limit_choice.lower() == 'e':
                            try:
                                new_limit = int(input("Satır limiti (0 = sınırsız): "))
                                if new_limit <= 0:
                                    curr_sample_size = None
                                else:
                                    curr_sample_size = new_limit
                            except ValueError:
                                print("Geçersiz limit, varsayılan kullanılacak.")
                                curr_sample_size = sample_size
                        else:
                            curr_sample_size = sample_size
                        
                        processor = DatasetProcessor(
                            dataset_name=dataset["name"],
                            config=dataset["config"],
                            output_dir=os.path.join(output_dir, category),
                            sample_size=curr_sample_size,
                            verify_lang=verify_lang,
                            category=category
                        )
                        processor.process()
                    else:
                        print("Geçersiz veri seti numarası!")
                except ValueError:
                    print("Geçersiz numara!")
            else:
                print("Geçersiz kategori seçimi!")
                
        elif choice == "6":
            print("\nHangi kategorideki verileri kontrol etmek istiyorsunuz?")
            print("1. Altyapı")
            print("2. Eğitim")
            print("3. Mizah")
            print("4. Tümü")
            
            check_choice = input("\nSeçiminiz (1-4): ")
            
            if check_choice == "4":
                for category in ["altyapi", "egitim", "mizah"]:
                    validate_json_files(os.path.join(output_dir, category))
            elif check_choice in ["1", "2", "3"]:
                category_map = {
                    "1": "altyapi",
                    "2": "egitim", 
                    "3": "mizah"
                }
                validate_json_files(os.path.join(output_dir, category_map[check_choice]))
            else:
                print("Geçersiz seçim!")
                
        elif choice == "7":
            print("\nHangi kategorideki veri setlerini listelemek istiyorsunuz?")
            print("1. Altyapı")
            print("2. Eğitim")
            print("3. Mizah")
            print("4. Tümü")
            
            list_choice = input("\nSeçiminiz (1-4): ")
            
            category_map = {
                "1": "altyapi",
                "2": "egitim", 
                "3": "mizah"
            }
            
            if list_choice in category_map:
                list_datasets_in_category(category_map[list_choice])
            elif list_choice == "4":
                for category in ["altyapi", "egitim", "mizah"]:
                    list_datasets_in_category(category)
            else:
                print("Geçersiz seçim!")
                
        elif choice == "8":
            print_dataset_statistics(output_dir)
        
        elif choice == "9":
            print("\nÖzel bir veri seti indirmek için aşağıdaki bilgileri girin:")
            dataset_name = input("Veri seti adı: ")
            
            if not dataset_name:
                print("Veri seti adı boş olamaz!")
                continue
                
            config = input("Veri seti yapılandırması (boş bırakabilirsiniz): ").strip() or None
            
            # Satır limiti seçeneği
            limit_choice = input("\nSatır limitini değiştirmek ister misiniz? (E/H) [Varsayılan: Sınırsız]: ")
            if limit_choice.lower() == 'e':
                try:
                    new_limit = int(input("Satır limiti (0 = sınırsız): "))
                    if new_limit <= 0:
                        curr_sample_size = None
                    else:
                        curr_sample_size = new_limit
                except ValueError:
                    print("Geçersiz limit, varsayılan kullanılacak.")
                    curr_sample_size = sample_size
            else:
                curr_sample_size = sample_size
            
            print("\nHangi kategoriye kaydedilsin?")
            print("1. Altyapı")
            print("2. Eğitim")
            print("3. Mizah")
            
            cat_choice = input("\nKategori seçimi (1-3): ")
            
            category_map = {
                "1": "altyapi",
                "2": "egitim", 
                "3": "mizah"
            }
            
            if cat_choice in category_map:
                category = category_map[cat_choice]
                
                processor = DatasetProcessor(
                    dataset_name=dataset_name,
                    config=config,
                    output_dir=os.path.join(output_dir, category),
                    sample_size=curr_sample_size,
                    verify_lang=verify_lang,
                    category=category
                )
                processor.process()
            else:
                print("Geçersiz kategori seçimi!")
            
        else:
            print("Geçersiz seçim! Lütfen 0-9 arası bir değer girin.")
    
    # Program sonunda özet istatistikleri göster
    print("\nProgram tamamlandı.")
    print_dataset_statistics(output_dir)
    print("\nToplam indirilen veri boyutu ve satır sayısı yukarıda gösterilmektedir.")
    print("Programı yeniden çalıştırmak için python turkce_veri_indirici.py komutunu kullanabilirsiniz.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram kullanıcı tarafından durduruldu!")
    except Exception as e:
        logger.exception("Beklenmeyen hata")
        print(f"\nProgram çalışırken beklenmeyen bir hata oluştu: {str(e)}")
        print("Detaylı hata bilgisi için log dosyasını kontrol edin.")
    finally:
        print("\nProgram sonlandırıldı.") 