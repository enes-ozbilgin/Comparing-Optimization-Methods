import tensorflow as tf
from PIL import Image
import os
import numpy as np

# 1. Fashion MNIST Veri Setini Yükle
(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Tişört ve Pantolon Görsellerini Filtrele
# Label 0 = Tişört, Label 1 = Pantolon
tshirt_images = x_train[y_train == 0][:100]  # İlk 100 tişört
pant_images = x_train[y_train == 1][:100]    # İlk 100 pantolon

# Çıkış klasörlerini oluştur
output_dir_png = "fashion_mnist_png"
output_dir_pgm = "fashion_mnist_pgm"
os.makedirs(output_dir_png, exist_ok=True)
os.makedirs(output_dir_pgm, exist_ok=True)

# 3. PNG ve PGM Dosyalarını Kaydetme Fonksiyonu
def save_images(images, label_name):
    for i, img in enumerate(images):
        # Görüntüyü normalize edip uint8 formatına çevir
        img_normalized = (img / 255.0) * 255
        img_normalized = img_normalized.astype('uint8')

        # PNG olarak kaydet
        png_filename = os.path.join(output_dir_png, f"{label_name}_{i+1}.png")
        im = Image.fromarray(img_normalized)
        im.save(png_filename)

        # PGM olarak kaydet
        pgm_filename = os.path.join(output_dir_pgm, f"{label_name}_{i+1}.pgm")
        im.save(pgm_filename)

        print(f"Görüntü {i+1} kaydedildi: {png_filename} ve {pgm_filename}")

# 4. Tişört ve Pantolon Görsellerini Kaydet
print("Tişört görselleri kaydediliyor...")
save_images(tshirt_images, "tshirt")

print("Pantolon görselleri kaydediliyor...")
save_images(pant_images, "pantolon")

# İşlem Tamamlandı
print(f"Tişört ve pantolon görselleri PNG ve PGM formatında kaydedildi. Klasörler:")
print(f"- PNG: {output_dir_png}")
print(f"- PGM: {output_dir_pgm}")
