# OCT Preprocessing & Semantic Segmentation Labeling App (Fase 1)

Aplikasi untuk melakukan *Auto-Cropping* Region of Interest (ROI `512x512` piksel) pada citra OCT mentah dan anotasi piksel (`Class 1 = Permukaan`, `Class 2 = Batas Bawah`, piksel murni `0, 1, 2`) baik di **Tablet Android (Redmi Pad 2 + Stylus Pen)** maupun di **Desktop Windows (PyQt6)**.

---

## A. Cara Menggunakan di Tablet Android (Redmi Pad 2 + Stylus Pen)

Tersedia **2 cara** yang bisa Anda pilih sesuai kenyamanan:

### Cara 1: Kirim 1 File & Pakai 100% Mandiri (Offline) di Redmi Pad 2
1. Kirim file **[`OCT_Labeler_RedmiPad2.html`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/OCT_Labeler_RedmiPad2.html)** (serta folder foto OCT Anda) ke **Redmi Pad 2** melalui WhatsApp, Telegram, Google Drive, atau kabel USB.
2. Buka file `OCT_Labeler_RedmiPad2.html` menggunakan **Google Chrome** di Redmi Pad 2.
3. Ketuk tombol **"📂 Pilih Gambar OCT"** untuk memuat banyak gambar sekaligus dari memori Redmi Pad 2.
4. Aktifkan centang **"Mode Khusus Stylus Pen"** agar telapak tangan/jari Anda tidak ikut mencoret saat menggambar dengan Pen (dan jari otomatis berfungsi untuk menggeser/zoom kanvas).
5. Setelah selesai melabeli, ketuk **"📦 Download Semua Dataset (.ZIP)"** — Anda akan mendapatkan file `OCT_Labeled_Dataset_Fase1.zip` yang sudah berisi `images/*_crop.png` dan `masks/*_mask.png` (format PNG Grayscale 8-bit murni bernilai `0, 1, 2`).

### Cara 2: Install sebagai Aplikasi (PWA) & Sinkronisasi Langsung ke Laptop via Wi-Fi
Jika Anda ingin semua gambar OCT tetap berada di Laptop dan hasil coretan Pen di Redmi Pad 2 langsung tersimpan ke folder Laptop:
1. Pastikan **Laptop** dan **Redmi Pad 2** terhubung ke Wi-Fi atau Hotspot HP yang sama.
2. Di Laptop, double-click **[`run_tablet_server.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/run_tablet_server.bat)**.
3. Lihat alamat IP yang muncul di layar hitam Laptop (misal `http://192.168.1.10:8080`), lalu buka alamat tersebut di **Google Chrome pada Redmi Pad 2**.
4. Ketuk tombol **"Install App"** (atau menu titik tiga Chrome `⋮` → **"Tambahkan ke layar utama" / "Install aplikasi"**) agar aplikasi terpasang di layar depan Redmi Pad 2 secara *Full Screen*.
5. Setiap kali Anda menekan **"💾 Simpan Gambar Ini"** di Redmi Pad 2, file `*_crop.png` dan `*_mask.png` langsung tersimpan otomatis di folder Output Laptop!

---

## B. Cara Menggunakan di Desktop Windows (PyQt6)

- Double-click **[`run_app.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/run_app.bat)** untuk membuka aplikasi desktop [label_app.py](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/label_app.py).
- Double-click **[`verify_dataset.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/verify_dataset.bat)** untuk memvalidasi dimensi `512x512` dan nilai piksel `{0, 1, 2}` pada semua file mask yang sudah dibuat.
