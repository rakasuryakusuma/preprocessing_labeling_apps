# OCT Preprocessing & Semantic Segmentation Labeling App (v2 — Spatial-to-Temporal Domain Adaptation)

Aplikasi pelabelan citra Optical Coherence Tomography (OCT) untuk strategi **"Train on Spatial (B-Scan 3D), Infer on Temporal (M-Scan)"** menggunakan Region of Interest (ROI `512x512` piksel) dan anotasi **3 garis batas lapisan** dengan ketebalan **Brush `3px`** (mencegah *vanishing gradient* / *information loss* saat melewati *pooling layer* di U-Net):
- **Class 1 (`Top SC` - Permukaan Kulit / Batas Atas Stratum Corneum)**: Garis **Merah** (nilai piksel mask = `1`)
- **Class 2 (`Bot SC` - Batas Bawah Stratum Corneum)**: Garis **Kuning** (nilai piksel mask = `2`)
- **Class 3 (`End ED` - Batas Akhir Pantulan Sinyal di Epidermis/Dermis)**: Garis **Hijau** (nilai piksel mask = `3`)
- **Background / Eraser**: Nilai piksel mask = `0`

### Fitur Utama Aplikasi v2:
1. **Default Brush `3px`**: Ketebalan `3px` diaktifkan secara default baik di aplikasi Desktop (`PyQt6`) maupun Server Tablet untuk menjaga sinyal gradien spasial pada arsitektur U-Net (ResNet34 encoder).
2. **Mode `🎯 1 Sampel per Titik` (1 Subjek = 4 Titik, 1 Sampel/Titik)**:
   - **Mode Gabung Semua Titik**: Cukup pilih folder induk dataset (atau klik tombol **`📂 Gabung Semua Titik (1 Sampel/Titik)`**), maka aplikasi otomatis memindai semua folder titik (misal `alefootR.bin`, `alefootL.bin`, `alehandR.bin`, `alehandL.bin`) dan mengambil **tepat 1 sampel acak dari setiap titik** ke dalam 1 daftar kerja.
   - **Mode 1 Titik**: Jika membuka 1 folder titik saja, target otomatis diset **`1 / 1 sampel selesai ✓`**, langsung memilihkan 1 *slice* acak saat dibuka, dan menyediakan tombol **`⏮ Titik Sblm` / `Titik Brkt ⏭`** untuk langsung berpindah ke titik berikutnya tanpa harus *browse* ulang.
   - **Ganti Slice Acak per Titik (`🎲 Ganti Slice Acak (Titik Ini) [R]`)**: Mengundi ulang sampel untuk titik aktif dari daftar *slice* di folder titik tersebut.
   - **Penamaan Bebas Bentrok**: Hasil simpan otomatis diberi awalan nama titik (`{nama_titik}_{nomor_slice}_crop.png` & `_mask.png`, contoh: `alefootR_005_crop.png`) saat disimpan ke satu folder output gabungan.
3. **Output Dataset Standar U-Net**: Menyimpan pasangan citra B-Scan `512x512` (`images/*_crop.png`), mask 8-bit murni `{0, 1, 2, 3}` (`masks/*_mask.png`), serta visualisasi referensi (`overlays/*_overlay.png`).

---

## A. Cara Run di Laptop & Akses dari Tablet (Tanpa Download Dataset ke Tablet)

Dengan mode ini, **seluruh dataset gambar OCT yang besar tetap berada di hard disk Laptop**. Tablet Anda hanya membuka antarmuka kanvas melalui browser (Wi-Fi/Hotspot), dan setiap kali Anda menekan Simpan di Tablet, file `*_crop.png` dan `*_mask.png` langsung tersimpan otomatis di folder Laptop.

### Langkah-langkah:
1. Pastikan **Laptop** dan **Tablet** terhubung ke jaringan **Wi-Fi** atau **Hotspot HP** yang sama.
2. Di Laptop, double-click **[`run_tablet_server.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/run_tablet_server.bat)** (yang akan menjalankan [`tablet_server.py`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/tablet_server.py)).
3. Lihat alamat IP yang muncul di layar terminal Laptop (contoh: `http://192.168.1.10:8080`), lalu ketik alamat tersebut di **Google Chrome / Browser pada Tablet**.
4. Di layar Tablet:
   - Ketuk tombol **"Pilih Folder Sumber"** untuk menjelajahi folder di Laptop Anda langsung dari layar Tablet, lalu pilih folder dataset OCT mentah Anda.
   - Aktifkan **"Mode Khusus Stylus Pen (Jari = Geser)"** agar telapak tangan/jari tidak ikut mencoret saat menggambar dengan Stylus Pen (jari otomatis berfungsi untuk menggeser kanvas saat di-zoom).
   - Gunakan tombol **Zoom (`100%`, `150%`, `200%`, `300%`)** untuk memperbesar area lapisan OCT agar lebih presisi.
   - Ketuk **"💾 SIMPAN KE LAPTOP"** (atau aktifkan **"Auto-Save saat pindah gambar"**) — hasil `*_crop.png`, `*_mask.png` (PNG Grayscale 8-bit murni bernilai `0, 1, 2, 3`), serta visualisasi `overlays/*_overlay.png` (lengkap dengan garis Merah/Kuning/Hijau dan kotak legenda `top SC`, `bot SC`, `end ED`) langsung tersimpan di hard disk Laptop!

> **Catatan Firewall Windows:** Saat pertama kali menjalankan `run_tablet_server.bat`, jika muncul pop-up *Windows Defender Firewall*, centang **Private networks** dan **Public networks** lalu klik **Allow access** agar Tablet dapat mengakses Laptop.

---

## B. Cara Menggunakan Langsung di Desktop Windows (PyQt6)

- Double-click **[`run_app.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/run_app.bat)** untuk membuka aplikasi desktop [`label_app.py`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/label_app.py).
- Double-click **[`verify_dataset.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/verify_dataset.bat)** (yang menjalankan [`verify_dataset.py`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/verify_dataset.py)) untuk memvalidasi dimensi `512x512` dan nilai piksel `{0, 1, 2, 3}` pada semua file mask yang sudah dibuat.
