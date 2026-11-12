# OCT Preprocessing & Semantic Segmentation Labeling App (v2.1 — Spatial-to-Temporal Domain Adaptation)

An interactive preprocessing and multi-class boundary annotation suite for **Optical Coherence Tomography (OCT)** skin imaging, designed for the **"Train on Spatial (3D B-Scan), Infer on Temporal (M-Scan)"** domain adaptation strategy.

The application extracts a consistent `512×512` pixel Region of Interest (ROI) and provides a **2px default brush thickness** to annotate **three anatomical skin layer boundaries** without overlapping thin layers (such as the *Stratum Corneum*):
- **Class 1 (`Top SC` — Skin Surface / Top of Stratum Corneum)**: **Red** line (mask pixel value = `1`)
- **Class 2 (`Bot SC` — Bottom of Stratum Corneum)**: **Yellow** line (mask pixel value = `2`)
- **Class 3 (`End ED` — End of Epidermis / Dermis Signal Boundary)**: **Green** line (mask pixel value = `3`)
- **Background / Eraser**: Mask pixel value = `0`

---

## What's New in Version 2.1

1. **`1 Subject = 4 Points, 1 Sample per Point` Sampling Workflow (`🎯 Mode 1 Sampel per Titik`)**:
   - **Combined Multi-Point Mode**: Select a parent dataset folder (or click **`📂 Gabung Semua Titik (1 Sampel/Titik)`**) to automatically scan all point folders (e.g., `alefootR.bin`, `alefootL.bin`, `alehandR.bin`, `alehandL.bin`) and load **strictly 1 sample per point** into a single unified queue.
   - **Single-Point Mode with Direct Sibling Navigation**: When opening a single point folder, the app displays the 1 active sample for that point (`Target: 1 / 1 sample`) and provides **`⏮ Titik Sblm` / `Titik Brkt ⏭`** buttons to switch directly to the next point folder without re-browsing.
   - **Non-Saving Random Reroll (`🎲 Ganti Slice Acak (Titik Ini) [R]`)**: Pressing the random slice button rejects the current slice if its layer morphology is unclear, **does not auto-save** the rejected slice, removes any previously saved output for that point, and resets the point status to **Unfinished (`[ ]`, `0 / 1`)**.
2. **Label Revision & Auto-Remove/Replace Workflow (`✏️ Mode Revisi`)**:
   - **Automatic Removal & Replacement on Save**: When revising an already-saved label (`[✓]`), clicking **`💾 SIMPAN REVISI (GANTI FILE LAMA)`** automatically deletes all previously saved files (`*_crop.png`, `*_mask.png`, `*_overlay.png`, `*_strip.png`, `*_strip_mask.png`) for that point before writing the newly revised image and mask—guaranteeing zero duplicate or leftover files in the dataset.
   - **Delete Saved Output & Revise from Scratch (`🗑️ Hapus Gambar Tersimpan (Revisi Ulang)`)**: Immediately removes the previously saved image and mask files from disk, clears the canvas mask, and resets the sample status back to `[ ]` (Unfinished).
   - **Per-Point Slice Selector (`🎞️ Pilih / Revisi Slice Titik Ini`)**: Allows you to switch to any specific slice (`1..N`) within the active point folder to replace the previously saved slice.
3. **Default `2px` Brush Thickness**:
   - Configured to `2px` by default in both the Tablet Web Server (`tablet_server.py`) and Desktop PyQt6 App (`label_app.py`) to keep `Top SC` and `Bot SC` cleanly separated on thin *Stratum Corneum* regions while preserving spatial gradient continuity for U-Net training.
4. **Standard U-Net Dataset Export**:
   - Exports paired `512×512` B-Scan crops (`images/*_crop.png`), pure 8-bit grayscale masks `{0, 1, 2, 3}` (`masks/*_mask.png`), and MATLAB-style reference visualizations (`overlays/*_overlay.png`).

---

## A. Running on Laptop & Accessing via Tablet over Wi-Fi (No Dataset Download Required)

In this mode, **your entire OCT dataset stays on your Laptop's hard drive**. Your Tablet accesses the annotation canvas via a mobile browser over Wi-Fi/Hotspot, and every save or revision writes directly to your Laptop's output directory.

### Steps:
1. Connect both your **Laptop** and **Tablet** to the same **Wi-Fi network** or **Mobile Hotspot**.
2. On your Laptop, double-click **[`run_tablet_server.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/run_tablet_server.bat)** (which launches [`tablet_server.py`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/tablet_server.py)).
3. Check the IP address displayed in the Laptop terminal window (e.g., `http://192.168.1.10:8080`) and open that URL in **Google Chrome / Safari** on your Tablet.
4. On your Tablet screen:
   - Tap **"Pilih Folder Sumber"** to browse your Laptop's drives remotely and select either a point folder or the parent dataset folder.
   - Enable **"Mode Khusus Stylus Pen (Jari = Geser)"** for palm rejection (your stylus draws the layer boundaries while your finger pans the zoomed canvas).
   - Use the **Zoom buttons (`100%`, `150%`, `200%`, `300%`)** for precise boundary tracing.
   - Tap **"💾 SIMPAN KE LAPTOP"** (or **"💾 SIMPAN REVISI (GANTI FILE LAMA)"** when revising an existing sample) to save `*_crop.png`, `*_mask.png`, and `overlays/*_overlay.png` directly to your Laptop.

> **Windows Firewall Note:** On first launch of `run_tablet_server.bat`, if a *Windows Defender Firewall* prompt appears, check both **Private networks** and **Public networks** and click **Allow access** so your Tablet can reach the Laptop server.

---

## B. Running Locally on Windows Desktop (PyQt6)

- Double-click **[`run_app.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/run_app.bat)** to launch the native desktop application [`label_app.py`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/label_app.py).
- Double-click **[`verify_dataset.bat`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/verify_dataset.bat)** (which runs [`verify_dataset.py`](file:///d:/kuliah/TEEP/preprocessing_labeling_apps/verify_dataset.py)) to verify that all generated masks are `512×512` single-channel 8-bit PNGs containing only valid class IDs `{0, 1, 2, 3}`.

---

## Keyboard Shortcuts (Desktop App)

| Key | Action |
| :--- | :--- |
| `1` | Select Class 1: `Top SC` (Red) |
| `2` | Select Class 2: `Bot SC` (Yellow) |
| `3` | Select Class 3: `End ED` (Green) |
| `0` / Right-Click Drag | Select `Eraser` (Background = `0`) |
| `R` | Random Slice Reroll for Current Point (Rejects current slice without saving) |
| `Ctrl + Z` | Undo last brush stroke |
| `C` | Clear current canvas mask |
| `Space` | Toggle live mask overlay visibility |
| `Ctrl + S` | Save / Replace Revision (`_crop.png` + `_mask.png`) |
| `A` / `D` | Previous / Next sample |
