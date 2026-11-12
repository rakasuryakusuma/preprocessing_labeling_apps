# OCT Skin Layer Preprocessing & Boundary Annotation Suite

[![Version](https://img.shields.io/badge/version-v2.1.0-blue.svg)](https://github.com/rakasuryakusuma/preprocessing_labeling_apps/releases)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-PyQt6%20%7C%20HTML5%20Tablet%20Canvas-41CD52.svg)](#6-usage-guide)
[![License](https://img.shields.io/badge/license-BSD--3--Clause%20%2B%20Academic%20Citation-orange.svg)](LICENSE)

An interactive preprocessing and multi-class semantic boundary annotation framework for **Optical Coherence Tomography (OCT)** skin imaging. Purpose-built to support **Spatial-to-Temporal Domain Adaptation** (*Train on Spatial 3D B-Scans, Infer on Temporal M-Scans*) for automated epidermal and *Stratum Corneum* thickness quantification.

---

## Table of Contents

1. [Scientific Overview](#1-scientific-overview)
2. [Key Features (v2.1.0)](#2-key-features-v210)
3. [Annotation Classes & Specifications](#3-annotation-classes--specifications)
4. [Dataset Output Structure](#4-dataset-output-structure)
5. [Installation](#5-installation)
6. [Usage Guide](#6-usage-guide)
   - [Mode A: Laptop Server + Remote Tablet Stylus (Wi-Fi)](#mode-a-laptop-server--remote-tablet-stylus-wi-fi)
   - [Mode B: Native Windows Desktop Application (PyQt6)](#mode-b-native-windows-desktop-application-pyqt6)
   - [Dataset Integrity Verification](#dataset-integrity-verification)
7. [Keyboard Shortcuts](#7-keyboard-shortcuts)
8. [Version History (Changelog)](#8-version-history-changelog)
9. [Citation & Academic Use](#9-citation--academic-use)
10. [License](#10-license)

---

## 1. Scientific Overview

In Optical Coherence Tomography (OCT) skin thickness analysis, the deployment target is often a **Temporal M-Scan** (500+ frames captured at a single static location over time). However, training a deep Convolutional Neural Network (e.g., U-Net with a ResNet-34 encoder) directly on static temporal slices suffers from severe **lack of morphological variance** and **data leakage**.

This suite implements the **"Train on Spatial, Infer on Temporal"** paradigm:
- **Spatial B-Scan Training (`512×512` ROI)**: The model is trained on full 2D spatial B-Scan slices sampled across multiple anatomical points per subject. These images exhibit rich morphological variance (dermatoglyphic ridges and valleys), forcing the network to learn the vertical optical reflectance transitions ($\text{Air} \rightarrow \text{Stratum Corneum} \rightarrow \text{Epidermis/Dermis}$).
- **Stratified Point Sampling (`1 Subject = 4 Points, 1 Sample/Point`)**: Rather than labeling hundreds of redundant adjacent frames, the tool enforces a strict **1 representative sample per anatomical point** rule, maximizing morphological diversity across subjects with minimal manual annotation overhead.

---

## 2. Key Features (v2.1.0)

* **Stratified `1 Sample per Point` Workflow**:
  * **Unified Multi-Point Queue**: Point the application to a parent dataset directory (or click **`📂 Gabung Semua Titik (1 Sampel/Titik)`**) to automatically scan all anatomical point subfolders (e.g., `footR.bin`, `footL.bin`, `handR.bin`, `handL.bin`) and queue **strictly 1 sample per point**.
  * **Single-Point Mode with Direct Sibling Navigation**: When opening an individual point folder, the application sets the point target to `1 / 1 sample` and provides **`⏮ Titik Sblm` / `Titik Brkt ⏭`** buttons to jump directly between sibling point folders.
  * **Non-Saving Random Slice Reroll (`🎲 Ganti Slice Acak [R]`)**: Rejects slices with poor optical structure without triggering auto-save, automatically removes any previously saved files for that point, tracks rejected slices to prevent immediate repeats, and resets the point status to **Unfinished (`[ ]`, `0 / 1`)**.
* **Label Revision & Auto-Replace Engine (`✏️ Mode Revisi`)**:
  * **Atomic Overwrite on Revision**: When editing an already-labeled sample (`[✓]`), clicking **`💾 SIMPAN REVISI (GANTI FILE LAMA)`** automatically purges all previously exported files (`*_crop.png`, `*_mask.png`, `*_overlay.png`, `*_strip.png`, `*_strip_mask.png`) for that point before writing the updated image and mask—preventing duplicate or orphaned files in your training set.
  * **One-Click Reset (`🗑️ Hapus Gambar Tersimpan`)**: Deletes the saved files of the current point from disk, clears the canvas mask, and resets its status to `[ ]` for a clean re-annotation.
  * **Per-Point Slice Selector (`🎞️ Pilih / Revisi Slice Titik Ini`)**: Switch to any specific slice (`1..N`) within the active point folder to replace the previously selected sample.
* **Dual Deployment Architecture**:
  * **Remote Tablet Stylus Server (`tablet_server.py`)**: Keep large OCT datasets on your Laptop/Workstation while annotating from a Tablet browser (iPad / Android Tablet) over local Wi-Fi with hardware **Stylus Palm Rejection** (`Stylus = Draw`, `Finger Touch = Pan Canvas`).
  * **Desktop Native GUI (`label_app.py`)**: Full-featured PyQt6 desktop interface with real-time Bresenham line interpolation and keyboard-driven navigation.

---

## 3. Annotation Classes & Specifications

| Class ID | Layer Boundary Name | Overlay Color | Description |
| :---: | :--- | :---: | :--- |
| **`0`** | `Background` | Transparent | Unlabeled background pixels / Eraser tool |
| **`1`** | `Top SC` | **Red** (`#FF2D2D`) | Skin surface / Upper boundary of the *Stratum Corneum* |
| **`2`** | `Bot SC` | **Yellow** (`#FFE600`) | Lower boundary of the *Stratum Corneum* (SC–Epidermis junction) |
| **`3`** | `End ED` | **Green** (`#00E640`) | Lower boundary of the Epidermis / Dermis attenuation limit |

- **ROI Dimensions**: `512 × 512` pixels (locked horizontal center $X$ with configurable vertical alignment).
- **Default Brush Thickness**: **`2 px`** (optimized to prevent overlap between thin `Top SC` and `Bot SC` boundaries while maintaining continuous spatial gradients across U-Net pooling layers).
- **Mask Format**: 8-bit single-channel grayscale PNG (`uint8`), containing strictly discrete integer values `{0, 1, 2, 3}` with zero anti-aliasing artifacts.

---

## 4. Dataset Output Structure

When saving annotations, the suite automatically prefixes filenames with the anatomical point identifier (`{point_tag}_{slice_stem}`) to prevent naming collisions across folders:

```text
output_directory/
├── images/
│   ├── alefootR_005_crop.png       # 512x512 RGB cropped B-Scan ROI
│   └── alehandL_008_crop.png
├── masks/
│   ├── alefootR_005_mask.png       # 512x512 8-bit grayscale mask {0, 1, 2, 3}
│   └── alehandL_008_mask.png
├── overlays/
│   ├── alefootR_005_overlay.png    # MATLAB-style visual overlay with legend
│   └── alehandL_008_overlay.png
├── strips/                         # Locked center vertical strip crops
└── strips_mask/                    # Corresponding center vertical strip masks
```

---

## 5. Installation

### Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: `3.9` or newer

### Install Dependencies
```bash
pip install -r requirements.txt
```
*(Core dependencies: `PyQt6`, `opencv-python`, `numpy`)*

---

## 6. Usage Guide

### Mode A: Laptop Server + Remote Tablet Stylus (Wi-Fi)

Ideal for annotating with a tablet stylus (e.g., Apple Pencil, Samsung S-Pen) without transferring multi-gigabyte OCT datasets to your tablet.

1. Connect both your **Laptop** and **Tablet** to the same **Wi-Fi network** or **Mobile Hotspot**.
2. On your Laptop, launch **[`run_tablet_server.bat`](run_tablet_server.bat)** (or run `python tablet_server.py`).
3. Note the local IP URL printed in the terminal (e.g., `http://192.168.1.10:8080`) and open it in **Chrome / Safari** on your Tablet.
4. On the Tablet interface:
   - Tap **"Pilih Folder Sumber"** to browse your Laptop's filesystem remotely and select your dataset folder.
   - Ensure **"🎯 Mode 1 Sampel per Titik"** and **"Mode Khusus Stylus Pen (Jari = Geser)"** are checked.
   - Use **Zoom (`100%` – `300%`)** to trace the three boundaries accurately.
   - Tap **"💾 SIMPAN KE LAPTOP"** (or **"💾 SIMPAN REVISI (GANTI FILE LAMA)"** when revising an existing label).

> **Windows Firewall Note:** On first launch, allow Python through *Windows Defender Firewall* (check both **Private** and **Public** networks) so your Tablet can connect to port `8080`.

### Mode B: Native Windows Desktop Application (PyQt6)

- Double-click **[`run_app.bat`](run_app.bat)** (or run `python label_app.py`) to open the desktop interface.

### Dataset Integrity Verification

Before training your U-Net model, validate that all exported masks have matching dimensions (`512×512`) and valid pixel classes (`{0, 1, 2, 3}`):
- Double-click **[`verify_dataset.bat`](verify_dataset.bat)** or run:
  ```bash
  python verify_dataset.py "path/to/output_directory"
  ```

---

## 7. Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `1` | Select **Class 1 (`Top SC` — Red)** |
| `2` | Select **Class 2 (`Bot SC` — Yellow)** |
| `3` | Select **Class 3 (`End ED` — Green)** |
| `0` or `Right-Click Drag` | Select **Eraser (`Background = 0`)** |
| `R` | **Random Slice Reroll** for current point (rejects current slice without saving) |
| `Ctrl + Z` | Undo last brush stroke |
| `C` | Clear all annotations on current canvas |
| `Space` | Toggle live mask overlay visibility |
| `Ctrl + S` | Save / Overwrite Revised Crop & Mask |
| `A` / `D` | Navigate to Previous / Next sample |

---

## 8. Version History (Changelog)

- **`v2.1.0` (Current)**
  - Added **Label Revision & Auto-Remove/Replace Engine** (`✏️ Mode Revisi`, atomic deletion of previous point outputs on revision save, one-click saved file deletion, and per-point slice selector dropdown).
  - Fixed random slice reroll (`🎲 Ganti Slice Acak`) so rejected slices are never auto-saved and immediately reset the point status to *Unfinished* (`0 / 1`).
  - Enforced **`1 Subject = 4 Points, 1 Sample per Point`** sampling across both single-folder and multi-folder modes.
  - Set default brush thickness to **`2 px`** to prevent overlap on thin *Stratum Corneum* layers.
- **`v2.0.0`**
  - Introduced **Spatial-to-Temporal Domain Adaptation** workflow (`512×512` B-Scan spatial ROI).
  - Added 3-class layer boundary annotation (`Top SC`, `Bot SC`, `End ED`) with real-time layer thickness calculation (`SC` and `ED` thickness in pixels) and MATLAB-style reference overlay export.
  - Added Remote Tablet Wi-Fi Server (`tablet_server.py`) with Stylus Palm Rejection.
- **`v1.0.0`**
  - Initial PyQt6 desktop ROI cropping and binary/multi-class mask annotation tool.

---

## 9. Citation & Academic Use

**Mandatory Attribution Notice:**  
If you use this software, its preprocessing workflow, or datasets generated by this tool in your research, thesis, or academic publication, **you must cite this repository**.

### BibTeX
```bibtex
@software{kusuma_oct_preprocessing_labeling_suite_2026,
  author       = {Kusuma, Raka Surya},
  title        = {{OCT Skin Layer Preprocessing and Boundary Annotation Suite (Spatial-to-Temporal Domain Adaptation)}},
  version      = {2.1.0},
  year         = {2026},
  url          = {https://github.com/rakasuryakusuma/preprocessing_labeling_apps}
}
```

### APA Style
> Kusuma, R. S. (2026). *OCT Skin Layer Preprocessing and Boundary Annotation Suite (Spatial-to-Temporal Domain Adaptation)* (Version 2.1.0) [Computer software]. GitHub. https://github.com/rakasuryakusuma/preprocessing_labeling_apps

*(You can also click the **"Cite this repository"** button on the right sidebar of the GitHub repository page, powered by [`CITATION.cff`](CITATION.cff)).*

---

## 10. License

This project is licensed under the **BSD 3-Clause License with an Academic Citation Requirement** — see the [`LICENSE`](LICENSE) file for full details. Reuse, modification, and redistribution in source or binary forms must retain the copyright notice, attribution, and citation to the original author's repository.
