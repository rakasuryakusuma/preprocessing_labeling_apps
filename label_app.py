import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


CROP_SIZE = 512

# Warna visualisasi di GUI (format RGB)
# Class 1: Permukaan (Biru Sian Terang agar kontras dengan garis putih/kuning OCT)
# Class 2: Batas Bawah (Merah Terang)
CLASS_COLORS_RGB = {
    1: (0, 220, 255),   # Class 1: Permukaan (Biru/Cyan)
    2: (255, 50, 50),   # Class 2: Batas Bawah (Merah)
}


def imread_unicode(file_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Membaca gambar dengan aman pada path Windows yang memiliki karakter non-ASCII."""
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(file_path: str, img: np.ndarray) -> bool:
    """Menyimpan gambar (.png) dengan aman pada path Windows."""
    try:
        ext = os.path.splitext(file_path)[1] or ".png"
        success, encoded = cv2.imencode(ext, img)
        if not success:
            return False
        encoded.tofile(file_path)
        return True
    except Exception:
        return False


def ensure_min_size(img: np.ndarray, min_size: int = CROP_SIZE) -> np.ndarray:
    """Jika ukuran gambar asli lebih kecil dari 512x512, lakukan zero-padding."""
    h, w = img.shape[:2]
    pad_top = max(0, (min_size - h) // 2)
    pad_bottom = max(0, min_size - h - pad_top)
    pad_left = max(0, (min_size - w) // 2)
    pad_right = max(0, min_size - w - pad_left)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
        )
    return img


def compute_crop_coords(
    img: np.ndarray,
    mode: str = "top_center",
    custom_x: int = 0,
    custom_y: int = 0,
    crop_size: int = CROP_SIZE,
) -> Tuple[int, int, int, int]:
    """
    Menghitung koordinat (start_x, start_y, end_x, end_y) untuk ROI 512x512:
    - 'top_center': X berada di tengah ((lebar // 2) - 256), Y dimulai dari atas (0).
      Cocok untuk citra OCT di mana lapisan permukaan berada di bagian atas gambar.
    - 'auto_surface': X berada di tengah, Y otomatis mendeteksi baris terang lapisan OCT.
    - 'center': X dan Y berada tepat di tengah gambar.
    - 'custom': Menggunakan start_x dan start_y pilihan pengguna.
    """
    h, w = img.shape[:2]
    half = crop_size // 2
    center_x = max(0, min(w - crop_size, (w // 2) - half))

    if mode == "top_center":
        start_x = center_x
        start_y = 0
    elif mode == "auto_surface":
        start_x = center_x
        # Cari profil kecerahan vertikal di area tengah (abaikan 5 piksel border paling atas)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        roi_strip = gray[5 : max(6, h - 5), start_x : start_x + crop_size]
        row_means = np.mean(roi_strip, axis=1)
        peak_y = int(np.argmax(row_means)) + 5
        # Tempatkan puncak permukaan sekitar 90 piksel dari batas atas crop
        start_y = max(0, min(h - crop_size, peak_y - 90))
    elif mode == "center":
        start_x = center_x
        start_y = max(0, min(h - crop_size, (h // 2) - half))
    else:  # custom
        start_x = max(0, min(w - crop_size, custom_x))
        start_y = max(0, min(h - crop_size, custom_y))

    end_x = start_x + crop_size
    end_y = start_y + crop_size
    return start_x, start_y, end_x, end_y


class RawPreviewLabel(QLabel):
    """
    Mini preview dari gambar OCT mentah yang menampilkan kotak hijau area crop 512x512.
    Pengguna juga dapat mengklik/menggeser pada preview ini untuk memindahkan kotak crop.
    """

    cropCenterClicked = pyqtSignal(int, int)  # new_start_x, new_start_y

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(180, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "QLabel { background-color: #141414; border: 1px solid #444; border-radius: 4px; }"
        )
        self.raw_bgr: Optional[np.ndarray] = None
        self.crop_coords: Tuple[int, int, int, int] = (0, 0, CROP_SIZE, CROP_SIZE)
        self._dragging = False

    def set_raw_preview(
        self, raw_bgr: np.ndarray, coords: Tuple[int, int, int, int]
    ) -> None:
        self.raw_bgr = raw_bgr
        self.crop_coords = coords
        self._render()

    def _render(self) -> None:
        if self.raw_bgr is None:
            self.clear()
            self.setText("Preview Asli")
            return

        vis = cv2.cvtColor(self.raw_bgr, cv2.COLOR_BGR2RGB)
        sx, sy, ex, ey = self.crop_coords
        # Gambar kotak hijau terang pada area yang di-crop
        cv2.rectangle(vis, (sx, sy), (ex - 1, ey - 1), (0, 255, 100), thickness=8)

        h, w, ch = vis.shape
        qimg = QImage(vis.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg).scaled(
            self.width() - 4,
            self.height() - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pm)

    def _emit_from_pos(self, pos: QPoint) -> None:
        if self.raw_bgr is None or self.pixmap() is None:
            return
        pm = self.pixmap()
        pm_w, pm_h = pm.width(), pm.height()
        off_x = (self.width() - pm_w) / 2.0
        off_y = (self.height() - pm_h) / 2.0
        rx = pos.x() - off_x
        ry = pos.y() - off_y
        if rx < 0 or ry < 0 or rx >= pm_w or ry >= pm_h:
            return
        raw_h, raw_w = self.raw_bgr.shape[:2]
        cx = int(rx * raw_w / pm_w)
        cy = int(ry * raw_h / pm_h)
        new_sx = max(0, min(raw_w - CROP_SIZE, cx - CROP_SIZE // 2))
        new_sy = max(0, min(raw_h - CROP_SIZE, cy - CROP_SIZE // 2))
        self.cropCenterClicked.emit(new_sx, new_sy)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._emit_from_pos(event.position().toPoint())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._emit_from_pos(event.position().toPoint())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False


class AnnotationCanvas(QLabel):
    """
    Komponen QLabel kustom untuk menampilkan gambar ROI 512x512
    dan menangani event mouse (press, move, release) untuk menggambar mask.
    """

    maskChanged = pyqtSignal()
    cursorMoved = pyqtSignal(int, int, int)  # x, y, nilai_mask

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(CROP_SIZE, CROP_SIZE)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QLabel { background-color: #1e1e1e; border: 2px solid #3c3c3c; border-radius: 4px; }"
        )

        # Data matriks
        self.roi_bgr: Optional[np.ndarray] = None
        self.mask: np.ndarray = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        self.undo_stack: List[np.ndarray] = []
        self.max_undo: int = 25

        # Status alat gambar
        self.active_class: int = 1  # 1: Permukaan, 2: Batas Bawah, 0: Eraser
        self.brush_size: int = 2    # Default ketebalan 2 piksel
        self.overlay_alpha: float = 0.75
        self.show_overlay: bool = True

        # Status interaksi mouse
        self.is_drawing: bool = False
        self.last_point: Optional[Tuple[int, int]] = None
        self.cursor_pos: Optional[Tuple[int, int]] = None

    def set_image_and_mask(
        self, roi_bgr: np.ndarray, existing_mask: Optional[np.ndarray] = None
    ) -> None:
        """Memuat gambar ROI 512x512 baru dan menginisialisasi matriks masker."""
        self.roi_bgr = roi_bgr.copy()
        if existing_mask is not None and existing_mask.shape == (CROP_SIZE, CROP_SIZE):
            self.mask = existing_mask.astype(np.uint8).copy()
        else:
            self.mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        self.undo_stack.clear()
        self.update_display()

    def update_roi_keep_mask(self, roi_bgr: np.ndarray) -> None:
        """Memperbarui gambar ROI tanpa mereset mask jika sedang menggeser koordinat crop."""
        self.roi_bgr = roi_bgr.copy()
        self.update_display()

    def clear_mask(self) -> None:
        """Mereset matriks masker kembali ke 0 semua."""
        if self.roi_bgr is None:
            return
        self.push_undo()
        self.mask.fill(0)
        self.update_display()
        self.maskChanged.emit()

    def push_undo(self) -> None:
        """Menyimpan state mask ke stack undo sebelum perubahan."""
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        """Mengembalikan state mask terakhir."""
        if not self.undo_stack:
            return False
        self.mask = self.undo_stack.pop()
        self.update_display()
        self.maskChanged.emit()
        return True

    def _map_widget_to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        """Mengonversi koordinat QLabel di layar ke koordinat piksel matriks 512x512."""
        if self.roi_bgr is None:
            return None
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None

        pm_w = pixmap.width()
        pm_h = pixmap.height()
        lbl_w = self.width()
        lbl_h = self.height()

        offset_x = (lbl_w - pm_w) / 2.0
        offset_y = (lbl_h - pm_h) / 2.0

        rel_x = pos.x() - offset_x
        rel_y = pos.y() - offset_y

        if rel_x < 0 or rel_y < 0 or rel_x >= pm_w or rel_y >= pm_h:
            return None

        img_x = int(rel_x * CROP_SIZE / pm_w)
        img_y = int(rel_y * CROP_SIZE / pm_h)
        img_x = max(0, min(CROP_SIZE - 1, img_x))
        img_y = max(0, min(CROP_SIZE - 1, img_y))
        return (img_x, img_y)

    def _draw_at(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> None:
        """Menggambar garis pada matriks mask menggunakan cv2.line."""
        if self.roi_bgr is None:
            return
        cv2.line(
            self.mask,
            pt1,
            pt2,
            color=int(self.active_class),
            thickness=int(self.brush_size),
            lineType=cv2.LINE_8,
        )

    def mousePressEvent(self, event) -> None:
        if self.roi_bgr is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            pt = self._map_widget_to_image(event.position().toPoint())
            if pt is not None:
                self.push_undo()
                self.is_drawing = True
                self.last_point = pt
                self._draw_at(pt, pt)
                self.update_display()
                self.maskChanged.emit()
        elif event.button() == Qt.MouseButton.RightButton:
            pt = self._map_widget_to_image(event.position().toPoint())
            if pt is not None:
                self.push_undo()
                self.is_drawing = True
                self._temp_prev_class = self.active_class
                self.active_class = 0
                self.last_point = pt
                self._draw_at(pt, pt)
                self.update_display()
                self.maskChanged.emit()

    def mouseMoveEvent(self, event) -> None:
        pt = self._map_widget_to_image(event.position().toPoint())
        self.cursor_pos = pt
        if pt is not None and self.roi_bgr is not None:
            val = int(self.mask[pt[1], pt[0]])
            self.cursorMoved.emit(pt[0], pt[1], val)

        if self.is_drawing and pt is not None and self.last_point is not None:
            self._draw_at(self.last_point, pt)
            self.last_point = pt
            self.update_display()
            self.maskChanged.emit()
        else:
            self.update_display()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.last_point = None
        elif event.button() == Qt.MouseButton.RightButton and self.is_drawing:
            self.is_drawing = False
            self.last_point = None
            if hasattr(self, "_temp_prev_class"):
                self.active_class = self._temp_prev_class

    def leaveEvent(self, event) -> None:
        self.cursor_pos = None
        self.update_display()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_display()

    def update_display(self) -> None:
        """
        Menggabungkan matriks gambar ROI asli dengan array masker untuk live overlay,
        lalu merendernya ke komponen QLabel dalam format QImage (RGB).
        """
        if self.roi_bgr is None:
            self.clear()
            self.setText("Belum ada gambar yang dimuat.\nPilih Folder Sumber di panel kiri.")
            return

        rgb_base = cv2.cvtColor(self.roi_bgr, cv2.COLOR_BGR2RGB)

        if self.show_overlay and np.any(self.mask > 0):
            overlay = rgb_base.copy()
            for cls_id, color_rgb in CLASS_COLORS_RGB.items():
                cls_indices = self.mask == cls_id
                if np.any(cls_indices):
                    overlay[cls_indices] = color_rgb

            labeled_mask = self.mask > 0
            blended = rgb_base.copy()
            blended[labeled_mask] = (
                (1.0 - self.overlay_alpha) * rgb_base[labeled_mask]
                + self.overlay_alpha * overlay[labeled_mask]
            ).astype(np.uint8)
            display_rgb = blended
        else:
            display_rgb = rgb_base

        h, w, ch = display_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            display_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qimg)

        target_size = min(self.width() - 8, self.height() - 8)
        target_size = max(CROP_SIZE, target_size)
        scaled_pixmap = pixmap.scaled(
            target_size,
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        if self.cursor_pos is not None:
            painter = QPainter(scaled_pixmap)
            scale_factor = scaled_pixmap.width() / float(CROP_SIZE)
            cx = int((self.cursor_pos[0] + 0.5) * scale_factor)
            cy = int((self.cursor_pos[1] + 0.5) * scale_factor)
            radius = max(2, int((self.brush_size * scale_factor) / 2.0))

            if self.active_class == 1:
                pen_color = QColor(0, 220, 255, 220)
            elif self.active_class == 2:
                pen_color = QColor(255, 60, 60, 220)
            else:
                pen_color = QColor(255, 255, 0, 220)

            pen = QPen(pen_color, 1, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawEllipse(QPoint(cx, cy), radius, radius)
            painter.end()

        self.setPixmap(scaled_pixmap)


class OCTLabelingApp(QMainWindow):
    """
    Aplikasi Desktop Fase 1: Preprocessing (ROI Crop 512x512) & Pixel Labeling OCT.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "OCT Preprocessing & Semantic Segmentation Labeling App (Fase 1)"
        )
        self.resize(1240, 840)

        # State direktori dan file
        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.image_files: List[Path] = []
        self.current_index: int = -1
        self.current_raw_img: Optional[np.ndarray] = None
        self.current_crop_coords: Tuple[int, int, int, int] = (0, 0, CROP_SIZE, CROP_SIZE)
        self.is_dirty: bool = False

        self._init_ui()
        self._init_shortcuts()

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # =====================================================================
        # PANEL KIRI: Kontrol Direktori, Pengaturan Crop ROI, Alat Anotasi
        # =====================================================================
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 1. Grup Manajemen Direktori
        dir_group = QGroupBox("1. Manajemen Direktori")
        dir_layout = QVBoxLayout(dir_group)

        btn_dir_row = QHBoxLayout()
        self.btn_load_input = QPushButton("Pilih Folder Sumber")
        self.btn_load_input.clicked.connect(self.select_input_folder)
        self.btn_load_output = QPushButton("Pilih Folder Output")
        self.btn_load_output.clicked.connect(self.select_output_folder)
        btn_dir_row.addWidget(self.btn_load_input)
        btn_dir_row.addWidget(self.btn_load_output)

        self.lbl_input_dir = QLabel("Sumber: (Belum dipilih)")
        self.lbl_input_dir.setWordWrap(True)
        self.lbl_input_dir.setStyleSheet("color: #888; font-size: 11px;")

        self.lbl_output_dir = QLabel("Output: (Belum dipilih)")
        self.lbl_output_dir.setWordWrap(True)
        self.lbl_output_dir.setStyleSheet("color: #888; font-size: 11px;")

        self.chk_subfolder_mode = QCheckBox(
            "Simpan juga ke subfolder /images/ && /masks/"
        )
        self.chk_subfolder_mode.setChecked(True)

        dir_layout.addLayout(btn_dir_row)
        dir_layout.addWidget(self.lbl_input_dir)
        dir_layout.addWidget(self.lbl_output_dir)
        dir_layout.addWidget(self.chk_subfolder_mode)
        left_layout.addWidget(dir_group)

        # 2. Grup Pengaturan Auto-Crop ROI (512x512) & Preview Gambar Mentah
        crop_group = QGroupBox("2. Pengaturan Crop ROI (512x512 px)")
        crop_main_layout = QHBoxLayout(crop_group)

        # Mini preview gambar asli + kotak hijau area crop
        self.raw_preview = RawPreviewLabel()
        self.raw_preview.setToolTip(
            "Kotak hijau menunjukkan area 512x512 yang di-crop.\n"
            "Klik atau geser pada preview ini untuk menggeser posisi crop secara manual."
        )
        self.raw_preview.cropCenterClicked.connect(self._on_preview_dragged)
        crop_main_layout.addWidget(self.raw_preview)

        crop_ctrl_layout = QVBoxLayout()
        crop_ctrl_layout.addWidget(QLabel("Mode Posisi Crop:"))
        self.combo_crop_mode = QComboBox()
        self.combo_crop_mode.addItem("Atas-Tengah (Top-Center: Y=0)", "top_center")
        self.combo_crop_mode.addItem("Auto-Deteksi Permukaan OCT", "auto_surface")
        self.combo_crop_mode.addItem("Tengah Penuh (Center X & Y)", "center")
        self.combo_crop_mode.addItem("Manual / Custom (Geser X & Y)", "custom")
        self.combo_crop_mode.setCurrentIndex(0)  # Default: Atas-Tengah sesuai gambar contoh OCT
        self.combo_crop_mode.currentIndexChanged.connect(self._on_crop_mode_changed)
        crop_ctrl_layout.addWidget(self.combo_crop_mode)

        form_xy = QFormLayout()
        self.spin_start_x = QSpinBox()
        self.spin_start_x.setRange(0, 4096)
        self.spin_start_x.setValue(244)
        self.spin_start_x.valueChanged.connect(self._on_custom_xy_changed)

        self.spin_start_y = QSpinBox()
        self.spin_start_y.setRange(0, 4096)
        self.spin_start_y.setValue(0)
        self.spin_start_y.valueChanged.connect(self._on_custom_xy_changed)

        form_xy.addRow("Start X:", self.spin_start_x)
        form_xy.addRow("Start Y:", self.spin_start_y)
        crop_ctrl_layout.addLayout(form_xy)

        self.lbl_raw_dim = QLabel("Asli: -")
        self.lbl_raw_dim.setStyleSheet("font-size: 11px; color: #aaa;")
        self.lbl_crop_dim = QLabel("Crop: 512x512")
        self.lbl_crop_dim.setStyleSheet("font-size: 11px; color: #2b8a3e; font-weight: bold;")
        crop_ctrl_layout.addWidget(self.lbl_raw_dim)
        crop_ctrl_layout.addWidget(self.lbl_crop_dim)
        crop_ctrl_layout.addStretch()

        crop_main_layout.addLayout(crop_ctrl_layout, stretch=1)
        left_layout.addWidget(crop_group)

        # 3. Grup Alat Anotasi (Drawing Tools)
        tools_group = QGroupBox("3. Alat Anotasi (Drawing Tools)")
        tools_layout = QVBoxLayout(tools_group)

        self.radio_class1 = QRadioButton(
            "Class 1: Permukaan Atas (Biru - Nilai 1) [Key: 1]"
        )
        self.radio_class1.setChecked(True)
        self.radio_class1.setStyleSheet("font-weight: bold; color: #00b4d8;")
        self.radio_class1.toggled.connect(self._on_class_changed)

        self.radio_class2 = QRadioButton(
            "Class 2: Batas Bawah (Merah - Nilai 2) [Key: 2]"
        )
        self.radio_class2.setStyleSheet("font-weight: bold; color: #e63946;")
        self.radio_class2.toggled.connect(self._on_class_changed)

        self.radio_eraser = QRadioButton(
            "Eraser / Hapus Piksel (Nilai 0) [Key: 0 / Klik Kanan]"
        )
        self.radio_eraser.toggled.connect(self._on_class_changed)

        tools_layout.addWidget(self.radio_class1)
        tools_layout.addWidget(self.radio_class2)
        tools_layout.addWidget(self.radio_eraser)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Ketebalan Brush (px):"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 25)
        self.spin_brush.setValue(2)  # Default ketebalan 2 piksel
        self.spin_brush.valueChanged.connect(self._on_brush_size_changed)
        brush_row.addWidget(self.spin_brush)
        tools_layout.addLayout(brush_row)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Opasitas Overlay:"))
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(20, 100)
        self.slider_alpha.setValue(75)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        alpha_row.addWidget(self.slider_alpha)
        tools_layout.addLayout(alpha_row)

        self.chk_show_overlay = QCheckBox("Tampilkan Live Overlay [Key: Space]")
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.toggled.connect(self._on_toggle_overlay)
        tools_layout.addWidget(self.chk_show_overlay)

        edit_btn_row = QHBoxLayout()
        self.btn_undo = QPushButton("Undo (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo_annotation)
        self.btn_clear = QPushButton("Clear Label (C)")
        self.btn_clear.setStyleSheet("color: #d9534f; font-weight: bold;")
        self.btn_clear.clicked.connect(self.clear_annotation)
        edit_btn_row.addWidget(self.btn_undo)
        edit_btn_row.addWidget(self.btn_clear)
        tools_layout.addLayout(edit_btn_row)

        self.lbl_mask_stats = QLabel("Class 1: 0 px | Class 2: 0 px")
        self.lbl_mask_stats.setStyleSheet("font-size: 11px;")
        tools_layout.addWidget(self.lbl_mask_stats)

        left_layout.addWidget(tools_group)

        # 4. Daftar File Gambar
        list_group = QGroupBox("4. Daftar Gambar (Progres)")
        list_layout = QVBoxLayout(list_group)
        self.lbl_progress = QLabel("0 / 0 gambar")
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentRowChanged.connect(self._on_list_row_changed)
        list_layout.addWidget(self.lbl_progress)
        list_layout.addWidget(self.file_list_widget)
        left_layout.addWidget(list_group, stretch=1)

        # 5. Grup Navigasi & Eksekusi
        nav_group = QGroupBox("5. Navigasi && Eksekusi")
        nav_layout = QVBoxLayout(nav_group)

        nav_btn_row = QHBoxLayout()
        self.btn_prev = QPushButton("<< Previous (A)")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("Next (D) >>")
        self.btn_next.clicked.connect(self.next_image)
        nav_btn_row.addWidget(self.btn_prev)
        nav_btn_row.addWidget(self.btn_next)
        nav_layout.addLayout(nav_btn_row)

        self.chk_autosave = QCheckBox("Auto-Save saat pindah gambar (Next/Prev)")
        self.chk_autosave.setChecked(False)
        nav_layout.addWidget(self.chk_autosave)

        self.btn_save = QPushButton("SIMPAN CEPAT: Crop + Mask (Ctrl+S)")
        self.btn_save.setMinimumHeight(38)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #2b8a3e; color: white; font-weight: bold; font-size: 13px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2f9e44; }"
        )
        self.btn_save.clicked.connect(self.save_current)
        nav_layout.addWidget(self.btn_save)

        left_layout.addWidget(nav_group)

        # =====================================================================
        # PANEL KANAN: Area Kanvas Gambar (QLabel Kustom)
        # =====================================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = AnnotationCanvas()
        self.canvas.maskChanged.connect(self._on_mask_modified)
        self.canvas.cursorMoved.connect(self._on_cursor_moved)
        right_layout.addWidget(self.canvas, stretch=1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Siap. Pilih Folder Sumber OCT dan Folder Output untuk memulai."
        )

    def _init_shortcuts(self) -> None:
        act_save = QAction(self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self.save_current)
        self.addAction(act_save)

        act_undo = QAction(self)
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.triggered.connect(self.undo_annotation)
        self.addAction(act_undo)

        act_prev = QAction(self)
        act_prev.setShortcut(QKeySequence("A"))
        act_prev.triggered.connect(self.prev_image)
        self.addAction(act_prev)

        act_next = QAction(self)
        act_next.setShortcut(QKeySequence("D"))
        act_next.triggered.connect(self.next_image)
        self.addAction(act_next)

        act_cls1 = QAction(self)
        act_cls1.setShortcut(QKeySequence("1"))
        act_cls1.triggered.connect(lambda: self.radio_class1.setChecked(True))
        self.addAction(act_cls1)

        act_cls2 = QAction(self)
        act_cls2.setShortcut(QKeySequence("2"))
        act_cls2.triggered.connect(lambda: self.radio_class2.setChecked(True))
        self.addAction(act_cls2)

        act_cls0 = QAction(self)
        act_cls0.setShortcut(QKeySequence("0"))
        act_cls0.triggered.connect(lambda: self.radio_eraser.setChecked(True))
        self.addAction(act_cls0)

        act_clear = QAction(self)
        act_clear.setShortcut(QKeySequence("C"))
        act_clear.triggered.connect(self.clear_annotation)
        self.addAction(act_clear)

        act_space = QAction(self)
        act_space.setShortcut(QKeySequence("Space"))
        act_space.triggered.connect(
            lambda: self.chk_show_overlay.setChecked(not self.chk_show_overlay.isChecked())
        )
        self.addAction(act_space)

    # =========================================================================
    # Manajemen Direktori & Pemuatan Gambar
    # =========================================================================
    def select_input_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Pilih Folder Sumber Gambar OCT Mentah"
        )
        if not folder:
            return

        self.input_dir = Path(folder)
        self.lbl_input_dir.setText(f"Sumber: {self.input_dir}")

        if self.output_dir is None:
            default_out = self.input_dir.parent / f"{self.input_dir.name}_output"
            self.output_dir = default_out
            self.lbl_output_dir.setText(f"Output: {self.output_dir}")

        self._scan_input_images()

    def select_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Pilih Folder Output Penyimpanan Crop & Mask"
        )
        if not folder:
            return

        self.output_dir = Path(folder)
        self.lbl_output_dir.setText(f"Output: {self.output_dir}")
        self._refresh_file_list_status()
        if 0 <= self.current_index < len(self.image_files):
            self.load_image_at_index(self.current_index)

    def _scan_input_images(self) -> None:
        if self.input_dir is None or not self.input_dir.exists():
            return

        files = [
            p
            for p in sorted(self.input_dir.iterdir())
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        self.image_files = files

        self.file_list_widget.blockSignals(True)
        self.file_list_widget.clear()
        for p in self.image_files:
            item = QListWidgetItem(p.name)
            self.file_list_widget.addItem(item)
        self.file_list_widget.blockSignals(False)

        self._refresh_file_list_status()

        if self.image_files:
            self.file_list_widget.setCurrentRow(0)
            self.load_image_at_index(0)
            self.status_bar.showMessage(
                f"Berhasil memuat {len(self.image_files)} gambar dari {self.input_dir}"
            )
        else:
            self.current_index = -1
            self.current_raw_img = None
            self.canvas.roi_bgr = None
            self.canvas.update_display()
            QMessageBox.warning(
                self,
                "Folder Kosong",
                "Tidak ditemukan file gambar (.jpg, .png, dll.) pada folder sumber tersebut.",
            )

    def _get_output_paths(self, src_path: Path) -> Tuple[Path, Path]:
        assert self.output_dir is not None
        stem = src_path.stem
        crop_path = self.output_dir / f"{stem}_crop.png"
        mask_path = self.output_dir / f"{stem}_mask.png"
        return crop_path, mask_path

    def _is_file_labeled(self, src_path: Path) -> bool:
        if self.output_dir is None:
            return False
        _, mask_path = self._get_output_paths(src_path)
        sub_mask_path = self.output_dir / "masks" / f"{src_path.stem}_mask.png"
        return mask_path.exists() or sub_mask_path.exists()

    def _refresh_file_list_status(self) -> None:
        labeled_count = 0
        for idx, p in enumerate(self.image_files):
            item = self.file_list_widget.item(idx)
            if item is None:
                continue
            if self._is_file_labeled(p):
                labeled_count += 1
                item.setText(f"[✓] {p.name}")
                item.setForeground(QColor("#2b8a3e"))
            else:
                item.setText(f"[ ] {p.name}")
                item.setForeground(QColor("#dddddd"))

        total = len(self.image_files)
        self.lbl_progress.setText(f"Selesai dilabeli: {labeled_count} / {total} gambar")

    def load_image_at_index(self, index: int) -> None:
        if index < 0 or index >= len(self.image_files):
            return

        src_path = self.image_files[index]
        raw_img = imread_unicode(str(src_path), cv2.IMREAD_COLOR)
        if raw_img is None:
            QMessageBox.critical(
                self, "Gagal Membaca Gambar", f"Tidak dapat membuka file:\n{src_path}"
            )
            return

        raw_img = ensure_min_size(raw_img, CROP_SIZE)
        self.current_raw_img = raw_img
        raw_h, raw_w = raw_img.shape[:2]

        # Atur batas maksimum spinbox X dan Y sesuai dimensi gambar asli
        self.spin_start_x.blockSignals(True)
        self.spin_start_y.blockSignals(True)
        self.spin_start_x.setMaximum(max(0, raw_w - CROP_SIZE))
        self.spin_start_y.setMaximum(max(0, raw_h - CROP_SIZE))
        self.spin_start_x.blockSignals(False)
        self.spin_start_y.blockSignals(False)

        # Hitung koordinat crop sesuai mode yang aktif
        mode = self.combo_crop_mode.currentData()
        sx, sy, ex, ey = compute_crop_coords(
            raw_img,
            mode=mode,
            custom_x=self.spin_start_x.value(),
            custom_y=self.spin_start_y.value(),
            crop_size=CROP_SIZE,
        )
        self.current_crop_coords = (sx, sy, ex, ey)

        # Sinkronkan nilai spinbox
        self.spin_start_x.blockSignals(True)
        self.spin_start_y.blockSignals(True)
        self.spin_start_x.setValue(sx)
        self.spin_start_y.setValue(sy)
        self.spin_start_x.blockSignals(False)
        self.spin_start_y.blockSignals(False)

        cropped_bgr = raw_img[sy:ey, sx:ex].copy()
        self.raw_preview.set_raw_preview(raw_img, self.current_crop_coords)

        # Cek apakah sudah ada mask tersimpan untuk gambar ini
        existing_mask = None
        if self.output_dir is not None:
            _, mask_path = self._get_output_paths(src_path)
            sub_mask_path = self.output_dir / "masks" / f"{src_path.stem}_mask.png"
            candidate = mask_path if mask_path.exists() else sub_mask_path
            if candidate.exists():
                loaded_mask = imread_unicode(str(candidate), cv2.IMREAD_GRAYSCALE)
                if loaded_mask is not None and loaded_mask.shape == (CROP_SIZE, CROP_SIZE):
                    existing_mask = loaded_mask

        self.current_index = index
        self.canvas.set_image_and_mask(cropped_bgr, existing_mask)
        self.is_dirty = False

        self.lbl_raw_dim.setText(f"Asli: {raw_w} x {raw_h} px ({src_path.name})")
        self.lbl_crop_dim.setText(f"ROI: X[{sx}:{ex}], Y[{sy}:{ey}] (512x512)")
        self._update_mask_stats()

    def _apply_current_crop(self) -> None:
        """Memotong ulang gambar mentah saat ini sesuai koordinat terbaru."""
        if self.current_raw_img is None:
            return
        mode = self.combo_crop_mode.currentData()
        sx, sy, ex, ey = compute_crop_coords(
            self.current_raw_img,
            mode=mode,
            custom_x=self.spin_start_x.value(),
            custom_y=self.spin_start_y.value(),
            crop_size=CROP_SIZE,
        )
        self.current_crop_coords = (sx, sy, ex, ey)

        self.spin_start_x.blockSignals(True)
        self.spin_start_y.blockSignals(True)
        self.spin_start_x.setValue(sx)
        self.spin_start_y.setValue(sy)
        self.spin_start_x.blockSignals(False)
        self.spin_start_y.blockSignals(False)

        cropped_bgr = self.current_raw_img[sy:ey, sx:ex].copy()
        self.raw_preview.set_raw_preview(self.current_raw_img, self.current_crop_coords)
        self.canvas.update_roi_keep_mask(cropped_bgr)
        self.lbl_crop_dim.setText(f"ROI: X[{sx}:{ex}], Y[{sy}:{ey}] (512x512)")

    def _on_crop_mode_changed(self) -> None:
        self._apply_current_crop()

    def _on_custom_xy_changed(self) -> None:
        # Jika pengguna mengubah angka SpinBox X/Y secara langsung, ubah ke mode Custom
        if self.combo_crop_mode.currentData() != "custom":
            self.combo_crop_mode.blockSignals(True)
            idx = self.combo_crop_mode.findData("custom")
            if idx >= 0:
                self.combo_crop_mode.setCurrentIndex(idx)
            self.combo_crop_mode.blockSignals(False)
        self._apply_current_crop()

    def _on_preview_dragged(self, new_sx: int, new_sy: int) -> None:
        self.spin_start_x.blockSignals(True)
        self.spin_start_y.blockSignals(True)
        self.spin_start_x.setValue(new_sx)
        self.spin_start_y.setValue(new_sy)
        self.spin_start_x.blockSignals(False)
        self.spin_start_y.blockSignals(False)
        self._on_custom_xy_changed()

    # =========================================================================
    # Handler Event Alat Anotasi
    # =========================================================================
    def _on_class_changed(self) -> None:
        if self.radio_class1.isChecked():
            self.canvas.active_class = 1
        elif self.radio_class2.isChecked():
            self.canvas.active_class = 2
        else:
            self.canvas.active_class = 0
        self.canvas.update_display()

    def _on_brush_size_changed(self, value: int) -> None:
        self.canvas.brush_size = value
        self.canvas.update_display()

    def _on_alpha_changed(self, value: int) -> None:
        self.canvas.overlay_alpha = value / 100.0
        self.canvas.update_display()

    def _on_toggle_overlay(self, checked: bool) -> None:
        self.canvas.show_overlay = checked
        self.canvas.update_display()

    def _on_mask_modified(self) -> None:
        self.is_dirty = True
        self._update_mask_stats()

    def _update_mask_stats(self) -> None:
        c1 = int(np.sum(self.canvas.mask == 1))
        c2 = int(np.sum(self.canvas.mask == 2))
        self.lbl_mask_stats.setText(f"Class 1: {c1} px | Class 2: {c2} px")

    def _on_cursor_moved(self, x: int, y: int, mask_val: int) -> None:
        cls_names = {0: "Background (0)", 1: "Permukaan (1)", 2: "Batas Bawah (2)"}
        cls_label = cls_names.get(mask_val, str(mask_val))
        self.status_bar.showMessage(
            f"Koordinat ROI: X={x}, Y={y} | Nilai Piksel Mask: {cls_label}"
        )

    def undo_annotation(self) -> None:
        if self.canvas.undo():
            self.status_bar.showMessage("Undo berhasil.")
        else:
            self.status_bar.showMessage("Tidak ada riwayat gambar untuk di-undo.")

    def clear_annotation(self) -> None:
        if self.canvas.roi_bgr is None:
            return
        self.canvas.clear_mask()
        self.status_bar.showMessage("Label direset (Clear Label).")

    # =========================================================================
    # Penyimpanan (Export) & Navigasi
    # =========================================================================
    def save_current(self) -> bool:
        """
        Menyimpan gambar hasil crop ([namafile]_crop.png) dan matriks masker
        ([namafile]_mask.png) berisi nilai piksel murni 0, 1, atau 2.
        """
        if self.canvas.roi_bgr is None or self.current_index < 0:
            QMessageBox.warning(self, "Peringatan", "Belum ada gambar yang dimuat.")
            return False

        if self.output_dir is None:
            self.select_output_folder()
            if self.output_dir is None:
                return False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        src_path = self.image_files[self.current_index]
        crop_path, mask_path = self._get_output_paths(src_path)

        clean_mask = np.clip(self.canvas.mask, 0, 2).astype(np.uint8)

        ok_crop = imwrite_unicode(str(crop_path), self.canvas.roi_bgr)
        ok_mask = imwrite_unicode(str(mask_path), clean_mask)

        if self.chk_subfolder_mode.isChecked():
            img_dir = self.output_dir / "images"
            msk_dir = self.output_dir / "masks"
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(str(img_dir / f"{src_path.stem}_crop.png"), self.canvas.roi_bgr)
            imwrite_unicode(str(msk_dir / f"{src_path.stem}_mask.png"), clean_mask)

        if ok_crop and ok_mask:
            self.is_dirty = False
            self._refresh_file_list_status()
            unique_vals = np.unique(clean_mask).tolist()
            self.status_bar.showMessage(
                f"Tersimpan: {crop_path.name} & {mask_path.name} (Nilai unik mask: {unique_vals})"
            )
            return True
        else:
            QMessageBox.critical(
                self,
                "Gagal Menyimpan",
                f"Terjadi kesalahan saat menyimpan ke:\n{self.output_dir}",
            )
            return False

    def _on_list_row_changed(self, row: int) -> None:
        if row < 0 or row == self.current_index:
            return
        if self.chk_autosave.isChecked() and self.is_dirty:
            self.save_current()
        self.load_image_at_index(row)

    def prev_image(self) -> None:
        if not self.image_files or self.current_index <= 0:
            return
        if self.chk_autosave.isChecked() and self.is_dirty:
            self.save_current()
        new_idx = self.current_index - 1
        self.file_list_widget.setCurrentRow(new_idx)

    def next_image(self) -> None:
        if not self.image_files or self.current_index >= len(self.image_files) - 1:
            return
        if self.chk_autosave.isChecked() and self.is_dirty:
            self.save_current()
        new_idx = self.current_index + 1
        self.file_list_widget.setCurrentRow(new_idx)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = OCTLabelingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
