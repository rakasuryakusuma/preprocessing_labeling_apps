import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
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
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


CROP_SIZE = 512

# Warna visualisasi di GUI (format RGB) sesuai gambar referensi:
# Class 1: top SC - Batas Paling Atas / Permukaan SC (Merah)
# Class 2: bot SC - Batas Tengah / Bawah SC (Kuning)
# Class 3: end ED - Batas Bawah / Akhir Epidermis (Hijau)
CLASS_COLORS_RGB = {
    1: (255, 45, 45),   # Class 1: top SC (Merah)
    2: (255, 230, 0),   # Class 2: bot SC (Kuning)
    3: (0, 230, 64),    # Class 3: end ED (Hijau)
}
MAX_CLASS_ID = 3


APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #121418;
    color: #e4e7eb;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12px;
}

QGroupBox {
    background-color: #1a1d24;
    border: 1px solid #2c313c;
    border-radius: 8px;
    margin-top: 10px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    font-size: 12px;
    color: #f0f4f8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #7cc4fa;
    background-color: #1a1d24;
}

QLabel {
    background-color: transparent;
    color: #d8dde6;
}

QPushButton {
    background-color: #262b36;
    color: #f0f4f8;
    border: 1px solid #3a4152;
    border-radius: 5px;
    padding: 5px 10px;
    min-height: 22px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #323947;
    border-color: #4f596f;
}

QPushButton:pressed {
    background-color: #1e222b;
}

QComboBox, QSpinBox {
    background-color: #222630;
    color: #f0f4f8;
    border: 1px solid #3a4152;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 22px;
}

QComboBox:hover, QSpinBox:hover {
    border-color: #52607a;
}

QComboBox::drop-down {
    border: none;
    width: 22px;
}

QComboBox QAbstractItemView {
    background-color: #222630;
    color: #f0f4f8;
    selection-background-color: #2b6cb0;
    selection-color: #ffffff;
    border: 1px solid #3a4152;
}

QRadioButton, QCheckBox {
    background-color: transparent;
    spacing: 8px;
    padding: 3px 2px;
    min-height: 20px;
    color: #e4e7eb;
}

QRadioButton::indicator, QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QSlider::groove:horizontal {
    border: 1px solid #323846;
    height: 6px;
    background: #222630;
    margin: 2px 0;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #e4e7eb;
    border: 1px solid #3b82f6;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
}

QListWidget {
    background-color: #14171d;
    border: 1px solid #2c313c;
    border-radius: 6px;
    padding: 4px;
    outline: none;
}

QListWidget::item {
    padding: 6px 8px;
    border-radius: 4px;
    margin-bottom: 2px;
}

QListWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
    font-weight: bold;
}

QListWidget::item:hover:!selected {
    background-color: #222733;
}

QProgressBar {
    background-color: #14171d;
    border: 1px solid #2c313c;
    border-radius: 5px;
    text-align: center;
    color: #e4e7eb;
    font-size: 11px;
    font-weight: bold;
    min-height: 18px;
    max-height: 18px;
}

QProgressBar::chunk {
    background-color: #2b8a3e;
    border-radius: 4px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background: #14171d;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #363d4d;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4b556b;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QStatusBar {
    background-color: #161920;
    color: #9aa5b1;
    border-top: 1px solid #282d37;
    font-size: 12px;
}
"""


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


def compute_layer_thickness(mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Menghitung rata-rata jarak vertikal (ketebalan dalam piksel) antar garis batas:
    - Tebal SC (Class 1 'top SC' ke Class 2 'bot SC')
    - Tebal ED (Class 2 'bot SC' ke Class 3 'end ED')
    """
    h, w = mask.shape[:2]
    y_grid = np.arange(h, dtype=np.float32)[:, None]

    def mean_y_per_col(cls_id: int) -> Tuple[np.ndarray, np.ndarray]:
        m = (mask == cls_id).astype(np.float32)
        counts = m.sum(axis=0)
        valid = counts > 0
        y_mean = np.zeros(w, dtype=np.float32)
        y_mean[valid] = (m[:, valid] * y_grid).sum(axis=0) / counts[valid]
        return y_mean, valid

    y1, v1 = mean_y_per_col(1)
    y2, v2 = mean_y_per_col(2)
    y3, v3 = mean_y_per_col(3)

    v_sc = v1 & v2
    sc_thick = float(np.mean(np.abs(y2[v_sc] - y1[v_sc]))) if np.any(v_sc) else None

    v_ed = v2 & v3
    ed_thick = float(np.mean(np.abs(y3[v_ed] - y2[v_ed]))) if np.any(v_ed) else None

    return sc_thick, ed_thick


def render_reference_overlay(roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Membuat gambar visualisasi hasil akhir seperti gambar referensi:
    - Garis Merah: top SC (Class 1)
    - Garis Kuning: bot SC (Class 2)
    - Garis Hijau: end ED (Class 3)
    Dilengkapi kotak legenda di kanan-bawah/tengah.
    """
    vis = roi_bgr.copy()
    # BGR colors
    vis[mask == 1] = (45, 45, 255)    # Merah (top SC)
    vis[mask == 2] = (0, 230, 255)    # Kuning (bot SC)
    vis[mask == 3] = (64, 230, 0)     # Hijau (end ED)

    # Gambar kotak legenda di pojok kanan bawah agar tidak menutupi lapisan atas
    box_w, box_h = 132, 74
    bx1 = CROP_SIZE - box_w - 10
    by1 = CROP_SIZE - box_h - 10
    bx2 = CROP_SIZE - 10
    by2 = CROP_SIZE - 10

    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (15, 15, 15), thickness=-1)
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (90, 90, 90), thickness=1)

    items = [
        ("top SC", (45, 45, 255)),
        ("bot SC", (0, 230, 255)),
        ("end ED", (64, 230, 0)),
    ]
    for idx, (label, bgr) in enumerate(items):
        ly = by1 + 18 + idx * 20
        cv2.line(vis, (bx1 + 10, ly), (bx1 + 42, ly), bgr, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            label,
            (bx1 + 50, ly + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return vis


class RawPreviewLabel(QLabel):
    """
    Mini preview dari gambar OCT mentah yang menampilkan kotak hijau area crop 512x512.
    Pengguna juga dapat mengklik/menggeser pada preview ini untuk memindahkan kotak crop.
    """

    cropCenterClicked = pyqtSignal(int, int)  # new_start_x, new_start_y

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(136, 136)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setStyleSheet(
            "QLabel { background-color: #101216; border: 1px solid #363d4d; border-radius: 6px; color: #6c7586; font-size: 11px; }"
        )
        self.setText("Preview Asli\n(Klik/Geser ROI)")
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
            self.setText("Preview Asli\n(Klik/Geser ROI)")
            return

        vis = cv2.cvtColor(self.raw_bgr, cv2.COLOR_BGR2RGB)
        sx, sy, ex, ey = self.crop_coords
        # Gambar kotak hijau terang pada area yang di-crop
        cv2.rectangle(vis, (sx, sy), (ex - 1, ey - 1), (0, 255, 100), thickness=10)

        h, w, ch = vis.shape
        qimg = QImage(vis.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg).scaled(
            self.width() - 6,
            self.height() - 6,
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
        self.setMinimumSize(420, 420)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QLabel { background-color: #161920; border: 2px solid #2c313c; border-radius: 8px; color: #8a94a6; font-size: 13px; }"
        )

        # Data matriks
        self.roi_bgr: Optional[np.ndarray] = None
        self.mask: np.ndarray = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        self.undo_stack: List[np.ndarray] = []
        self.max_undo: int = 25

        # Status alat gambar (Default Brush 2px)
        self.active_class: int = 1  # 1: top SC, 2: bot SC, 3: end ED, 0: Eraser
        self.brush_size: int = 2    # Default ketebalan 2 piksel
        self.overlay_alpha: float = 0.75
        self.show_overlay: bool = True
        self.show_slice_guide: bool = False
        self.strip_width: int = 8
        self.frame_segments: List[dict] = []

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
            self.setText(
                "Belum ada gambar yang dimuat.\n\n"
                "Klik tombol 'Pilih Folder Sumber' di panel kiri\n"
                "untuk memuat citra OCT mentah."
            )
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

        target_size = min(self.width() - 12, self.height() - 12)
        target_size = max(256, target_size)
        scaled_pixmap = pixmap.scaled(
            target_size,
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        scale_factor = scaled_pixmap.width() / float(CROP_SIZE)
        if self.show_slice_guide:
            painter = QPainter(scaled_pixmap)
            if len(self.frame_segments) > 1:
                guide_pen = QPen(QColor(124, 196, 250, 120), 1, Qt.PenStyle.DashLine)
                painter.setPen(guide_pen)
                for idx, seg in enumerate(self.frame_segments):
                    sx0 = int(seg["x0"] * scale_factor)
                    if idx > 0:
                        painter.drawLine(sx0, 0, sx0, scaled_pixmap.height() - 1)
                    painter.drawText(sx0 + 4, 14, f"F{idx + 1}")
            else:
                sw = max(1, min(64, int(self.strip_width)))
                x0 = int(max(0, 256 - sw // 2) * scale_factor)
                w_scaled = max(2, int(sw * scale_factor))
                guide_pen = QPen(QColor(255, 50, 50, 210), 1, Qt.PenStyle.DashLine)
                painter.setPen(guide_pen)
                painter.drawRect(x0, 0, w_scaled, scaled_pixmap.height() - 1)
            painter.end()

        if self.cursor_pos is not None:
            painter = QPainter(scaled_pixmap)
            cx = int((self.cursor_pos[0] + 0.5) * scale_factor)
            cy = int((self.cursor_pos[1] + 0.5) * scale_factor)
            radius = max(2, int((self.brush_size * scale_factor) / 2.0))

            if self.active_class == 1:
                pen_color = QColor(255, 60, 60, 220)   # Merah (top SC)
            elif self.active_class == 2:
                pen_color = QColor(255, 230, 0, 220)   # Kuning (bot SC)
            elif self.active_class == 3:
                pen_color = QColor(0, 230, 64, 220)    # Hijau (end ED)
            else:
                pen_color = QColor(56, 189, 248, 220)  # Eraser (Biru Muda)

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
    RESERVED_DIR_NAMES = {
        "images",
        "masks",
        "overlays",
        "strips",
        "strips_mask",
        "stitched",
        "debug_previews",
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "OCT Labeling App v2 — Spatial-to-Temporal Domain Adaptation (B-Scan 512×512)"
        )
        self.resize(1280, 760)
        self.setMinimumSize(980, 600)

        # State direktori dan file (1 Subjek = 4 Titik, 1 Titik = 1 Sampel)
        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.image_files: List[Path] = []
        self.point_groups: dict[str, List[Path]] = {}
        self.selected_point_sample: dict[str, Path] = {}
        self.rejected_point_samples: dict[str, set] = {}
        self.sibling_points: List[Path] = []
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
        main_layout.setSpacing(10)

        # =====================================================================
        # KOLOM KIRI (Scrollable): Direktori, Crop ROI, Alat Anotasi
        # =====================================================================
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(360)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(10)

        # 1. Grup Manajemen Direktori
        dir_group = QGroupBox("1. Manajemen Direktori")
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setSpacing(8)

        btn_dir_row = QHBoxLayout()
        btn_dir_row.setSpacing(8)
        self.btn_load_input = QPushButton("Pilih Folder Sumber")
        self.btn_load_input.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_input.setStyleSheet(
            "QPushButton { background-color: #1d4ed8; border-color: #2563eb; color: white; }"
            "QPushButton:hover { background-color: #2563eb; }"
        )
        self.btn_load_input.clicked.connect(self.select_input_folder)

        self.btn_load_output = QPushButton("Pilih Folder Output")
        self.btn_load_output.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_output.clicked.connect(self.select_output_folder)
        btn_dir_row.addWidget(self.btn_load_input)
        btn_dir_row.addWidget(self.btn_load_output)

        path_box = QFrame()
        path_box.setStyleSheet(
            "QFrame { background-color: #14171d; border: 1px solid #262b36; border-radius: 5px; padding: 4px; }"
        )
        path_box_layout = QVBoxLayout(path_box)
        path_box_layout.setContentsMargins(6, 4, 6, 4)
        path_box_layout.setSpacing(3)

        self.lbl_input_dir = QLabel("Sumber: (Belum dipilih)")
        self.lbl_input_dir.setWordWrap(True)
        self.lbl_input_dir.setStyleSheet("color: #9aa5b1; font-size: 11px; border: none;")

        self.lbl_output_dir = QLabel("Output: (Belum dipilih)")
        self.lbl_output_dir.setWordWrap(True)
        self.lbl_output_dir.setStyleSheet("color: #9aa5b1; font-size: 11px; border: none;")

        path_box_layout.addWidget(self.lbl_input_dir)
        path_box_layout.addWidget(self.lbl_output_dir)

        self.chk_subfolder_mode = QCheckBox(
            "Simpan juga ke subfolder /images/ && /masks/"
        )
        self.chk_subfolder_mode.setChecked(True)

        self.chk_one_sample_per_point = QCheckBox(
            "🎯 Mode 1 Sampel per Titik (4 Titik/Subjek)"
        )
        self.chk_one_sample_per_point.setChecked(True)
        self.chk_one_sample_per_point.setStyleSheet("color: #4ade80; font-weight: bold;")
        self.chk_one_sample_per_point.toggled.connect(self._scan_input_images)

        self.btn_load_parent_points = QPushButton(
            "📂 Gabung Semua Titik (1 Sampel/Titik)"
        )
        self.btn_load_parent_points.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_parent_points.setStyleSheet(
            "QPushButton { background-color: #0284c7; border-color: #38bdf8; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.btn_load_parent_points.clicked.connect(self.load_parent_all_points)
        self.btn_load_parent_points.setVisible(False)

        dir_layout.addLayout(btn_dir_row)
        dir_layout.addWidget(path_box)
        dir_layout.addWidget(self.chk_subfolder_mode)
        dir_layout.addWidget(self.chk_one_sample_per_point)
        dir_layout.addWidget(self.btn_load_parent_points)
        left_layout.addWidget(dir_group)

        # 2. Grup Pengaturan Auto-Crop ROI (512x512) & Preview Gambar Mentah
        crop_group = QGroupBox("2. Pengaturan Crop ROI (512×512 px)")
        crop_group_layout = QVBoxLayout(crop_group)
        crop_group_layout.setSpacing(8)

        mode_row = QVBoxLayout()
        mode_row.setSpacing(4)
        lbl_mode = QLabel("Mode Posisi Crop:")
        lbl_mode.setStyleSheet("color: #b0b8c4; font-size: 11px;")
        self.combo_crop_mode = QComboBox()
        self.combo_crop_mode.addItem("Atas-Tengah (Top-Center: Y=0)", "top_center")
        self.combo_crop_mode.addItem("Auto-Deteksi Permukaan OCT", "auto_surface")
        self.combo_crop_mode.addItem("Tengah Penuh (Center X & Y)", "center")
        self.combo_crop_mode.addItem("Manual / Custom (Geser X & Y)", "custom")
        self.combo_crop_mode.setCurrentIndex(0)
        self.combo_crop_mode.currentIndexChanged.connect(self._on_crop_mode_changed)
        mode_row.addWidget(lbl_mode)
        mode_row.addWidget(self.combo_crop_mode)
        crop_group_layout.addLayout(mode_row)

        crop_body_layout = QHBoxLayout()
        crop_body_layout.setSpacing(10)

        # Mini preview gambar asli + kotak hijau area crop
        self.raw_preview = RawPreviewLabel()
        self.raw_preview.setToolTip(
            "Kotak hijau menunjukkan area 512x512 yang di-crop.\n"
            "Klik atau geser pada preview ini untuk menggeser posisi crop secara manual."
        )
        self.raw_preview.cropCenterClicked.connect(self._on_preview_dragged)
        crop_body_layout.addWidget(self.raw_preview)

        crop_ctrl_layout = QVBoxLayout()
        crop_ctrl_layout.setSpacing(6)

        form_xy = QFormLayout()
        form_xy.setContentsMargins(0, 0, 0, 0)
        form_xy.setSpacing(6)
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
        self.lbl_raw_dim.setWordWrap(True)
        self.lbl_raw_dim.setStyleSheet("font-size: 11px; color: #9aa5b1;")
        self.lbl_crop_dim = QLabel("ROI: 512×512 px")
        self.lbl_crop_dim.setWordWrap(True)
        self.lbl_crop_dim.setStyleSheet(
            "font-size: 11px; color: #4ade80; font-weight: bold;"
        )
        crop_ctrl_layout.addWidget(self.lbl_raw_dim)
        crop_ctrl_layout.addWidget(self.lbl_crop_dim)
        crop_ctrl_layout.addStretch()

        crop_body_layout.addLayout(crop_ctrl_layout, stretch=1)
        crop_group_layout.addLayout(crop_body_layout)

        self.chk_lock_crop = QCheckBox(
            "🔒 Kunci Lokasi Tengah ROI (Konsisten Antar-Slice)"
        )
        self.chk_lock_crop.setChecked(True)
        self.chk_lock_crop.setStyleSheet("color: #4ade80; font-weight: bold;")
        self.chk_lock_crop.toggled.connect(self._on_lock_crop_toggled)
        self.spin_start_x.setEnabled(False)
        self.spin_start_y.setEnabled(False)
        crop_group_layout.addWidget(self.chk_lock_crop)

        strip_row = QHBoxLayout()
        self.chk_slice_guide = QCheckBox("Garis Panduan Tengah (Opsional)")
        self.chk_slice_guide.setChecked(False)
        self.chk_slice_guide.toggled.connect(self._on_slice_guide_toggled)
        self.spin_strip_w = QSpinBox()
        self.spin_strip_w.setRange(1, 64)
        self.spin_strip_w.setValue(8)
        self.spin_strip_w.setSuffix(" px")
        self.spin_strip_w.setFixedWidth(68)
        self.spin_strip_w.valueChanged.connect(self._on_strip_width_changed)
        strip_row.addWidget(self.chk_slice_guide)
        strip_row.addStretch()
        strip_row.addWidget(QLabel("Lebar:"))
        strip_row.addWidget(self.spin_strip_w)
        crop_group_layout.addLayout(strip_row)

        left_layout.addWidget(crop_group)

        # 3. Grup Alat Anotasi (Drawing Tools - Aplikasi v2)
        tools_group = QGroupBox("3. Garis Batas Lapisan (3 Class — Brush 3px)")
        tools_layout = QVBoxLayout(tools_group)
        tools_layout.setSpacing(8)

        self.radio_class1 = QRadioButton(
            "Class 1: Top SC - Permukaan Kulit (Merah - Nilai 1) [1]"
        )
        self.radio_class1.setChecked(True)
        self.radio_class1.setStyleSheet(
            "QRadioButton { font-weight: bold; color: #ff4d4d; }"
        )
        self.radio_class1.toggled.connect(self._on_class_changed)

        self.radio_class2 = QRadioButton(
            "Class 2: Bot SC - Bawah Stratum Corneum (Kuning - Nilai 2) [2]"
        )
        self.radio_class2.setStyleSheet(
            "QRadioButton { font-weight: bold; color: #fde047; }"
        )
        self.radio_class2.toggled.connect(self._on_class_changed)

        self.radio_class3 = QRadioButton(
            "Class 3: End ED - Akhir Epidermis/Dermis (Hijau - Nilai 3) [3]"
        )
        self.radio_class3.setStyleSheet(
            "QRadioButton { font-weight: bold; color: #4ade80; }"
        )
        self.radio_class3.toggled.connect(self._on_class_changed)

        self.radio_eraser = QRadioButton(
            "Eraser / Hapus (Nilai 0) [Key: 0 / Klik Kanan]"
        )
        self.radio_eraser.setStyleSheet(
            "QRadioButton { font-weight: 600; color: #38bdf8; }"
        )
        self.radio_eraser.toggled.connect(self._on_class_changed)

        tools_layout.addWidget(self.radio_class1)
        tools_layout.addWidget(self.radio_class2)
        tools_layout.addWidget(self.radio_class3)
        tools_layout.addWidget(self.radio_eraser)

        # Pemisah tipis
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2c313c;")
        tools_layout.addWidget(sep)

        # Ketebalan Brush (Default 2px)
        brush_row = QHBoxLayout()
        brush_row.setSpacing(8)
        brush_row.addWidget(QLabel("Ketebalan Brush (Default 2px):"))
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setRange(1, 25)
        self.slider_brush.setValue(2)
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 25)
        self.spin_brush.setValue(2)
        self.spin_brush.setSuffix(" px")
        self.spin_brush.setFixedWidth(68)

        self.slider_brush.valueChanged.connect(self.spin_brush.setValue)
        self.spin_brush.valueChanged.connect(self.slider_brush.setValue)
        self.spin_brush.valueChanged.connect(self._on_brush_size_changed)

        brush_row.addWidget(self.slider_brush, stretch=1)
        brush_row.addWidget(self.spin_brush)
        tools_layout.addLayout(brush_row)

        # Opasitas Overlay
        alpha_row = QHBoxLayout()
        alpha_row.setSpacing(8)
        alpha_row.addWidget(QLabel("Opasitas Overlay:"))
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(20, 100)
        self.slider_alpha.setValue(75)
        self.lbl_alpha_val = QLabel("75%")
        self.lbl_alpha_val.setFixedWidth(36)
        self.lbl_alpha_val.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        alpha_row.addWidget(self.slider_alpha, stretch=1)
        alpha_row.addWidget(self.lbl_alpha_val)
        tools_layout.addLayout(alpha_row)

        self.chk_show_overlay = QCheckBox("Tampilkan Live Overlay [Key: Space]")
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.toggled.connect(self._on_toggle_overlay)
        tools_layout.addWidget(self.chk_show_overlay)

        edit_btn_row = QHBoxLayout()
        edit_btn_row.setSpacing(8)
        self.btn_undo = QPushButton("Undo (Ctrl+Z)")
        self.btn_undo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_undo.clicked.connect(self.undo_annotation)

        self.btn_clear = QPushButton("Clear Label (C)")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setStyleSheet(
            "QPushButton { color: #fca5a5; border-color: #7f1d1d; background-color: #2a181b; }"
            "QPushButton:hover { background-color: #3f1d24; border-color: #b91c1c; }"
        )
        self.btn_clear.clicked.connect(self.clear_annotation)
        edit_btn_row.addWidget(self.btn_undo)
        edit_btn_row.addWidget(self.btn_clear)
        tools_layout.addLayout(edit_btn_row)

        left_layout.addWidget(tools_group)
        left_layout.addStretch()

        left_scroll.setWidget(left_panel)

        # =====================================================================
        # KOLOM TENGAH: Header Info + Area Kanvas Gambar (512x512)
        # =====================================================================
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        # Top Info Bar di atas kanvas
        info_bar = QFrame()
        info_bar.setStyleSheet(
            "QFrame { background-color: #1a1d24; border: 1px solid #2c313c; border-radius: 8px; }"
        )
        info_bar_layout = QHBoxLayout(info_bar)
        info_bar_layout.setContentsMargins(12, 6, 12, 6)
        info_bar_layout.setSpacing(12)

        self.lbl_active_file = QLabel("File: (Belum ada gambar)")
        self.lbl_active_file.setStyleSheet(
            "font-weight: bold; color: #f0f4f8; font-size: 12px; border: none;"
        )

        self.lbl_mask_stats = QLabel(
            "top SC: 0 px | bot SC: 0 px | end ED: 0 px"
        )
        self.lbl_mask_stats.setStyleSheet(
            "color: #38bdf8; font-weight: bold; font-size: 12px; border: none;"
        )

        info_bar_layout.addWidget(self.lbl_active_file)
        info_bar_layout.addStretch()
        info_bar_layout.addWidget(self.lbl_mask_stats)
        center_layout.addWidget(info_bar)

        self.canvas = AnnotationCanvas()
        self.canvas.maskChanged.connect(self._on_mask_modified)
        self.canvas.cursorMoved.connect(self._on_cursor_moved)
        center_layout.addWidget(self.canvas, stretch=1)

        # =====================================================================
        # KOLOM KANAN: Daftar Gambar (Progres) & Navigasi/Simpan
        # =====================================================================
        right_panel = QWidget()
        right_panel.setFixedWidth(285)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 4. Daftar File Gambar (QListWidget & Random Sampling)
        list_group = QGroupBox("4. Daftar Titik / Sampel (QListWidget)")
        list_layout = QVBoxLayout(list_group)
        list_layout.setSpacing(6)

        self.lbl_progress = QLabel("Target Titik: 0 / 1 sampel")
        self.lbl_progress.setStyleSheet("color: #b0b8c4; font-size: 11px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.btn_random = QPushButton("🎲 Ganti Slice Acak (Titik Ini) [R]")
        self.btn_random.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_random.setToolTip(
            "Mengambil slice acak lain dari titik saat ini (1 subjek = 4 titik, 1 titik = 1 sampel)."
        )
        self.btn_random.setStyleSheet(
            "QPushButton { background-color: #6d28d9; border-color: #7c3aed; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #7c3aed; }"
        )
        self.btn_random.clicked.connect(self.random_slice)

        self.slice_selector_widget = QWidget()
        slice_sel_layout = QVBoxLayout(self.slice_selector_widget)
        slice_sel_layout.setContentsMargins(0, 0, 0, 0)
        slice_sel_layout.setSpacing(2)
        lbl_slice_sel = QLabel("🎞️ Pilih / Revisi Slice Titik Ini:")
        lbl_slice_sel.setStyleSheet("color: #b0b8c4; font-size: 11px;")
        self.combo_point_slice = QComboBox()
        self.combo_point_slice.currentIndexChanged.connect(self._on_point_slice_selected)
        slice_sel_layout.addWidget(lbl_slice_sel)
        slice_sel_layout.addWidget(self.combo_point_slice)
        self.slice_selector_widget.setVisible(False)

        self.sibling_nav_widget = QWidget()
        sib_row = QHBoxLayout(self.sibling_nav_widget)
        sib_row.setContentsMargins(0, 0, 0, 0)
        sib_row.setSpacing(6)
        self.btn_prev_point = QPushButton("⏮ Titik Sblm")
        self.btn_prev_point.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev_point.setStyleSheet(
            "QPushButton { background-color: #0f766e; border-color: #14b8a6; color: white; font-weight: bold; }"
        )
        self.btn_prev_point.clicked.connect(lambda: self.switch_sibling_point(-1))
        self.btn_next_point = QPushButton("Titik Brkt ⏭")
        self.btn_next_point.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next_point.setStyleSheet(
            "QPushButton { background-color: #0f766e; border-color: #14b8a6; color: white; font-weight: bold; }"
        )
        self.btn_next_point.clicked.connect(lambda: self.switch_sibling_point(1))
        sib_row.addWidget(self.btn_prev_point)
        sib_row.addWidget(self.btn_next_point)
        self.sibling_nav_widget.setVisible(False)

        self.file_list_widget = QListWidget()
        self.file_list_widget.currentRowChanged.connect(self._on_list_row_changed)

        list_layout.addWidget(self.lbl_progress)
        list_layout.addWidget(self.progress_bar)
        list_layout.addWidget(self.btn_random)
        list_layout.addWidget(self.slice_selector_widget)
        list_layout.addWidget(self.sibling_nav_widget)
        list_layout.addWidget(self.file_list_widget, stretch=1)
        right_layout.addWidget(list_group, stretch=1)

        # 5. Grup Navigasi, Simpan & Revisi
        nav_group = QGroupBox("5. Navigasi, Simpan && Revisi")
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setSpacing(8)

        nav_btn_row = QHBoxLayout()
        nav_btn_row.setSpacing(8)
        self.btn_prev = QPushButton("<< Previous (A)")
        self.btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev.clicked.connect(self.prev_image)

        self.btn_next = QPushButton("Next (D) >>")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.clicked.connect(self.next_image)
        nav_btn_row.addWidget(self.btn_prev)
        nav_btn_row.addWidget(self.btn_next)
        nav_layout.addLayout(nav_btn_row)

        self.chk_autosave = QCheckBox("Auto-Save saat pindah gambar")
        self.chk_autosave.setChecked(False)
        nav_layout.addWidget(self.chk_autosave)

        self.lbl_revision_badge = QLabel(
            "✏️ Mode Revisi: Sampel ini sudah tersimpan. Edit garis lalu tekan Simpan Revisi untuk menghapus & mengganti file lama."
        )
        self.lbl_revision_badge.setWordWrap(True)
        self.lbl_revision_badge.setStyleSheet(
            "background-color: #3b2506; border: 1px solid #d97706; color: #fcd34d; border-radius: 6px; padding: 6px; font-size: 11px;"
        )
        self.lbl_revision_badge.setVisible(False)
        nav_layout.addWidget(self.lbl_revision_badge)

        self.btn_save = QPushButton("SIMPAN CEPAT: Crop + Mask (Ctrl+S)")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setMinimumHeight(40)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #2b8a3e; color: white; font-weight: bold; font-size: 12px; border: 1px solid #37b24d; border-radius: 6px; }"
            "QPushButton:hover { background-color: #2f9e44; }"
            "QPushButton:pressed { background-color: #237032; }"
        )
        self.btn_save.clicked.connect(self.save_current)
        nav_layout.addWidget(self.btn_save)

        self.btn_delete_saved = QPushButton("🗑️ Hapus Gambar Tersimpan (Revisi Ulang)")
        self.btn_delete_saved.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_saved.setStyleSheet(
            "QPushButton { color: #fca5a5; border-color: #7f1d1d; background-color: #2a181b; font-weight: 600; }"
            "QPushButton:hover { background-color: #3f1d24; border-color: #b91c1c; }"
        )
        self.btn_delete_saved.setVisible(False)
        self.btn_delete_saved.clicked.connect(self.delete_saved_revision)
        nav_layout.addWidget(self.btn_delete_saved)

        right_layout.addWidget(nav_group)

        # Gabungkan ketiga kolom ke layout utama
        main_layout.addWidget(left_scroll)
        main_layout.addWidget(center_panel, stretch=1)
        main_layout.addWidget(right_panel)

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

        act_cls3 = QAction(self)
        act_cls3.setShortcut(QKeySequence("3"))
        act_cls3.triggered.connect(lambda: self.radio_class3.setChecked(True))
        self.addAction(act_cls3)

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

        act_random = QAction(self)
        act_random.setShortcut(QKeySequence("R"))
        act_random.triggered.connect(self.random_slice)
        self.addAction(act_random)

    # =========================================================================
    # Manajemen Direktori & Pemuatan Gambar (1 Subjek = 4 Titik, 1 Sampel/Titik)
    # =========================================================================
    @staticmethod
    def _clean_point_tag(folder_name: str) -> str:
        if folder_name.lower().endswith(".bin"):
            return folder_name[:-4]
        return folder_name

    def _get_point_tag(self, src_path: Path) -> str:
        return self._clean_point_tag(src_path.parent.name)

    def _get_save_stem(self, src_path: Path) -> str:
        stem = src_path.stem
        ptag = self._get_point_tag(src_path)
        if self.output_dir is not None:
            out_clean = self._clean_point_tag(self.output_dir.name)
            if out_clean.lower() == ptag.lower():
                return stem
        if stem.lower().startswith(ptag.lower()):
            return stem
        return f"{ptag}_{stem}"

    def _find_mask_for_source(self, src_path: Path) -> Optional[Path]:
        if self.output_dir is None:
            return None
        stem = src_path.stem
        ptag = self._get_point_tag(src_path)
        pstem = f"{ptag}_{stem}" if not stem.lower().startswith(ptag.lower()) else stem
        out_matches_pt = self._clean_point_tag(self.output_dir.name).lower() == ptag.lower()

        candidates = [
            self.output_dir / f"{pstem}_mask.png",
            self.output_dir / "masks" / f"{pstem}_mask.png",
            self.output_dir / ptag / f"{stem}_mask.png",
            self.output_dir / ptag / "masks" / f"{stem}_mask.png",
            self.output_dir / ptag / f"{pstem}_mask.png",
            self.output_dir / ptag / "masks" / f"{pstem}_mask.png",
        ]
        if out_matches_pt:
            candidates.extend(
                [
                    self.output_dir / f"{stem}_mask.png",
                    self.output_dir / "masks" / f"{stem}_mask.png",
                ]
            )
        if self.output_dir.parent != self.output_dir:
            candidates.extend(
                [
                    self.output_dir.parent / ptag / f"{stem}_mask.png",
                    self.output_dir.parent / ptag / "masks" / f"{stem}_mask.png",
                    self.output_dir.parent / ptag / f"{pstem}_mask.png",
                    self.output_dir.parent / ptag / "masks" / f"{pstem}_mask.png",
                    self.output_dir.parent / f"{pstem}_mask.png",
                    self.output_dir.parent / "masks" / f"{pstem}_mask.png",
                ]
            )
        for c in candidates:
            if c.exists():
                return c
        return None

    def _delete_source_outputs(self, src_path: Path) -> None:
        """Menghapus semua file hasil simpan (_crop, _mask, _overlay, _strip) milik satu file sumber."""
        if self.output_dir is None:
            return
        stem = src_path.stem
        ptag = self._get_point_tag(src_path)
        pstem = f"{ptag}_{stem}" if not stem.lower().startswith(ptag.lower()) else stem
        out_matches_pt = self._clean_point_tag(self.output_dir.name).lower() == ptag.lower()

        dir_stem_pairs: List[Tuple[Path, str]] = [
            (self.output_dir, pstem),
            (self.output_dir / ptag, pstem),
            (self.output_dir / ptag, stem),
        ]
        if out_matches_pt:
            dir_stem_pairs.append((self.output_dir, stem))
        if self.output_dir.parent != self.output_dir:
            dir_stem_pairs.extend(
                [
                    (self.output_dir.parent, pstem),
                    (self.output_dir.parent / ptag, pstem),
                    (self.output_dir.parent / ptag, stem),
                ]
            )

        for base_dir, s in dir_stem_pairs:
            if not base_dir.exists():
                continue
            targets = [
                base_dir / f"{s}_crop.png",
                base_dir / f"{s}_mask.png",
                base_dir / f"{s}_thickness.csv",
                base_dir / "images" / f"{s}_crop.png",
                base_dir / "masks" / f"{s}_mask.png",
                base_dir / "overlays" / f"{s}_overlay.png",
                base_dir / "strips" / f"{s}_strip.png",
                base_dir / "strips_mask" / f"{s}_strip_mask.png",
            ]
            for t in targets:
                try:
                    if t.exists():
                        t.unlink()
                except Exception:
                    pass

    def _delete_point_outputs(
        self, pt_files: List[Path], except_src: Optional[Path] = None
    ) -> None:
        """Menghapus semua file hasil simpan pada suatu titik (kecuali except_src bila ditentukan) agar 1 titik = maksimal 1 sampel."""
        for p in pt_files:
            if except_src is not None and p == except_src:
                continue
            self._delete_source_outputs(p)

    def _list_images_in_dir(self, d: Path) -> List[Path]:
        try:
            return [
                p
                for p in sorted(d.iterdir())
                if p.is_file()
                and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
                and not p.stem.startswith("000_TRANSIT_OCT_")
            ]
        except Exception:
            return []

    def select_input_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Pilih Folder Sumber Gambar OCT Mentah (Folder Titik atau Folder Dataset)"
        )
        if not folder:
            return

        self._set_input_directory(Path(folder))

    def _set_input_directory(self, new_in: Path) -> None:
        old_in = self.input_dir
        self.input_dir = new_in
        self.lbl_input_dir.setText(f"Sumber: {self.input_dir}")

        if self.output_dir is None:
            default_out = self.input_dir.parent / f"{self._clean_point_tag(self.input_dir.name)}_output"
            self.output_dir = default_out
        elif old_in is not None:
            old_tag = self._clean_point_tag(old_in.name)
            new_tag = self._clean_point_tag(new_in.name)
            if self.output_dir.name.lower() == old_tag.lower():
                self.output_dir = self.output_dir.parent / new_tag
            elif self.output_dir.name.lower() == f"{old_in.name.lower()}_output":
                self.output_dir = self.output_dir.parent / f"{new_in.name}_output"

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

        direct_files = self._list_images_in_dir(self.input_dir)
        point_groups: dict[str, List[Path]] = {}

        if direct_files:
            point_groups[self.input_dir.name] = direct_files
            parent = self.input_dir.parent
            siblings = []
            if parent.exists() and parent != self.input_dir:
                try:
                    for sub in sorted(parent.iterdir(), key=lambda x: x.name.lower()):
                        if (
                            sub.is_dir()
                            and not sub.name.startswith(".")
                            and sub.name.lower() not in self.RESERVED_DIR_NAMES
                            and not sub.name.endswith("_output")
                        ):
                            if self._list_images_in_dir(sub):
                                siblings.append(sub)
                except Exception:
                    pass
            self.sibling_points = siblings
        else:
            self.sibling_points = []
            try:
                for sub in sorted(self.input_dir.iterdir(), key=lambda x: x.name.lower()):
                    if (
                        not sub.is_dir()
                        or sub.name.startswith(".")
                        or sub.name.lower() in self.RESERVED_DIR_NAMES
                        or sub.name.endswith("_output")
                    ):
                        continue
                    sub_files = self._list_images_in_dir(sub)
                    if sub_files:
                        point_groups[sub.name] = sub_files
                    else:
                        for sub2 in sorted(sub.iterdir(), key=lambda x: x.name.lower()):
                            if (
                                not sub2.is_dir()
                                or sub2.name.startswith(".")
                                or sub2.name.lower() in self.RESERVED_DIR_NAMES
                                or sub2.name.endswith("_output")
                            ):
                                continue
                            sub2_files = self._list_images_in_dir(sub2)
                            if sub2_files:
                                point_groups[f"{sub.name}/{sub2.name}"] = sub2_files
            except Exception:
                pass

        self.point_groups = point_groups
        one_sample = self.chk_one_sample_per_point.isChecked()

        if len(point_groups) >= 1 and one_sample:
            sampled_list: List[Path] = []
            for pt_key, pt_files in point_groups.items():
                labeled_in_pt = [p for p in pt_files if self._is_file_labeled(p)]
                if labeled_in_pt:
                    chosen = labeled_in_pt[0]
                    if len(labeled_in_pt) > 1:
                        self._delete_point_outputs(pt_files, except_src=chosen)
                    self.selected_point_sample[pt_key] = chosen
                else:
                    prev_chosen = self.selected_point_sample.get(pt_key)
                    if prev_chosen in pt_files:
                        chosen = prev_chosen
                    else:
                        chosen = random.choice(pt_files)
                        self.selected_point_sample[pt_key] = chosen
                sampled_list.append(chosen)
            self.image_files = sampled_list
        else:
            all_files: List[Path] = []
            for pt_key, pt_files in point_groups.items():
                all_files.extend(pt_files)
                if pt_files and pt_key not in self.selected_point_sample:
                    labeled_in_pt = [p for p in pt_files if self._is_file_labeled(p)]
                    self.selected_point_sample[pt_key] = (
                        labeled_in_pt[0] if labeled_in_pt else random.choice(pt_files)
                    )
            self.image_files = all_files

        has_siblings = len(point_groups) == 1 and len(self.sibling_points) > 1
        self.btn_load_parent_points.setVisible(has_siblings)
        if has_siblings:
            self.btn_load_parent_points.setText(
                f"📂 Gabung Semua Titik ({len(self.sibling_points)} Titik — 1 Sampel/Titik)"
            )
        self.sibling_nav_widget.setVisible(has_siblings)

        self.file_list_widget.blockSignals(True)
        self.file_list_widget.clear()
        for p in self.image_files:
            item = QListWidgetItem(self._get_file_display_name(p))
            self.file_list_widget.addItem(item)
        self.file_list_widget.blockSignals(False)

        self._refresh_file_list_status()

        if self.image_files:
            start_idx = 0
            if len(point_groups) == 1:
                labeled_indices = [
                    i for i, p in enumerate(self.image_files) if self._is_file_labeled(p)
                ]
                if labeled_indices:
                    start_idx = labeled_indices[0]
                else:
                    pt_key = next(iter(point_groups.keys()))
                    chosen_p = self.selected_point_sample.get(pt_key)
                    if chosen_p in self.image_files:
                        start_idx = self.image_files.index(chosen_p)
            else:
                unlabeled_indices = [
                    i
                    for i, p in enumerate(self.image_files)
                    if not self._is_file_labeled(p)
                ]
                start_idx = unlabeled_indices[0] if unlabeled_indices else 0

            self.file_list_widget.setCurrentRow(start_idx)
            self.load_image_at_index(start_idx)
            self.status_bar.showMessage(
                f"Berhasil memuat {len(self.image_files)} sampel dari {len(point_groups)} titik ({self.input_dir})"
            )
        else:
            self.current_index = -1
            self.current_raw_img = None
            self.canvas.roi_bgr = None
            self.canvas.update_display()
            self.lbl_active_file.setText("File: (Belum ada gambar)")
            QMessageBox.warning(
                self,
                "Folder Kosong",
                "Tidak ditemukan file gambar (.jpg, .png, dll.) pada folder sumber tersebut.",
            )

    def _get_file_display_name(self, src_path: Path) -> str:
        ptag = self._get_point_tag(src_path)
        if self.chk_one_sample_per_point.isChecked():
            total_in_pt = 0
            for pt_files in self.point_groups.values():
                if src_path in pt_files:
                    total_in_pt = len(pt_files)
                    break
            if total_in_pt > 1:
                return f"[{ptag}] {src_path.name} (1 dari {total_in_pt} slice)"
            return f"[{ptag}] {src_path.name}"
        if len(self.point_groups) > 1:
            return f"[{ptag}] {src_path.name}"
        return src_path.name

    def _get_output_paths(self, src_path: Path) -> Tuple[Path, Path]:
        assert self.output_dir is not None
        save_stem = self._get_save_stem(src_path)
        crop_path = self.output_dir / f"{save_stem}_crop.png"
        mask_path = self.output_dir / f"{save_stem}_mask.png"
        return crop_path, mask_path

    def _is_file_labeled(self, src_path: Path) -> bool:
        return self._find_mask_for_source(src_path) is not None

    def _refresh_file_list_status(self) -> None:
        labeled_count = 0
        multi_pt = len(self.point_groups) > 1
        one_sample = self.chk_one_sample_per_point.isChecked()
        for idx, p in enumerate(self.image_files):
            item = self.file_list_widget.item(idx)
            if item is None:
                continue
            disp_name = self._get_file_display_name(p)
            if self._is_file_labeled(p):
                labeled_count += 1
                item.setText(f"[✓] {disp_name}")
                item.setForeground(QColor("#4ade80"))
            else:
                item.setText(f"[ ] {disp_name}")
                item.setForeground(QColor("#d8dde6"))

        if multi_pt:
            total_pts = len(self.point_groups)
            labeled_pts = 0
            for pt_key, pt_files in self.point_groups.items():
                if one_sample:
                    chosen_p = self.selected_point_sample.get(pt_key)
                    if chosen_p is not None and self._is_file_labeled(chosen_p):
                        labeled_pts += 1
                else:
                    if any(self._is_file_labeled(p) for p in pt_files):
                        labeled_pts += 1
            self.lbl_progress.setText(
                f"Selesai: {labeled_pts} / {total_pts} titik (1 sampel/titik)"
            )
            self.progress_bar.setRange(0, max(1, total_pts))
            self.progress_bar.setValue(labeled_pts)
        else:
            done1 = 1 if labeled_count >= 1 else 0
            self.lbl_progress.setText(
                "✓ Target Titik Tercapai: 1 / 1 sampel selesai"
                if done1
                else "Target Titik Ini: 0 / 1 sampel (Pilih 1 slice saja)"
            )
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(done1)

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
            candidate = self._find_mask_for_source(src_path)
            if candidate is not None and candidate.exists():
                loaded_mask = imread_unicode(str(candidate), cv2.IMREAD_GRAYSCALE)
                if loaded_mask is not None and loaded_mask.shape == (CROP_SIZE, CROP_SIZE):
                    existing_mask = loaded_mask

        self.current_index = index
        meta_path = src_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                self.canvas.frame_segments = meta_obj.get("segments", [])
            except Exception:
                self.canvas.frame_segments = []
        else:
            self.canvas.frame_segments = []
        self.canvas.set_image_and_mask(cropped_bgr, existing_mask)
        self.is_dirty = False

        ptag = self._get_point_tag(src_path)
        self.lbl_active_file.setText(
            f"File [{index + 1}/{len(self.image_files)}] [{ptag}]: {src_path.name}"
        )
        self.lbl_raw_dim.setText(f"Asli: {raw_w} × {raw_h} px")
        self.lbl_crop_dim.setText(f"ROI: X[{sx}:{ex}], Y[{sy}:{ey}]")
        self._update_mask_stats()
        self._update_revision_ui(existing_mask is not None)

        # Perbarui dropdown slice untuk titik aktif
        pt_slices: List[Path] = []
        for pt_files in self.point_groups.values():
            if src_path in pt_files:
                pt_slices = pt_files
                break
        if len(pt_slices) > 1:
            self.slice_selector_widget.setVisible(True)
            self.combo_point_slice.blockSignals(True)
            self.combo_point_slice.clear()
            active_s_idx = 0
            for s_idx, sp in enumerate(pt_slices):
                label_txt = f"{s_idx + 1}. {sp.name}" + (" (Aktif)" if sp == src_path else "")
                self.combo_point_slice.addItem(label_txt, sp.name)
                if sp == src_path:
                    active_s_idx = s_idx
            self.combo_point_slice.setCurrentIndex(active_s_idx)
            self.combo_point_slice.blockSignals(False)
        else:
            self.slice_selector_widget.setVisible(False)

    def _update_revision_ui(self, is_labeled: bool) -> None:
        self.lbl_revision_badge.setVisible(is_labeled)
        self.btn_delete_saved.setVisible(is_labeled)
        if is_labeled:
            self.btn_save.setText("💾 SIMPAN REVISI: Ganti File Lama (Ctrl+S)")
            self.btn_save.setStyleSheet(
                "QPushButton { background-color: #d97706; color: white; font-weight: bold; font-size: 12px; border: 1px solid #f59e0b; border-radius: 6px; }"
                "QPushButton:hover { background-color: #b45309; }"
            )
        else:
            self.btn_save.setText("SIMPAN CEPAT: Crop + Mask (Ctrl+S)")
            self.btn_save.setStyleSheet(
                "QPushButton { background-color: #2b8a3e; color: white; font-weight: bold; font-size: 12px; border: 1px solid #37b24d; border-radius: 6px; }"
                "QPushButton:hover { background-color: #2f9e44; }"
                "QPushButton:pressed { background-color: #237032; }"
            )

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
        self.lbl_crop_dim.setText(f"ROI: X[{sx}:{ex}], Y[{sy}:{ey}]")

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
        if self.chk_lock_crop.isChecked():
            self.status_bar.showMessage(
                "🔒 Lokasi Crop sedang dikunci! Buka centang 'Patenkan / Kunci Lokasi' jika ingin menggeser manual."
            )
            return
        self.spin_start_x.blockSignals(True)
        self.spin_start_y.blockSignals(True)
        self.spin_start_x.setValue(new_sx)
        self.spin_start_y.setValue(new_sy)
        self.spin_start_x.blockSignals(False)
        self.spin_start_y.blockSignals(False)
        self._on_custom_xy_changed()

    def _on_lock_crop_toggled(self, locked: bool) -> None:
        self.spin_start_x.setEnabled(not locked)
        self.spin_start_y.setEnabled(not locked)

    def _on_slice_guide_toggled(self, checked: bool) -> None:
        self.canvas.show_slice_guide = checked
        self.canvas.update_display()

    def _on_strip_width_changed(self, val: int) -> None:
        self.canvas.strip_width = val
        self.canvas.update_display()

    def stitch_all_frames(self) -> None:
        if self.input_dir is None or not self.image_files:
            QMessageBox.warning(self, "Peringatan", "Pilih Folder Sumber terlebih dahulu.")
            return
        if self.output_dir is None:
            self.output_dir = self.input_dir.parent / f"{self.input_dir.name}_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        source_files = [
            p for p in self.image_files if not p.stem.startswith("000_TRANSIT_OCT_")
        ]
        if not source_files:
            return

        mode = self.combo_crop_mode.currentData()
        cx_val = self.spin_start_x.value()
        cy_val = self.spin_start_y.value()
        frames_per_group = int(self.spin_frames_per_group.value())

        if frames_per_group > 0 and len(source_files) > frames_per_group:
            groups = [
                source_files[i : i + frames_per_group]
                for i in range(0, len(source_files), frames_per_group)
            ]
        else:
            groups = [source_files]

        stitch_dir = self.output_dir / "stitched"
        stitch_dir.mkdir(parents=True, exist_ok=True)
        folder_name = self.input_dir.name
        created_files = []

        for g_idx, group_paths in enumerate(groups):
            k_frames = len(group_paths)
            if k_frames == 0:
                continue
            base_w = max(1, CROP_SIZE // k_frames)
            rem = CROP_SIZE - (base_w * k_frames) if base_w * k_frames <= CROP_SIZE else 0
            strips = []
            segments_meta = []
            cur_x = 0

            for f_idx, p in enumerate(group_paths):
                raw = imread_unicode(str(p), cv2.IMREAD_COLOR)
                if raw is None:
                    continue
                raw = ensure_min_size(raw, CROP_SIZE)
                sx, sy, ex, ey = compute_crop_coords(
                    raw, mode=mode, custom_x=cx_val, custom_y=cy_val, crop_size=CROP_SIZE
                )
                roi = raw[sy:ey, sx:ex]
                w_this = max(1, min(CROP_SIZE, base_w + (1 if f_idx < rem else 0)))
                x0 = max(0, min(CROP_SIZE - w_this, 256 - w_this // 2))
                x1 = x0 + w_this
                col_strip = roi[:, x0:x1].copy()
                if col_strip.shape[0] == CROP_SIZE:
                    strips.append(col_strip)
                    segments_meta.append(
                        {
                            "frame_index": f_idx + 1,
                            "name": p.name,
                            "x0": cur_x,
                            "x1": cur_x + w_this,
                        }
                    )
                    cur_x += w_this

            if not strips:
                continue

            stitched = np.concatenate(strips, axis=1)
            stitched_512 = (
                cv2.resize(stitched, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                if stitched.shape[1] != CROP_SIZE
                else stitched
            )
            group_tag = (
                f"{folder_name}_G{g_idx + 1:02d}_N{len(strips)}"
                if len(groups) > 1
                else f"{folder_name}_N{len(strips)}"
            )
            imwrite_unicode(str(stitch_dir / f"Transit_OCT_{group_tag}.png"), stitched)
            in_stitch_path = self.input_dir / f"000_TRANSIT_OCT_{group_tag}.png"
            imwrite_unicode(str(in_stitch_path), stitched_512)
            in_stitch_path.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "group_tag": group_tag,
                        "num_frames": len(strips),
                        "segments": segments_meta,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            created_files.append(in_stitch_path.name)

        if not created_files:
            QMessageBox.warning(self, "Gagal", "Tidak dapat membaca frame.")
            return

        self._scan_input_images()
        QMessageBox.information(
            self,
            "Berhasil Menggabungkan Frame",
            f"Berhasil membuat {len(created_files)} gambar gabungan ({created_files[0]})!\n\n"
            "Gambar gabungan sudah dimuat di urutan #1 — Anda cukup melabeli 1 gambar ini saja!",
        )

    # =========================================================================
    # Handler Event Alat Anotasi
    # =========================================================================
    def _on_class_changed(self) -> None:
        if self.radio_class1.isChecked():
            self.canvas.active_class = 1
        elif self.radio_class2.isChecked():
            self.canvas.active_class = 2
        elif self.radio_class3.isChecked():
            self.canvas.active_class = 3
        else:
            self.canvas.active_class = 0
        self.canvas.update_display()

    def _on_brush_size_changed(self, value: int) -> None:
        self.canvas.brush_size = value
        self.canvas.update_display()

    def _on_alpha_changed(self, value: int) -> None:
        self.lbl_alpha_val.setText(f"{value}%")
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
        c3 = int(np.sum(self.canvas.mask == 3))
        sc_t, ed_t = compute_layer_thickness(self.canvas.mask)
        sc_str = f"{sc_t:.1f} px" if sc_t is not None else "-"
        ed_str = f"{ed_t:.1f} px" if ed_t is not None else "-"
        self.lbl_mask_stats.setText(
            f"top SC: {c1} px | bot SC: {c2} px | end ED: {c3} px   ||   "
            f"Tebal SC (1→2): {sc_str} | Tebal ED (2→3): {ed_str}"
        )

    def _on_cursor_moved(self, x: int, y: int, mask_val: int) -> None:
        cls_names = {
            0: "Background (0)",
            1: "top SC - Batas Atas (1)",
            2: "bot SC - Batas Tengah (2)",
            3: "end ED - Batas Bawah (3)",
        }
        cls_label = cls_names.get(mask_val, str(mask_val))
        self.status_bar.showMessage(
            f"Koordinat ROI: X={x}, Y={y}  |  Nilai Piksel Mask: {cls_label}"
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
        save_stem = self._get_save_stem(src_path)
        crop_path, mask_path = self._get_output_paths(src_path)

        # Deteksi apakah titik/gambar ini sebelumnya sudah memiliki label tersimpan (Mode Revisi)
        was_revision = self._is_file_labeled(src_path)
        if self.chk_one_sample_per_point.isChecked():
            for pt_key, pt_files in self.point_groups.items():
                if src_path in pt_files:
                    if any(self._is_file_labeled(p) for p in pt_files):
                        was_revision = True
                    self.selected_point_sample[pt_key] = src_path
                    # Hapus seluruh file lama milik titik ini sebelum menulis gambar & mask revisi baru
                    self._delete_point_outputs(pt_files)
                    break
        else:
            if was_revision:
                self._delete_source_outputs(src_path)

        clean_mask = np.clip(self.canvas.mask, 0, MAX_CLASS_ID).astype(np.uint8)
        overlay_vis = render_reference_overlay(self.canvas.roi_bgr, clean_mask)

        ok_crop = imwrite_unicode(str(crop_path), self.canvas.roi_bgr)
        ok_mask = imwrite_unicode(str(mask_path), clean_mask)

        ovl_dir = self.output_dir / "overlays"
        ovl_dir.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(str(ovl_dir / f"{save_stem}_overlay.png"), overlay_vis)

        # Simpan juga potongan vertikal (Area yang diambil) di posisi X=256 yang terkunci
        sw = max(1, min(64, int(self.spin_strip_w.value())))
        x0 = max(0, min(CROP_SIZE - sw, 256 - sw // 2))
        x1 = x0 + sw
        strip_dir = self.output_dir / "strips"
        strip_msk_dir = self.output_dir / "strips_mask"
        strip_dir.mkdir(parents=True, exist_ok=True)
        strip_msk_dir.mkdir(parents=True, exist_ok=True)

        meta_path = src_path.with_suffix(".meta.json")
        if meta_path.exists() and self.canvas.frame_segments:
            csv_lines = [
                "frame_index,source_file,x_start,x_end,top_SC_y,bot_SC_y,end_ED_y,tebal_SC_px,tebal_ED_px"
            ]
            for seg_idx, seg in enumerate(self.canvas.frame_segments):
                sx0, sx1 = int(seg["x0"]), int(seg["x1"])
                orig_stem = Path(seg["name"]).stem
                s_bgr = self.canvas.roi_bgr[:, sx0:sx1].copy()
                s_msk = clean_mask[:, sx0:sx1].copy()
                imwrite_unicode(str(strip_dir / f"{orig_stem}_strip.png"), s_bgr)
                imwrite_unicode(str(strip_msk_dir / f"{orig_stem}_strip_mask.png"), s_msk)
                y_ys = {}
                for cid in (1, 2, 3):
                    ys, _ = np.where(s_msk == cid)
                    y_ys[cid] = float(np.mean(ys)) if ys.size > 0 else None
                sc_t = (
                    (y_ys[2] - y_ys[1])
                    if (y_ys[1] is not None and y_ys[2] is not None)
                    else None
                )
                ed_t = (
                    (y_ys[3] - y_ys[2])
                    if (y_ys[2] is not None and y_ys[3] is not None)
                    else None
                )
                csv_lines.append(
                    f"{seg_idx + 1},{seg['name']},{sx0},{sx1},"
                    f"{f'{y_ys[1]:.2f}' if y_ys[1] is not None else ''},"
                    f"{f'{y_ys[2]:.2f}' if y_ys[2] is not None else ''},"
                    f"{f'{y_ys[3]:.2f}' if y_ys[3] is not None else ''},"
                    f"{f'{sc_t:.2f}' if sc_t is not None else ''},"
                    f"{f'{ed_t:.2f}' if ed_t is not None else ''}"
                )
            (self.output_dir / f"{save_stem}_thickness.csv").write_text(
                "\n".join(csv_lines), encoding="utf-8"
            )
        else:
            imwrite_unicode(
                str(strip_dir / f"{save_stem}_strip.png"),
                self.canvas.roi_bgr[:, x0:x1].copy(),
            )
            imwrite_unicode(
                str(strip_msk_dir / f"{save_stem}_strip_mask.png"),
                clean_mask[:, x0:x1].copy(),
            )

        if self.chk_subfolder_mode.isChecked():
            img_dir = self.output_dir / "images"
            msk_dir = self.output_dir / "masks"
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(str(img_dir / f"{save_stem}_crop.png"), self.canvas.roi_bgr)
            imwrite_unicode(str(msk_dir / f"{save_stem}_mask.png"), clean_mask)

        if ok_crop and ok_mask:
            self.is_dirty = False
            self._refresh_file_list_status()
            self._update_revision_ui(True)
            unique_vals = np.unique(clean_mask).tolist()
            action_prefix = (
                "✓ Revisi Tersimpan (File lama dihapus & diganti)"
                if was_revision
                else "✓ Tersimpan"
            )
            self.status_bar.showMessage(
                f"{action_prefix}: {crop_path.name} & {mask_path.name} (Nilai unik mask: {unique_vals})"
            )
            return True
        else:
            QMessageBox.critical(
                self,
                "Gagal Menyimpan",
                f"Terjadi kesalahan saat menyimpan ke:\n{self.output_dir}",
            )
            return False

    def _has_any_mask_stroke(self) -> bool:
        return bool(np.any(self.canvas.mask > 0))

    def _on_list_row_changed(self, row: int) -> None:
        if row < 0 or row == self.current_index:
            return
        if self.chk_autosave.isChecked() and self.is_dirty and self._has_any_mask_stroke():
            self.save_current()
        self.load_image_at_index(row)

    def prev_image(self) -> None:
        if not self.image_files or self.current_index <= 0:
            return
        if self.chk_autosave.isChecked() and self.is_dirty and self._has_any_mask_stroke():
            self.save_current()
        new_idx = self.current_index - 1
        self.file_list_widget.setCurrentRow(new_idx)

    def next_image(self) -> None:
        if not self.image_files or self.current_index >= len(self.image_files) - 1:
            return
        if self.chk_autosave.isChecked() and self.is_dirty and self._has_any_mask_stroke():
            self.save_current()
        new_idx = self.current_index + 1
        self.file_list_widget.setCurrentRow(new_idx)

    def random_slice(self) -> None:
        """
        Ganti slice acak pada titik saat ini karena slice aktif strukturnya kurang bagus:
        - JANGAN Auto-Save slice saat ini (batalkan coretan).
        - Hapus file output tersimpan pada titik ini agar status titik kembali BELUM SELESAI (0/1)
          dan menjamin hanya ada maksimal 1 sampel per titik.
        """
        if not self.image_files or self.current_index < 0:
            return

        # Batalkan status dirty agar slice yang ditolak TIDAK tersimpan otomatis
        self.is_dirty = False

        curr_path = self.image_files[self.current_index]
        pt_tag = self._get_point_tag(curr_path)

        target_pt_key = None
        for pt_key, pt_files in self.point_groups.items():
            if curr_path in pt_files:
                target_pt_key = pt_key
                break

        if target_pt_key is not None:
            pt_files = self.point_groups[target_pt_key]
            # Hapus hasil simpan lama pada titik ini agar status kembali Belum Selesai (0/1)
            self._delete_point_outputs(pt_files)

            rej = self.rejected_point_samples.setdefault(target_pt_key, set())
            rej.add(curr_path)
            unseen_candidates = [p for p in pt_files if p not in rej]
            if not unseen_candidates:
                rej.clear()
                rej.add(curr_path)
                unseen_candidates = [p for p in pt_files if p != curr_path] or pt_files

            new_choice = random.choice(unseen_candidates)
            self.selected_point_sample[target_pt_key] = new_choice

            if self.chk_one_sample_per_point.isChecked():
                self.image_files[self.current_index] = new_choice
                self._refresh_file_list_status()
                self.load_image_at_index(self.current_index)
                self.status_bar.showMessage(
                    f"🎲 Titik [{pt_tag}]: Diganti ke {new_choice.name} (Slice lama dibatalkan — status: Belum Selesai)"
                )
                return
            else:
                self._refresh_file_list_status()
                if new_choice in self.image_files:
                    chosen_idx = self.image_files.index(new_choice)
                    self.file_list_widget.setCurrentRow(chosen_idx)
                    self.status_bar.showMessage(
                        f"🎲 Titik [{pt_tag}]: Melompat ke {new_choice.name} (Status direset: Belum Selesai)"
                    )
                    return

    def _on_point_slice_selected(self, combo_idx: int) -> None:
        """Memilih slice tertentu di dalam titik aktif untuk melakukan revisi/penggantian sampel."""
        if combo_idx < 0 or not self.image_files or self.current_index < 0:
            return
        slice_name = self.combo_point_slice.itemData(combo_idx)
        if not slice_name:
            return
        curr_path = self.image_files[self.current_index]
        if curr_path.name == slice_name:
            return

        self.is_dirty = False
        target_pt_key = None
        for pt_key, pt_files in self.point_groups.items():
            if curr_path in pt_files:
                target_pt_key = pt_key
                break
        if target_pt_key is None:
            return

        pt_files = self.point_groups[target_pt_key]
        chosen_path = next((p for p in pt_files if p.name == slice_name), None)
        if chosen_path is None:
            return

        # Hapus hasil simpan slice sebelumnya pada titik ini agar diganti dengan slice baru
        self._delete_point_outputs(pt_files)
        self.selected_point_sample[target_pt_key] = chosen_path
        pt_tag = self._get_point_tag(chosen_path)

        if self.chk_one_sample_per_point.isChecked():
            self.image_files[self.current_index] = chosen_path
            self._refresh_file_list_status()
            self.load_image_at_index(self.current_index)
            self.status_bar.showMessage(
                f"🎞️ Titik [{pt_tag}]: Beralih ke slice {chosen_path.name} (Siap dilabeli / direvisi)"
            )
        else:
            self._refresh_file_list_status()
            if chosen_path in self.image_files:
                self.file_list_widget.setCurrentRow(self.image_files.index(chosen_path))

    def delete_saved_revision(self) -> None:
        """Menghapus gambar & mask yang sudah tersimpan sebelumnya pada titik/sampel aktif agar dapat direvisi ulang."""
        if not self.image_files or self.current_index < 0:
            return
        src_path = self.image_files[self.current_index]
        pt_tag = self._get_point_tag(src_path)
        reply = QMessageBox.question(
            self,
            "Konfirmasi Hapus & Revisi Ulang",
            f"Hapus gambar & mask yang sudah tersimpan pada titik [{pt_tag}] ({src_path.name}) agar bisa direvisi dari awal?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self.chk_one_sample_per_point.isChecked():
            for _, pt_files in self.point_groups.items():
                if src_path in pt_files:
                    self._delete_point_outputs(pt_files)
                    break
        else:
            self._delete_source_outputs(src_path)

        self.canvas.clear_mask()
        self.is_dirty = False
        self._refresh_file_list_status()
        self._update_revision_ui(False)
        self.status_bar.showMessage(
            f"🗑️ Revisi [{pt_tag}]: Gambar & mask tersimpan sebelumnya berhasil dihapus. Silakan gambar label baru lalu Simpan."
        )

    def switch_sibling_point(self, delta: int) -> None:
        """Berpindah ke folder titik saudara berikutnya/sebelumnya di folder induk dataset."""
        if not self.sibling_points or self.input_dir is None:
            return
        if self.chk_autosave.isChecked() and self.is_dirty and self._has_any_mask_stroke():
            self.save_current()
        try:
            curr_idx = self.sibling_points.index(self.input_dir)
        except ValueError:
            curr_idx = 0
        next_idx = (curr_idx + delta) % len(self.sibling_points)
        target_dir = self.sibling_points[next_idx]
        if self.output_dir is not None and self._clean_point_tag(self.output_dir.name) == self._clean_point_tag(self.input_dir.name):
            self.output_dir = self.output_dir.parent / self._clean_point_tag(target_dir.name)
            self.lbl_output_dir.setText(f"Output: {self.output_dir}")
            self.lbl_output_dir.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        self._set_input_directory(target_dir)
        self.status_bar.showMessage(
            f"📍 Pindah Titik ({next_idx + 1}/{len(self.sibling_points)}): {target_dir.name}"
        )

    def load_parent_all_points(self) -> None:
        """Naik ke folder induk dan muat seluruh titik sekaligus (1 sampel per titik)."""
        if self.input_dir is None:
            return
        if self.chk_autosave.isChecked() and self.is_dirty and self._has_any_mask_stroke():
            self.save_current()
        parent_dir = self.input_dir.parent
        if not parent_dir.exists() or parent_dir == self.input_dir:
            return
        self.chk_one_sample_per_point.blockSignals(True)
        self.chk_one_sample_per_point.setChecked(True)
        self.chk_one_sample_per_point.blockSignals(False)
        if self.output_dir is not None and self._clean_point_tag(self.output_dir.name) == self._clean_point_tag(self.input_dir.name):
            self.output_dir = self.output_dir.parent
            self.lbl_output_dir.setText(f"Output: {self.output_dir}")
            self.lbl_output_dir.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        self._set_input_directory(parent_dir)
        self.status_bar.showMessage(
            f"📂 Mode 1 Sampel/Titik Aktif: Memuat {len(self.point_groups)} titik dari {parent_dir.name}"
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = OCTLabelingApp()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
