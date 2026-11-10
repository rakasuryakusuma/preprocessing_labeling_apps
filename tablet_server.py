import base64
import json
import os
import random
import socket
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

CROP_SIZE = 512
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
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BROWSE_DIR = PROJECT_DIR.parent if PROJECT_DIR.parent.exists() else PROJECT_DIR
CONFIG_FILE = PROJECT_DIR / ".tablet_server_config.json"


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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        roi_strip = gray[5 : max(6, h - 5), start_x : start_x + crop_size]
        row_means = np.mean(roi_strip, axis=1)
        peak_y = int(np.argmax(row_means)) + 5
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


def render_reference_overlay(roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Membuat gambar visualisasi hasil akhir seperti gambar referensi:
    - Garis Merah: top SC (Class 1)
    - Garis Kuning: bot SC (Class 2)
    - Garis Hijau: end ED (Class 3)
    Dilengkapi kotak legenda di pojok kanan bawah.
    """
    vis = roi_bgr.copy()
    vis[mask == 1] = (45, 45, 255)    # Merah (top SC)
    vis[mask == 2] = (0, 230, 255)    # Kuning (bot SC)
    vis[mask == 3] = (64, 230, 0)     # Hijau (end ED)

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


class ServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.subfolder_mode: bool = True
        self.one_sample_per_point: bool = True
        self.image_files: List[Path] = []
        self.point_groups: Dict[str, List[Path]] = {}
        self.selected_point_sample: Dict[str, Path] = {}
        self.rejected_point_samples: Dict[str, set] = {}
        self.sibling_points: List[Path] = []

        # Cache gambar mentah terakhir yang dibuka agar geser crop sangat cepat
        self.cached_index: int = -1
        self.cached_raw_img: Optional[np.ndarray] = None

        self.load_config()

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
            out_name_clean = self._clean_point_tag(self.output_dir.name)
            if out_name_clean.lower() == ptag.lower():
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

    def load_config(self) -> None:
        if not CONFIG_FILE.exists():
            return
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            in_dir = data.get("input_dir")
            out_dir = data.get("output_dir")
            self.subfolder_mode = bool(data.get("subfolder_mode", True))
            self.one_sample_per_point = bool(data.get("one_sample_per_point", True))
            if out_dir:
                self.output_dir = Path(out_dir)
            if in_dir and Path(in_dir).exists():
                self.input_dir = Path(in_dir)
                self._scan_files()
        except Exception:
            pass

    def save_config(self) -> None:
        try:
            data = {
                "input_dir": str(self.input_dir) if self.input_dir else None,
                "output_dir": str(self.output_dir) if self.output_dir else None,
                "subfolder_mode": self.subfolder_mode,
                "one_sample_per_point": self.one_sample_per_point,
            }
            CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def set_directories(
        self,
        input_dir: Optional[str],
        output_dir: Optional[str],
        subfolder_mode: Optional[bool] = None,
        one_sample_per_point: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        with self.lock:
            if subfolder_mode is not None:
                self.subfolder_mode = subfolder_mode
            if one_sample_per_point is not None:
                self.one_sample_per_point = one_sample_per_point

            if input_dir:
                p_in = Path(input_dir)
                if not p_in.exists() or not p_in.is_dir():
                    return False, f"Folder sumber tidak ditemukan: {input_dir}"
                old_in = self.input_dir
                self.input_dir = p_in
                if not output_dir:
                    if self.output_dir is None:
                        self.output_dir = p_in.parent / f"{self._clean_point_tag(p_in.name)}_output"
                    elif old_in is not None:
                        old_tag = self._clean_point_tag(old_in.name)
                        new_tag = self._clean_point_tag(p_in.name)
                        if self.output_dir.name.lower() == old_tag.lower():
                            self.output_dir = self.output_dir.parent / new_tag
                        elif self.output_dir.name.lower() == f"{old_in.name.lower()}_output":
                            self.output_dir = self.output_dir.parent / f"{p_in.name}_output"

            if output_dir:
                self.output_dir = Path(output_dir)

            self.cached_index = -1
            self.cached_raw_img = None
            self._scan_files()
            self.save_config()
            return True, f"Memuat {len(self.image_files)} gambar ({len(self.point_groups)} titik)."

    @staticmethod
    def _list_images_in_dir(d: Path) -> List[Path]:
        try:
            return [
                p
                for p in sorted(d.iterdir())
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTENSIONS
                and not p.stem.startswith("000_TRANSIT_OCT_")
            ]
        except Exception:
            return []

    def _scan_files(self) -> None:
        if self.input_dir is None or not self.input_dir.exists():
            self.image_files = []
            self.point_groups = {}
            self.sibling_points = []
            return

        direct_files = self._list_images_in_dir(self.input_dir)
        point_groups: Dict[str, List[Path]] = {}

        if direct_files:
            point_groups[self.input_dir.name] = direct_files
            # Deteksi folder titik saudara (sibling titik) di parent folder
            parent = self.input_dir.parent
            siblings = []
            if parent.exists() and parent != self.input_dir:
                try:
                    for sub in sorted(parent.iterdir(), key=lambda x: x.name.lower()):
                        if (
                            sub.is_dir()
                            and not sub.name.startswith(".")
                            and sub.name.lower() not in RESERVED_DIR_NAMES
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
                        or sub.name.lower() in RESERVED_DIR_NAMES
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
                                or sub2.name.lower() in RESERVED_DIR_NAMES
                                or sub2.name.endswith("_output")
                            ):
                                continue
                            sub2_files = self._list_images_in_dir(sub2)
                            if sub2_files:
                                point_groups[f"{sub.name}/{sub2.name}"] = sub2_files
            except Exception:
                pass

        self.point_groups = point_groups

        # Jika mode 1 Sampel per Titik aktif (baik 1 folder titik maupun multi-folder titik):
        # Ambil tepat 1 sampel per titik (prioritaskan yang sudah dilabeli, jika belum ambil 1 acak)
        if len(point_groups) >= 1 and self.one_sample_per_point:
            sampled_list: List[Path] = []
            for pt_key, pt_files in point_groups.items():
                labeled_in_pt = [p for p in pt_files if self.is_file_labeled(p)]
                if labeled_in_pt:
                    chosen = labeled_in_pt[0]
                    # Pastikan hanya 1 sampel yang tersimpan per titik (hapus sisa sampel lama di titik yg sama bila ada)
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
                    labeled_in_pt = [p for p in pt_files if self.is_file_labeled(p)]
                    self.selected_point_sample[pt_key] = (
                        labeled_in_pt[0] if labeled_in_pt else random.choice(pt_files)
                    )
            self.image_files = all_files

    def is_file_labeled(self, src_path: Path) -> bool:
        return self._find_mask_for_source(src_path) is not None

    def get_file_display_name(self, src_path: Path) -> str:
        ptag = self._get_point_tag(src_path)
        if self.one_sample_per_point:
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

    def reroll_point_sample(self, current_index: int) -> Tuple[bool, str, int]:
        """
        Mengambil 1 sampel acak lain untuk titik yang sedang dibuka karena slice saat ini
        strukturnya kurang bagus:
        - Menghapus hasil simpan lama pada titik tersebut (agar TIDAK terhitung selesai/finish
          dan menjamin hanya ada maksimal 1 sampel per titik).
        - Mengganti sampel titik aktif dengan slice acak lain yang belum ditolak.
        """
        with self.lock:
            if not self.image_files:
                return False, "Belum ada gambar yang dimuat.", -1

            idx = max(0, min(len(self.image_files) - 1, current_index))
            cur_path = self.image_files[idx]

            # Cari titik milik cur_path
            target_pt_key = None
            for pt_key, pt_files in self.point_groups.items():
                if cur_path in pt_files:
                    target_pt_key = pt_key
                    break

            if target_pt_key is not None:
                pt_files = self.point_groups[target_pt_key]
                # Hapus semua output tersimpan pada titik ini agar statusnya kembali BELUM SELESAI (0/1)
                self._delete_point_outputs(pt_files)

                # Catat cur_path sebagai slice yang ditolak agar tidak terpilih berulang sebelum semua slice dicoba
                rej = self.rejected_point_samples.setdefault(target_pt_key, set())
                rej.add(cur_path)
                unseen_candidates = [p for p in pt_files if p not in rej]
                if not unseen_candidates:
                    rej.clear()
                    rej.add(cur_path)
                    unseen_candidates = [p for p in pt_files if p != cur_path] or pt_files

                new_choice = random.choice(unseen_candidates)
                self.selected_point_sample[target_pt_key] = new_choice
                self.cached_index = -1
                self.cached_raw_img = None
                ptag = self._get_point_tag(new_choice)

                if self.one_sample_per_point:
                    self.image_files[idx] = new_choice
                    return (
                        True,
                        f"🎲 Titik [{ptag}]: Diganti ke {new_choice.name} (Slice lama dibatalkan — status: Belum Selesai)",
                        idx,
                    )
                else:
                    chosen_idx = (
                        self.image_files.index(new_choice)
                        if new_choice in self.image_files
                        else idx
                    )
                    return (
                        True,
                        f"🎲 Titik [{ptag}]: Melompat ke {new_choice.name} (Status direset: Belum Selesai)",
                        chosen_idx,
                    )

            return False, "Titik tidak ditemukan.", idx

    def switch_sibling_point(self, delta: int) -> Tuple[bool, str]:
        """Berpindah ke folder titik sebelumnya/berikutnya di dalam folder dataset."""
        with self.lock:
            if not self.sibling_points or self.input_dir is None:
                return False, "Tidak ditemukan folder titik lain di direktori induk."
            try:
                cur_pos = self.sibling_points.index(self.input_dir)
            except ValueError:
                cur_pos = 0
            new_pos = (cur_pos + delta) % len(self.sibling_points)
            target_dir = self.sibling_points[new_pos]

        return self.set_directories(str(target_dir), None)

    def load_parent_all_points(self) -> Tuple[bool, str]:
        """Naik 1 level ke folder dataset induk untuk memuat semua titik sekaligus (1 sampel/titik)."""
        with self.lock:
            if self.input_dir is None or self.input_dir.parent == self.input_dir:
                return False, "Folder induk tidak tersedia."
            parent_dir = self.input_dir.parent
            old_tag = self._clean_point_tag(self.input_dir.name)
            new_out = None
            if (
                self.output_dir is not None
                and self._clean_point_tag(self.output_dir.name).lower() == old_tag.lower()
            ):
                new_out = str(self.output_dir.parent)

        return self.set_directories(
            str(parent_dir), new_out, one_sample_per_point=True
        )

    def get_status_dict(self) -> Dict:
        with self.lock:
            files_info = []
            labeled_count = 0
            first_labeled_idx = -1
            for idx, p in enumerate(self.image_files):
                labeled = self.is_file_labeled(p)
                if labeled:
                    labeled_count += 1
                    if first_labeled_idx == -1:
                        first_labeled_idx = idx
                files_info.append(
                    {
                        "index": idx,
                        "name": self.get_file_display_name(p),
                        "raw_name": p.name,
                        "point_tag": self._get_point_tag(p),
                        "labeled": labeled,
                    }
                )

            # Hitung progres berbasis Titik (1 titik cukup 1 sampel)
            total_points = len(self.point_groups)
            labeled_points = 0
            for pt_key, pt_files in self.point_groups.items():
                if self.one_sample_per_point:
                    chosen_p = self.selected_point_sample.get(pt_key)
                    if chosen_p is not None and self.is_file_labeled(chosen_p):
                        labeled_points += 1
                else:
                    if any(self.is_file_labeled(p) for p in pt_files):
                        labeled_points += 1

            # Jika sedang di Single-Titik Mode tapi ada sibling_points, hitung juga progres sibling titik
            sibling_total = len(self.sibling_points)
            sibling_Idx = -1
            if sibling_total > 0 and self.input_dir in self.sibling_points:
                sibling_Idx = self.sibling_points.index(self.input_dir)

            # Tentukan indeks awal yang direkomendasikan (jika belum ada yg dilabeli di titik tunggal, pilih 1 sampel acak)
            recommended_index = 0
            if self.image_files:
                if len(self.point_groups) == 1:
                    if first_labeled_idx >= 0:
                        recommended_index = first_labeled_idx
                    else:
                        pt_key = next(iter(self.point_groups.keys()))
                        chosen_p = self.selected_point_sample.get(pt_key)
                        if chosen_p in self.image_files:
                            recommended_index = self.image_files.index(chosen_p)
                else:
                    # Multi-titik: arahkan ke titik pertama yang belum dilabeli
                    unlabeled_indices = [
                        i for i, f in enumerate(files_info) if not f["labeled"]
                    ]
                    recommended_index = unlabeled_indices[0] if unlabeled_indices else 0

            return {
                "input_dir": str(self.input_dir) if self.input_dir else "",
                "output_dir": str(self.output_dir) if self.output_dir else "",
                "default_browse_dir": str(DEFAULT_BROWSE_DIR),
                "subfolder_mode": self.subfolder_mode,
                "one_sample_per_point": self.one_sample_per_point,
                "multi_point_mode": len(self.point_groups) > 1,
                "total_points": total_points,
                "labeled_points": labeled_points,
                "sibling_total": sibling_total,
                "sibling_index": sibling_Idx,
                "current_point_name": self.input_dir.name if self.input_dir else "",
                "recommended_index": recommended_index,
                "total_files": len(self.image_files),
                "labeled_count": labeled_count,
                "files": files_info,
            }

    def get_raw_image(self, index: int) -> Optional[np.ndarray]:
        with self.lock:
            if index < 0 or index >= len(self.image_files):
                return None
            if self.cached_index == index and self.cached_raw_img is not None:
                return self.cached_raw_img
            src_path = self.image_files[index]
            raw = imread_unicode(str(src_path), cv2.IMREAD_COLOR)
            if raw is None:
                return None
            raw = ensure_min_size(raw, CROP_SIZE)
            self.cached_index = index
            self.cached_raw_img = raw
            return raw

    def get_image_payload(
        self,
        index: int,
        mode: str = "top_center",
        custom_x: int = 0,
        custom_y: int = 0,
        include_mask: bool = True,
    ) -> Optional[Dict]:
        raw_img = self.get_raw_image(index)
        if raw_img is None:
            return None

        with self.lock:
            src_path = self.image_files[index]
            raw_h, raw_w = raw_img.shape[:2]

            sx, sy, ex, ey = compute_crop_coords(
                raw_img,
                mode=mode,
                custom_x=custom_x,
                custom_y=custom_y,
                crop_size=CROP_SIZE,
            )
            cropped_bgr = raw_img[sy:ey, sx:ex].copy()

            # Encode cropped ROI sebagai PNG base64
            _, roi_png = cv2.imencode(".png", cropped_bgr)
            roi_b64 = base64.b64encode(roi_png.tobytes()).decode("ascii")

            # Buat thumbnail gambar mentah untuk preview posisi ROI
            max_thumb = 240
            scale = min(max_thumb / float(raw_w), max_thumb / float(raw_h), 1.0)
            tw = max(1, int(round(raw_w * scale)))
            th = max(1, int(round(raw_h * scale)))
            thumb_bgr = cv2.resize(raw_img, (tw, th), interpolation=cv2.INTER_AREA)
            _, thumb_jpg = cv2.imencode(
                ".jpg", thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            thumb_b64 = base64.b64encode(thumb_jpg.tobytes()).decode("ascii")

            # Muat mask yang sudah ada (jika diminta)
            mask_b64 = None
            if include_mask and self.output_dir is not None:
                candidate = self._find_mask_for_source(src_path)
                if candidate is not None and candidate.exists():
                    loaded_mask = imread_unicode(str(candidate), cv2.IMREAD_GRAYSCALE)
                    if (
                        loaded_mask is not None
                        and loaded_mask.shape == (CROP_SIZE, CROP_SIZE)
                    ):
                        clean_m = np.clip(loaded_mask, 0, 3).astype(np.uint8)
                        mask_b64 = base64.b64encode(clean_m.tobytes()).decode("ascii")

            frame_segments = []
            meta_path = src_path.with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                    frame_segments = meta_obj.get("segments", [])
                except Exception:
                    pass

            return {
                "index": index,
                "name": self.get_file_display_name(src_path),
                "raw_name": src_path.name,
                "point_tag": self._get_point_tag(src_path),
                "raw_w": raw_w,
                "raw_h": raw_h,
                "sx": sx,
                "sy": sy,
                "ex": ex,
                "ey": ey,
                "max_x": max(0, raw_w - CROP_SIZE),
                "max_y": max(0, raw_h - CROP_SIZE),
                "roi_png_b64": roi_b64,
                "thumb_jpg_b64": thumb_b64,
                "mask_raw_b64": mask_b64,
                "labeled": self.is_file_labeled(src_path),
                "frame_segments": frame_segments,
            }

    def save_crop_and_mask(
        self,
        index: int,
        mode: str,
        custom_x: int,
        custom_y: int,
        mask_raw_b64: str,
        subfolder_mode: bool,
        slice_x: int = 256,
        strip_width: int = 8,
    ) -> Tuple[bool, str, Dict]:
        raw_img = self.get_raw_image(index)
        if raw_img is None:
            return False, "Gambar sumber tidak ditemukan.", {}

        with self.lock:
            self.subfolder_mode = subfolder_mode
            if self.output_dir is None:
                if self.input_dir is not None:
                    self.output_dir = (
                        self.input_dir.parent / f"{self._clean_point_tag(self.input_dir.name)}_output"
                    )
                else:
                    return False, "Folder Output belum diatur.", {}

            self.output_dir.mkdir(parents=True, exist_ok=True)
            src_path = self.image_files[index]
            save_stem = self._get_save_stem(src_path)

            # Jika Mode 1 Sampel per Titik aktif, hapus output slice lain pada titik yang sama agar dijamin hanya 1 sampel/titik
            if self.one_sample_per_point:
                for pt_key, pt_files in self.point_groups.items():
                    if src_path in pt_files:
                        self.selected_point_sample[pt_key] = src_path
                        self._delete_point_outputs(pt_files, except_src=src_path)
                        break

            sx, sy, ex, ey = compute_crop_coords(
                raw_img,
                mode=mode,
                custom_x=custom_x,
                custom_y=custom_y,
                crop_size=CROP_SIZE,
            )
            cropped_bgr = raw_img[sy:ey, sx:ex].copy()

            raw_bytes = base64.b64decode(mask_raw_b64)
            if len(raw_bytes) != CROP_SIZE * CROP_SIZE:
                return (
                    False,
                    f"Ukuran buffer mask tidak valid ({len(raw_bytes)} byte).",
                    {},
                )

            mask_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                (CROP_SIZE, CROP_SIZE)
            )
            clean_mask = np.clip(mask_arr, 0, 3).astype(np.uint8)
            overlay_vis = render_reference_overlay(cropped_bgr, clean_mask)

            crop_path = self.output_dir / f"{save_stem}_crop.png"
            mask_path = self.output_dir / f"{save_stem}_mask.png"

            ok_crop = imwrite_unicode(str(crop_path), cropped_bgr)
            ok_mask = imwrite_unicode(str(mask_path), clean_mask)

            ovl_dir = self.output_dir / "overlays"
            ovl_dir.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(str(ovl_dir / f"{save_stem}_overlay.png"), overlay_vis)

            # Simpan potongan kecil vertikal (Area yang diambil) pada koordinat X yang terkunci
            sw = max(1, min(64, int(strip_width)))
            cx_local = max(0, min(CROP_SIZE - 1, int(slice_x)))
            x0 = max(0, min(CROP_SIZE - sw, cx_local - sw // 2))
            x1 = x0 + sw
            strip_bgr = cropped_bgr[:, x0:x1].copy()
            strip_msk = clean_mask[:, x0:x1].copy()

            strip_dir = self.output_dir / "strips"
            strip_msk_dir = self.output_dir / "strips_mask"
            strip_dir.mkdir(parents=True, exist_ok=True)
            strip_msk_dir.mkdir(parents=True, exist_ok=True)

            # Cek apakah gambar ini adalah hasil gabungan (memiliki .meta.json)
            meta_path = src_path.with_suffix(".meta.json")
            split_count = 0
            if meta_path.exists():
                try:
                    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                    segments = meta_obj.get("segments", [])
                    csv_lines = ["frame_index,source_file,x_start,x_end,top_SC_y,bot_SC_y,end_ED_y,tebal_SC_px,tebal_ED_px"]
                    for seg_idx, seg in enumerate(segments):
                        sx0 = int(seg["x0"])
                        sx1 = int(seg["x1"])
                        orig_stem = Path(seg["name"]).stem
                        s_bgr = cropped_bgr[:, sx0:sx1].copy()
                        s_msk = clean_mask[:, sx0:sx1].copy()
                        imwrite_unicode(str(strip_dir / f"{orig_stem}_strip.png"), s_bgr)
                        imwrite_unicode(str(strip_msk_dir / f"{orig_stem}_strip_mask.png"), s_msk)
                        split_count += 1

                        # Hitung rata-rata Y untuk frame ini
                        y_ys = {}
                        for cid in (1, 2, 3):
                            ys, _ = np.where(s_msk == cid)
                            y_ys[cid] = float(np.mean(ys)) if ys.size > 0 else None
                        sc_t = (y_ys[2] - y_ys[1]) if (y_ys[1] is not None and y_ys[2] is not None) else None
                        ed_t = (y_ys[3] - y_ys[2]) if (y_ys[2] is not None and y_ys[3] is not None) else None
                        csv_lines.append(
                            f"{seg_idx + 1},{seg['name']},{sx0},{sx1},"
                            f"{f'{y_ys[1]:.2f}' if y_ys[1] is not None else ''},"
                            f"{f'{y_ys[2]:.2f}' if y_ys[2] is not None else ''},"
                            f"{f'{y_ys[3]:.2f}' if y_ys[3] is not None else ''},"
                            f"{f'{sc_t:.2f}' if sc_t is not None else ''},"
                            f"{f'{ed_t:.2f}' if ed_t is not None else ''}"
                        )
                    csv_path = self.output_dir / f"{save_stem}_thickness.csv"
                    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
                except Exception:
                    pass
            else:
                imwrite_unicode(str(strip_dir / f"{save_stem}_strip.png"), strip_bgr)
                imwrite_unicode(str(strip_msk_dir / f"{save_stem}_strip_mask.png"), strip_msk)

            if self.subfolder_mode:
                img_dir = self.output_dir / "images"
                msk_dir = self.output_dir / "masks"
                img_dir.mkdir(parents=True, exist_ok=True)
                msk_dir.mkdir(parents=True, exist_ok=True)
                imwrite_unicode(str(img_dir / f"{save_stem}_crop.png"), cropped_bgr)
                imwrite_unicode(str(msk_dir / f"{save_stem}_mask.png"), clean_mask)

            self.save_config()

            if not (ok_crop and ok_mask):
                return False, f"Gagal menyimpan ke {self.output_dir}", {}

            unique_vals = np.unique(clean_mask).tolist()
            c1 = int(np.sum(clean_mask == 1))
            c2 = int(np.sum(clean_mask == 2))
            c3 = int(np.sum(clean_mask == 3))
            labeled_count = sum(1 for p in self.image_files if self.is_file_labeled(p))
            total_points = len(self.point_groups)
            labeled_points = 0
            for pt_key, pt_files in self.point_groups.items():
                if self.one_sample_per_point:
                    chosen_p = self.selected_point_sample.get(pt_key)
                    if chosen_p is not None and self.is_file_labeled(chosen_p):
                        labeled_points += 1
                else:
                    if any(self.is_file_labeled(p) for p in pt_files):
                        labeled_points += 1

            ptag = self._get_point_tag(src_path)
            return (
                True,
                f"Tersimpan [{ptag}]: {save_stem}_crop.png & {save_stem}_mask.png {unique_vals} (Titik selesai: {labeled_points}/{total_points})",
                {
                    "unique_vals": unique_vals,
                    "c1": c1,
                    "c2": c2,
                    "c3": c3,
                    "labeled_count": labeled_count,
                    "total_files": len(self.image_files),
                    "labeled_points": labeled_points,
                    "total_points": total_points,
                },
            )

    def stitch_all_frames(
        self,
        mode: str = "top_center",
        custom_x: int = 0,
        custom_y: int = 0,
        slice_x: int = 256,
        frames_per_group: int = 10,
    ) -> Tuple[bool, str, int]:
        """
        Menggabungkan frame di folder sumber menjadi gambar komposit Transit OCT (512x512).
        - Jika frames_per_group > 0 (default 10): setiap 10 frame (1 subject & 1 posisi)
          diambil irisan tengahnya secara proporsional (512 // N piksel tepat di tengah X=256)
          lalu digabungkan menjadi 1 gambar 512x512 beresolusi asli (tanpa stretch/blur!).
        - Jika frames_per_group == 0: menggabungkan seluruh gambar di folder menjadi 1 gambar.
        """
        with self.lock:
            if self.input_dir is None or not self.image_files:
                return False, "Pilih Folder Sumber terlebih dahulu.", -1

            if self.output_dir is None:
                self.output_dir = (
                    self.input_dir.parent / f"{self.input_dir.name}_output"
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)

            source_files = [
                p
                for p in self.image_files
                if not p.stem.startswith("000_TRANSIT_OCT_")
            ]
            if not source_files:
                return False, "Tidak ada file frame sumber untuk digabungkan.", -1

            # Bagi menjadi kelompok sesuai frames_per_group (misal per 10 frame)
            if frames_per_group > 0 and len(source_files) > frames_per_group:
                groups = [
                    source_files[i : i + frames_per_group]
                    for i in range(0, len(source_files), frames_per_group)
                ]
            else:
                groups = [source_files]

            cx_local = max(0, min(CROP_SIZE - 1, int(slice_x)))
            stitch_dir = self.output_dir / "stitched"
            stitch_dir.mkdir(parents=True, exist_ok=True)
            folder_name = self.input_dir.name
            created_files = []

            for g_idx, group_paths in enumerate(groups):
                k_frames = len(group_paths)
                if k_frames == 0:
                    continue

                # Hitung lebar strip proporsional per frame agar total pas 512 px
                # Contoh: 10 frame -> 8 frame @ 51 px + 2 frame @ 52 px = 512 px (resolusi asli 1:1!)
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
                        raw,
                        mode=mode,
                        custom_x=custom_x,
                        custom_y=custom_y,
                        crop_size=CROP_SIZE,
                    )
                    roi = raw[sy:ey, sx:ex]
                    w_this = base_w + (1 if f_idx < rem else 0)
                    w_this = max(1, min(CROP_SIZE, w_this))
                    x0 = max(0, min(CROP_SIZE - w_this, cx_local - w_this // 2))
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
                                "src_x0": int(sx + x0),
                                "src_x1": int(sx + x1),
                            }
                        )
                        cur_x += w_this

                if not strips:
                    continue

                stitched = np.concatenate(strips, axis=1)
                if stitched.shape[1] != CROP_SIZE:
                    stitched_512 = cv2.resize(
                        stitched, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    stitched_512 = stitched

                group_tag = (
                    f"{folder_name}_G{g_idx + 1:02d}_N{len(strips)}"
                    if len(groups) > 1
                    else f"{folder_name}_N{len(strips)}"
                )
                raw_stitch_path = stitch_dir / f"Transit_OCT_{group_tag}.png"
                imwrite_unicode(str(raw_stitch_path), stitched)

                in_stitch_path = self.input_dir / f"000_TRANSIT_OCT_{group_tag}.png"
                imwrite_unicode(str(in_stitch_path), stitched_512)

                meta_path = in_stitch_path.with_suffix(".meta.json")
                meta_path.write_text(
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
                return False, "Gagal membaca frame untuk digabungkan.", -1

            self.cached_index = -1
            self.cached_raw_img = None
            self._scan_files()

            return (
                True,
                f"Berhasil membuat {len(created_files)} gambar gabungan ({created_files[0]}) dari potongan tengah tiap frame!",
                0,
            )


STATE = ServerState()


def list_filesystem_dir(dir_path: Optional[str]) -> Dict:
    """Menjelajahi folder di Laptop langsung dari browser Tablet."""
    if not dir_path:
        drives = []
        if os.name == "nt":
            for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                dp = f"{letter}:\\"
                if os.path.exists(dp):
                    drives.append({"name": f"Drive {letter}:\\", "path": dp})
        else:
            drives.append({"name": "/", "path": "/"})
        return {
            "current_path": "",
            "parent_path": "",
            "entries": drives,
            "image_count": 0,
        }

    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Direktori tidak dapat diakses: {dir_path}")

    parent_path = str(p.parent) if p.parent != p else ""
    entries = []
    img_count = 0
    try:
        for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            try:
                if child.is_dir() and not child.name.startswith("."):
                    entries.append({"name": child.name, "path": str(child)})
                elif child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    img_count += 1
            except PermissionError:
                continue
    except PermissionError:
        pass

    return {
        "current_path": str(p),
        "parent_path": parent_path,
        "entries": entries,
        "image_count": img_count,
    }


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#121418">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<link rel="manifest" href="/manifest.json">
<title>OCT Labeling Tablet (Remote Laptop)</title>
<style>
  :root {
    --bg-main: #121418;
    --bg-panel: #1a1d24;
    --bg-input: #14171d;
    --border: #2c313c;
    --text: #e4e7eb;
    --text-muted: #9aa5b1;
    --accent: #2563eb;
    --green: #2b8a3e;
    --cyan: #00dcff;
    --red: #ff3232;
    --yellow: #facc15;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; user-select: none; }
  body {
    margin: 0; padding: 0;
    background: var(--bg-main); color: var(--text);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 13px;
    height: 100dvh; width: 100vw;
    overflow: hidden;
    display: flex; flex-direction: column;
  }
  /* Top Header */
  header {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    padding: 6px 12px;
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
    flex-shrink: 0;
  }
  .brand { font-weight: 700; color: #7cc4fa; font-size: 14px; display: flex; align-items: center; gap: 8px; }
  .file-badge {
    background: var(--bg-input); border: 1px solid var(--border);
    padding: 4px 10px; border-radius: 6px; font-weight: 600; color: #fff;
    max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .stats-badge { color: #38bdf8; font-weight: 700; font-size: 12px; white-space: nowrap; }

  /* Main 3-Column Layout */
  .workspace {
    flex: 1; display: flex; overflow: hidden; gap: 8px; padding: 8px;
  }
  .panel-left {
    width: 300px; flex-shrink: 0;
    display: flex; flex-direction: column; gap: 8px;
    overflow-y: auto; padding-right: 2px;
  }
  .panel-center {
    flex: 1; display: flex; flex-direction: column; gap: 6px;
    background: #161920; border: 2px solid var(--border); border-radius: 8px;
    overflow: hidden; position: relative;
  }
  .panel-right {
    width: 245px; flex-shrink: 0;
    display: flex; flex-direction: column; gap: 8px;
    overflow: hidden;
  }

  .card {
    background: var(--bg-panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px;
  }
  .card-title {
    font-weight: 700; color: #7cc4fa; font-size: 12px;
    margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;
  }

  button {
    background: #262b36; color: #f0f4f8;
    border: 1px solid #3a4152; border-radius: 6px;
    padding: 7px 10px; font-weight: 600; font-size: 12px;
    cursor: pointer; touch-action: manipulation;
  }
  button:active { transform: scale(0.98); }
  .btn-primary { background: #1d4ed8; border-color: #2563eb; color: #fff; }
  .btn-success {
    background: var(--green); border-color: #37b24d; color: #fff;
    font-size: 13px; font-weight: 700; padding: 11px; width: 100%;
  }
  .btn-danger { background: #2a181b; border-color: #7f1d1d; color: #fca5a5; }
  .row { display: flex; gap: 6px; align-items: center; }
  .row > * { flex: 1; }

  select, input[type="number"], input[type="text"] {
    background: #222630; color: #fff; border: 1px solid #3a4152;
    border-radius: 5px; padding: 6px 8px; font-size: 12px; width: 100%;
  }

  .path-info {
    background: var(--bg-input); border: 1px solid #262b36;
    border-radius: 5px; padding: 6px; font-size: 11px; color: var(--text-muted);
    word-break: break-all; margin: 6px 0; line-height: 1.4;
  }

  /* Mini Raw Preview */
  .crop-box { display: flex; gap: 8px; align-items: center; margin-top: 6px; }
  #rawPreviewCanvas {
    width: 116px; height: 116px; background: #101216;
    border: 1px solid #363d4d; border-radius: 6px;
    touch-action: none; cursor: crosshair; flex-shrink: 0;
  }

  /* Tool Radio Buttons */
  .tool-btn {
    display: flex; align-items: center; gap: 8px;
    width: 100%; text-align: left; padding: 8px 10px; margin-bottom: 5px;
    background: #14171d; border: 2px solid #2c313c; border-radius: 7px;
    font-weight: 700;
  }
  .tool-btn.active-c1 { border-color: #ff2d2d; background: rgba(255, 45, 45, 0.16); color: #ff6b6b; }
  .tool-btn.active-c2 { border-color: #ffe600; background: rgba(255, 230, 0, 0.16); color: #fde047; }
  .tool-btn.active-c3 { border-color: #00e640; background: rgba(0, 230, 64, 0.16); color: #4ade80; }
  .tool-btn.active-er { border-color: #38bdf8; background: rgba(56, 189, 248, 0.16); color: #38bdf8; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .legend-box {
    position: absolute; bottom: 12px; right: 12px;
    background: rgba(10, 12, 16, 0.88); border: 1px solid #4b5563;
    border-radius: 6px; padding: 7px 11px; pointer-events: none;
    display: flex; flex-direction: column; gap: 4px; font-size: 11px; font-weight: 700;
  }
  .legend-item { display: flex; align-items: center; gap: 8px; color: #f3f4f6; }
  .legend-line { width: 24px; height: 3px; border-radius: 2px; display: inline-block; }

  /* Canvas Viewport */
  .canvas-toolbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 10px; background: #1a1d24; border-bottom: 1px solid var(--border);
    gap: 6px; flex-wrap: wrap;
  }
  .canvas-viewport {
    flex: 1; position: relative; overflow: auto;
    display: flex; align-items: center; justify-content: center;
    touch-action: none;
  }
  .canvas-stack {
    position: relative; width: 512px; height: 512px;
    box-shadow: 0 0 20px rgba(0,0,0,0.6);
    transform-origin: center center;
  }
  .canvas-stack canvas {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    image-rendering: pixelated;
  }

  /* File List */
  .progress-wrap {
    background: var(--bg-input); border: 1px solid var(--border);
    border-radius: 5px; height: 18px; position: relative; overflow: hidden; margin: 6px 0;
  }
  .progress-fill { background: var(--green); height: 100%; width: 0%; transition: width 0.2s; }
  .progress-text {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700;
  }
  .file-list {
    flex: 1; overflow-y: auto; background: var(--bg-input);
    border: 1px solid var(--border); border-radius: 6px; padding: 4px;
  }
  .file-item {
    padding: 7px 8px; border-radius: 5px; margin-bottom: 2px;
    cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .file-item.labeled { color: #4ade80; }
  .file-item.active { background: var(--accent); color: #fff; font-weight: 700; }

  /* Status Bar */
  footer {
    background: #161920; border-top: 1px solid #282d37;
    padding: 5px 12px; color: var(--text-muted); font-size: 12px;
    display: flex; justify-content: space-between; align-items: center;
    flex-shrink: 0;
  }

  /* Folder Browser Modal */
  .modal-backdrop {
    position: fixed; inset: 0; background: rgba(0,0,0,0.75);
    display: none; align-items: center; justify-content: center; z-index: 1000;
  }
  .modal {
    background: var(--bg-panel); border: 1px solid var(--border);
    border-radius: 10px; width: min(560px, 94vw); max-height: 84vh;
    display: flex; flex-direction: column; padding: 14px; gap: 10px;
  }
  .folder-list {
    flex: 1; overflow-y: auto; background: var(--bg-input);
    border: 1px solid var(--border); border-radius: 6px; padding: 6px;
    min-height: 240px; max-height: 360px;
  }
  .folder-row {
    padding: 9px 10px; border-bottom: 1px solid #20242d;
    cursor: pointer; display: flex; align-items: center; gap: 8px; font-weight: 600;
  }
  .folder-row:hover, .folder-row:active { background: #252b37; }

  @media (max-width: 960px) {
    .panel-left { width: 255px; }
    .panel-right { width: 205px; }
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <span>OCT Labeler (Server Laptop)</span>
  </div>
  <div class="file-badge" id="lblActiveFile">Belum ada gambar dimuat</div>
  <div class="stats-badge" id="lblMaskStats">top SC: 0 | bot SC: 0 | end ED: 0</div>
</header>

<div class="workspace">
  <!-- KOLOM KIRI -->
  <div class="panel-left">
    <!-- 1. Direktori Laptop -->
    <div class="card">
      <div class="card-title">1. Direktori di Laptop</div>
      <div class="row">
        <button class="btn-primary" onclick="openFolderModal('input')">Pilih Folder Sumber</button>
        <button onclick="openFolderModal('output')">Pilih Folder Output</button>
      </div>
      <div class="path-info">
        <div id="lblInputDir">Sumber: (Belum dipilih)</div>
        <div id="lblOutputDir" style="margin-top:3px;">Output: (Belum dipilih)</div>
      </div>
      <label style="display:flex;align-items:center;gap:6px;font-size:11px;cursor:pointer;">
        <input type="checkbox" id="chkSubfolder" checked>
        Simpan juga ke /images/ &amp; /masks/
      </label>
      <label style="display:flex;align-items:center;gap:6px;margin-top:5px;font-size:11px;cursor:pointer;color:#4ade80;font-weight:700;" title="1 Subjek = 4 Titik, per titik hanya diambil 1 sampel acak">
        <input type="checkbox" id="chkOneSamplePerPoint" checked onchange="onToggleOneSamplePerPoint()">
        🎯 Mode 1 Sampel per Titik (4 Titik/Subjek)
      </label>
      <button id="btnLoadParentPoints" style="display:none;width:100%;margin-top:6px;background:#0284c7;border-color:#38bdf8;color:#fff;" onclick="loadParentAllPoints()">
        📂 Gabung Semua Titik (1 Sampel/Titik)
      </button>
    </div>

    <!-- 2. Pengaturan Crop ROI 512x512 (B-Scan Spatial) -->
    <div class="card">
      <div class="card-title">2. Crop ROI 512×512 (B-Scan Spatial)</div>
      <select id="selCropMode" onchange="onCropModeChange()">
        <option value="top_center">Atas-Tengah (Paten Tengah: Y=0)</option>
        <option value="auto_surface">Auto-Deteksi Permukaan OCT</option>
        <option value="center">Tengah Penuh (Center X &amp; Y)</option>
        <option value="custom">Manual / Custom (Geser X &amp; Y)</option>
      </select>
      <div class="crop-box">
        <canvas id="rawPreviewCanvas" width="116" height="116"></canvas>
        <div style="flex:1;display:flex;flex-direction:column;gap:5px;">
          <div>
            <span style="font-size:11px;color:var(--text-muted);">Start X:</span>
            <input type="number" id="numStartX" value="0" min="0" max="4096" disabled onchange="onCustomXYChange()">
          </div>
          <div>
            <span style="font-size:11px;color:var(--text-muted);">Start Y:</span>
            <input type="number" id="numStartY" value="0" min="0" max="4096" disabled onchange="onCustomXYChange()">
          </div>
          <div id="lblRawDim" style="font-size:11px;color:var(--text-muted);">Asli: -</div>
          <div id="lblCropDim" style="font-size:11px;color:#4ade80;font-weight:700;">ROI: 512×512</div>
        </div>
      </div>
      <label style="display:flex;align-items:center;gap:6px;margin-top:7px;font-size:11px;cursor:pointer;color:#4ade80;font-weight:700;">
        <input type="checkbox" id="chkLockCrop" checked onchange="onLockCropChange()">
        🔒 Kunci Lokasi Tengah ROI (Konsisten Antar-Slice)
      </label>
      <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;margin-top:5px;font-size:11px;">
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;">
          <input type="checkbox" id="chkShowSliceGuide" onchange="renderMaskOverlay()">
          Garis Panduan Tengah (Opsional)
        </label>
        <div style="display:flex;align-items:center;gap:4px;">
          <span>Lebar:</span>
          <input type="number" id="numStripWidth" value="8" min="1" max="64" style="width:52px;padding:3px 5px;" onchange="renderMaskOverlay()">
          <span>px</span>
        </div>
      </div>
    </div>

    <!-- 3. Alat Anotasi Stylus (3 Class Batas Lapisan — Brush 3px) -->
    <div class="card">
      <div class="card-title">3. Garis Batas Lapisan (3 Class — v2)</div>
      <button id="btnClass1" class="tool-btn active-c1" onclick="setActiveClass(1)">
        <span class="dot" style="background:#ff2d2d;"></span>
        Class 1: Top SC - Permukaan Kulit (Merah)
      </button>
      <button id="btnClass2" class="tool-btn" onclick="setActiveClass(2)">
        <span class="dot" style="background:#ffe600;"></span>
        Class 2: Bot SC - Bawah SC (Kuning)
      </button>
      <button id="btnClass3" class="tool-btn" onclick="setActiveClass(3)">
        <span class="dot" style="background:#00e640;"></span>
        Class 3: End ED - Akhir Epidermis (Hijau)
      </button>
      <button id="btnEraser" class="tool-btn" onclick="setActiveClass(0)">
        <span class="dot" style="background:#38bdf8;"></span>
        Eraser / Hapus (Nilai 0)
      </button>

      <div style="margin-top:8px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;">
          <span>Ketebalan Brush (U-Net 3px):</span>
          <strong id="lblBrushVal">3 px</strong>
        </div>
        <input type="range" id="rngBrush" min="1" max="25" value="3" style="width:100%;" oninput="onBrushChange(this.value)">
      </div>

      <div style="margin-top:6px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;">
          <span>Opasitas Overlay:</span>
          <strong id="lblAlphaVal">75%</strong>
        </div>
        <input type="range" id="rngAlpha" min="20" max="100" value="75" style="width:100%;" oninput="onAlphaChange(this.value)">
      </div>

      <label style="display:flex;align-items:center;gap:6px;margin-top:6px;font-size:12px;cursor:pointer;color:#7cc4fa;font-weight:600;">
        <input type="checkbox" id="chkPenOnly" checked>
        Mode Khusus Stylus Pen (Jari = Geser)
      </label>

      <label style="display:flex;align-items:center;gap:6px;margin-top:4px;font-size:12px;cursor:pointer;">
        <input type="checkbox" id="chkShowOverlay" checked onchange="renderMaskOverlay()">
        Tampilkan Live Overlay
      </label>

      <div class="row" style="margin-top:8px;">
        <button onclick="undoStroke()">↩ Undo</button>
        <button class="btn-danger" onclick="clearMask()">🗑 Clear</button>
      </div>
    </div>
  </div>

  <!-- KOLOM TENGAH: KANVAS 512x512 -->
  <div class="panel-center">
    <div class="canvas-toolbar">
      <div style="display:flex;gap:4px;align-items:center;">
        <span style="font-size:11px;color:var(--text-muted);">Zoom:</span>
        <button onclick="setZoom(1)">100%</button>
        <button onclick="setZoom(1.5)">150%</button>
        <button onclick="setZoom(2)">200%</button>
        <button onclick="setZoom(3)">300%</button>
      </div>
      <div style="font-size:11px;color:var(--text-muted);" id="lblPointerMode">
        Stylus = Gambar | Jari = Geser Kanvas
      </div>
    </div>

    <div class="canvas-viewport" id="canvasViewport">
      <div class="canvas-stack" id="canvasStack">
        <canvas id="imgCanvas" width="512" height="512"></canvas>
        <canvas id="maskCanvas" width="512" height="512"></canvas>
      </div>
      <div class="legend-box">
        <div class="legend-item"><span class="legend-line" style="background:#ff2d2d;"></span> top SC (Class 1)</div>
        <div class="legend-item"><span class="legend-line" style="background:#ffe600;"></span> bot SC (Class 2)</div>
        <div class="legend-item"><span class="legend-line" style="background:#00e640;"></span> end ED (Class 3)</div>
      </div>
    </div>
  </div>

  <!-- KOLOM KANAN: DAFTAR FILE & SIMPAN -->
  <div class="panel-right">
    <div class="card" style="flex:1;display:flex;flex-direction:column;overflow:hidden;">
      <div class="card-title">4. Daftar Titik / Sampel</div>
      <div id="lblProgress" style="font-size:11px;color:var(--text-muted);">Target Titik: 0 / 1 sampel</div>
      <div class="progress-wrap">
        <div class="progress-fill" id="progressBar"></div>
        <div class="progress-text" id="progressPct">0%</div>
      </div>
      <button id="btnRandomSlice" class="btn-primary" style="width:100%;margin-bottom:6px;background:#6d28d9;border-color:#7c3aed;" onclick="randomSlice()">
        🎲 Ganti Slice Acak (Titik Ini)
      </button>
      <div id="siblingNavRow" class="row" style="display:none;margin-bottom:6px;">
        <button style="background:#0f766e;border-color:#14b8a6;" onclick="switchSiblingPoint(-1)" title="Pindah ke folder titik sebelumnya">⏮ Titik Sblm</button>
        <button style="background:#0f766e;border-color:#14b8a6;" onclick="switchSiblingPoint(1)" title="Pindah ke folder titik berikutnya">Titik Brkt ⏭</button>
      </div>
      <div class="file-list" id="fileList"></div>
    </div>

    <div class="card">
      <div class="card-title">5. Navigasi &amp; Simpan</div>
      <div class="row" style="margin-bottom:8px;">
        <button onclick="prevImage()">&laquo; Prev</button>
        <button onclick="nextImage()">Next &raquo;</button>
      </div>
      <label style="display:flex;align-items:center;gap:6px;margin-bottom:8px;font-size:12px;cursor:pointer;">
        <input type="checkbox" id="chkAutoSave" checked>
        Auto-Save saat pindah gambar
      </label>
      <button class="btn-success" onclick="saveCurrent(true)">
        💾 SIMPAN KE LAPTOP
      </button>
    </div>
  </div>
</div>

<footer>
  <span id="lblStatus">Terhubung ke Server Laptop. Pilih Folder Sumber untuk memulai.</span>
  <span id="lblCoords">X=-, Y=-</span>
</footer>

<!-- MODAL PILIH FOLDER DI LAPTOP -->
<div class="modal-backdrop" id="folderModal">
  <div class="modal">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong id="modalTitle" style="color:#7cc4fa;font-size:14px;">Pilih Folder di Laptop</strong>
      <button onclick="closeFolderModal()">✕ Tutup</button>
    </div>
    <div class="row">
      <input type="text" id="txtFolderPath" placeholder="Ketik path folder di Laptop (contoh: D:\dataset_oct)">
      <button style="flex:0 0 auto;" onclick="browsePath(document.getElementById('txtFolderPath').value)">Buka</button>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;font-size:12px;color:var(--text-muted);">
      <div style="display:flex;gap:6px;">
        <button onclick="browseParent()">⬅ Naik 1 Folder</button>
        <button onclick="browsePath('')">💻 Semua Drive</button>
        <button onclick="browsePath(defaultBrowseDir)">📂 Folder TEEP</button>
      </div>
      <span id="lblFolderImgCount">0 gambar</span>
    </div>
    <div class="folder-list" id="folderList"></div>
    <div class="row">
      <button onclick="closeFolderModal()">Batal</button>
      <button class="btn-primary" onclick="confirmFolderSelection()">✓ Pilih Folder Ini</button>
    </div>
  </div>
</div>

<script>
const CROP_SIZE = 512;
let files = [];
let currentIndex = -1;
let activeClass = 1;
let brushSize = 3;
let overlayAlpha = 0.75;
let zoomLevel = 1.0;
let isDirty = false;
let defaultBrowseDir = '';
let multiPointMode = false;
let totalPoints = 1;
let labeledPoints = 0;

// Matriks Masker 8-bit murni {0, 1, 2, 3}
let maskData = new Uint8Array(CROP_SIZE * CROP_SIZE);
let undoStack = [];
const MAX_UNDO = 25;

// Info dimensi gambar mentah untuk preview crop
let rawInfo = { w: 512, h: 512, sx: 0, sy: 0, max_x: 0, max_y: 0 };
let rawThumbImg = new Image();
let currentFrameSegments = [];

const imgCanvas = document.getElementById('imgCanvas');
const imgCtx = imgCanvas.getContext('2d');
const maskCanvas = document.getElementById('maskCanvas');
const maskCtx = maskCanvas.getContext('2d');
const rawPreviewCanvas = document.getElementById('rawPreviewCanvas');
const rawPreviewCtx = rawPreviewCanvas.getContext('2d');

function setStatus(msg) {
  document.getElementById('lblStatus').textContent = msg;
}

// =========================================================================
// Inisialisasi & Sinkronisasi Status Server Laptop
// =========================================================================
async function fetchStatus(autoLoadFirst = true) {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    defaultBrowseDir = data.default_browse_dir || '';
    document.getElementById('lblInputDir').textContent = 'Sumber: ' + (data.input_dir || '(Belum dipilih)');
    document.getElementById('lblOutputDir').textContent = 'Output: ' + (data.output_dir || '(Belum dipilih)');
    document.getElementById('chkSubfolder').checked = data.subfolder_mode;
    const chkOne = document.getElementById('chkOneSamplePerPoint');
    if (chkOne && data.one_sample_per_point !== undefined) {
      chkOne.checked = data.one_sample_per_point;
    }

    multiPointMode = !!data.multi_point_mode;
    totalPoints = data.total_points || 1;
    labeledPoints = data.labeled_points || 0;

    const btnParent = document.getElementById('btnLoadParentPoints');
    const sibRow = document.getElementById('siblingNavRow');
    if (!multiPointMode && data.sibling_total > 1) {
      btnParent.style.display = 'block';
      btnParent.textContent = `📂 Gabung Semua Titik (${data.sibling_total} Titik — 1 Sampel/Titik)`;
      sibRow.style.display = 'flex';
    } else {
      btnParent.style.display = 'none';
      sibRow.style.display = 'none';
    }

    files = data.files || [];
    renderFileList(data.labeled_count, data.total_files, labeledPoints, totalPoints, data.current_point_name);
    if (autoLoadFirst && files.length > 0 && currentIndex === -1) {
      const startIdx = (data.recommended_index !== undefined && data.recommended_index >= 0) ? data.recommended_index : 0;
      await loadImage(startIdx, true);
    }
  } catch (e) {
    setStatus('Gagal terhubung ke server laptop: ' + e.message);
  }
}

function renderFileList(labeledCount, totalFiles, lPoints = labeledPoints, tPoints = totalPoints, ptName = '') {
  labeledPoints = lPoints;
  totalPoints = tPoints;
  if (multiPointMode) {
    document.getElementById('lblProgress').textContent = `Selesai: ${lPoints} / ${tPoints} titik (1 sampel/titik)`;
    const pct = tPoints > 0 ? Math.round((lPoints / tPoints) * 100) : 0;
    document.getElementById('progressBar').style.width = pct + '%';
    document.getElementById('progressPct').textContent = `${lPoints}/${tPoints} (${pct}%)`;
  } else {
    const done1 = labeledCount >= 1 ? 1 : 0;
    const pct = done1 * 100;
    document.getElementById('lblProgress').textContent =
      done1 ? `✓ Target Titik Tercapai: 1 / 1 sampel selesai` : `Target Titik Ini: 0 / 1 sampel (Pilih 1 slice saja)`;
    document.getElementById('progressBar').style.width = pct + '%';
    document.getElementById('progressPct').textContent = `${done1}/1 sampel (${pct}%)`;
  }

  const container = document.getElementById('fileList');
  container.innerHTML = '';
  files.forEach((f) => {
    const div = document.createElement('div');
    div.className = 'file-item' + (f.labeled ? ' labeled' : '') + (f.index === currentIndex ? ' active' : '');
    div.textContent = (f.labeled ? '[✓] ' : '[ ] ') + f.name;
    div.onclick = () => selectFileIndex(f.index);
    container.appendChild(div);
  });
}

function hasAnyMaskStroke() {
  for (let i = 0; i < maskData.length; i++) {
    if (maskData[i] > 0) return true;
  }
  return false;
}

async function selectFileIndex(idx) {
  if (idx === currentIndex || idx < 0 || idx >= files.length) return;
  if (document.getElementById('chkAutoSave').checked && isDirty && hasAnyMaskStroke()) {
    await saveCurrent(false);
  }
  await loadImage(idx, true);
}

async function loadImage(index, loadExistingMask = true) {
  if (index < 0 || index >= files.length) return;
  const mode = document.getElementById('selCropMode').value;
  const cx = parseInt(document.getElementById('numStartX').value || '0', 10);
  const cy = parseInt(document.getElementById('numStartY').value || '0', 10);

  setStatus(`Memuat gambar [${index + 1}/${files.length}] dari Laptop...`);
  const url = `/api/image?index=${index}&mode=${encodeURIComponent(mode)}&x=${cx}&y=${cy}&mask=${loadExistingMask ? 1 : 0}`;
  const res = await fetch(url);
  if (!res.ok) {
    setStatus('Gagal memuat gambar dari Laptop.');
    return;
  }
  const data = await res.json();
  currentIndex = index;
  currentFrameSegments = data.frame_segments || [];

  rawInfo = {
    w: data.raw_w, h: data.raw_h,
    sx: data.sx, sy: data.sy,
    max_x: data.max_x, max_y: data.max_y
  };

  const numX = document.getElementById('numStartX');
  const numY = document.getElementById('numStartY');
  numX.max = data.max_x;
  numY.max = data.max_y;
  numX.value = data.sx;
  numY.value = data.sy;

  document.getElementById('lblActiveFile').textContent = `[${index + 1}/${files.length}] ${data.name}`;
  document.getElementById('lblRawDim').textContent = `Asli: ${data.raw_w}×${data.raw_h}`;
  document.getElementById('lblCropDim').textContent = `ROI: X[${data.sx}:${data.ex}] Y[${data.sy}:${data.ey}]`;

  // Gambar ROI 512x512 ke imgCanvas
  const roiImg = new Image();
  roiImg.onload = () => {
    imgCtx.clearRect(0, 0, CROP_SIZE, CROP_SIZE);
    imgCtx.drawImage(roiImg, 0, 0, CROP_SIZE, CROP_SIZE);
  };
  roiImg.src = 'data:image/png;base64,' + data.roi_png_b64;

  // Gambar thumbnail ke RawPreview
  rawThumbImg.onload = () => drawRawPreview();
  rawThumbImg.src = 'data:image/jpeg;base64,' + data.thumb_jpg_b64;

  if (loadExistingMask) {
    undoStack = [];
    maskData.fill(0);
    if (data.mask_raw_b64) {
      const binStr = atob(data.mask_raw_b64);
      for (let i = 0; i < maskData.length && i < binStr.length; i++) {
        maskData[i] = binStr.charCodeAt(i);
      }
    }
    isDirty = false;
    renderMaskOverlay();
    updateMaskStats();
  }

  // Highlight item aktif di daftar
  const items = document.querySelectorAll('.file-item');
  items.forEach((el, i) => {
    el.classList.toggle('active', i === currentIndex);
    if (i === currentIndex) el.scrollIntoView({ block: 'nearest' });
  });
  setStatus(`Siap melabeli: ${data.name}`);
}

function drawRawPreview() {
  const cw = rawPreviewCanvas.width;
  const ch = rawPreviewCanvas.height;
  rawPreviewCtx.fillStyle = '#101216';
  rawPreviewCtx.fillRect(0, 0, cw, ch);
  if (!rawThumbImg.complete || !rawThumbImg.naturalWidth) return;

  const scale = Math.min((cw - 6) / rawInfo.w, (ch - 6) / rawInfo.h);
  const dw = rawInfo.w * scale;
  const dh = rawInfo.h * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;

  rawPreviewCtx.drawImage(rawThumbImg, ox, oy, dw, dh);
  rawPreviewCtx.strokeStyle = '#00ff64';
  rawPreviewCtx.lineWidth = 2;
  rawPreviewCtx.strokeRect(
    ox + rawInfo.sx * scale,
    oy + rawInfo.sy * scale,
    CROP_SIZE * scale,
    CROP_SIZE * scale
  );
}

// Klik / geser pada kotak preview gambar asli untuk memindahkan ROI
let draggingPreview = false;
function handlePreviewPointer(e) {
  if (currentIndex < 0) return;
  if (document.getElementById('chkLockCrop').checked) {
    setStatus('🔒 Lokasi Crop sedang dikunci! Buka centang "Patenkan / Kunci Lokasi" jika ingin menggeser manual.');
    return;
  }
  const rect = rawPreviewCanvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const cw = rawPreviewCanvas.width;
  const ch = rawPreviewCanvas.height;
  const scale = Math.min((cw - 6) / rawInfo.w, (ch - 6) / rawInfo.h);
  const dw = rawInfo.w * scale;
  const dh = rawInfo.h * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;

  const rx = (px - ox) / scale;
  const ry = (py - oy) / scale;
  const newSx = Math.max(0, Math.min(rawInfo.max_x, Math.round(rx - CROP_SIZE / 2)));
  const newSy = Math.max(0, Math.min(rawInfo.max_y, Math.round(ry - CROP_SIZE / 2)));

  document.getElementById('selCropMode').value = 'custom';
  document.getElementById('numStartX').value = newSx;
  document.getElementById('numStartY').value = newSy;
  rawInfo.sx = newSx;
  rawInfo.sy = newSy;
  drawRawPreview();
}

function onLockCropChange() {
  const locked = document.getElementById('chkLockCrop').checked;
  document.getElementById('numStartX').disabled = locked;
  document.getElementById('numStartY').disabled = locked;
  if (locked) {
    setStatus('🔒 Lokasi Start X & Start Y dikunci agar konsisten di semua frame.');
  } else {
    setStatus('🔓 Lokasi Crop dapat digeser secara manual.');
  }
}

rawPreviewCanvas.addEventListener('pointerdown', (e) => {
  draggingPreview = true;
  rawPreviewCanvas.setPointerCapture(e.pointerId);
  handlePreviewPointer(e);
});
rawPreviewCanvas.addEventListener('pointermove', (e) => {
  if (draggingPreview) handlePreviewPointer(e);
});
rawPreviewCanvas.addEventListener('pointerup', () => {
  if (draggingPreview) {
    draggingPreview = false;
    loadImage(currentIndex, false);
  }
});

function onCropModeChange() {
  if (currentIndex >= 0) loadImage(currentIndex, false);
}

function onCustomXYChange() {
  document.getElementById('selCropMode').value = 'custom';
  if (currentIndex >= 0) loadImage(currentIndex, false);
}

// =========================================================================
// Menggambar Masker Murni {0, 1, 2, 3} dengan Stylus Pen & Palm Rejection
// =========================================================================
const overlayImageData = maskCtx.createImageData(CROP_SIZE, CROP_SIZE);

function renderMaskOverlay() {
  maskCtx.clearRect(0, 0, CROP_SIZE, CROP_SIZE);
  if (!document.getElementById('chkShowOverlay').checked) return;

  const data = overlayImageData.data;
  const alphaByte = Math.round(overlayAlpha * 255);

  for (let i = 0; i < maskData.length; i++) {
    const v = maskData[i];
    const p = i * 4;
    if (v === 1) {
      // Class 1: top SC (Merah)
      data[p] = 255; data[p + 1] = 45; data[p + 2] = 45; data[p + 3] = alphaByte;
    } else if (v === 2) {
      // Class 2: bot SC (Kuning)
      data[p] = 255; data[p + 1] = 230; data[p + 2] = 0; data[p + 3] = alphaByte;
    } else if (v === 3) {
      // Class 3: end ED (Hijau)
      data[p] = 0; data[p + 1] = 230; data[p + 2] = 64; data[p + 3] = alphaByte;
    } else {
      data[p + 3] = 0;
    }
  }
  maskCtx.putImageData(overlayImageData, 0, 0);

  // Tampilkan Garis Pembatas Antar-Frame (F1..F10) jika gambar ini adalah hasil Gabungan,
  // atau Garis Panduan Vertikal "Area yang diambil" (X=256) jika gambar tunggal
  const showGuide = document.getElementById('chkShowSliceGuide');
  if (showGuide && showGuide.checked) {
    maskCtx.save();
    if (currentFrameSegments && currentFrameSegments.length > 1) {
      maskCtx.strokeStyle = 'rgba(124, 196, 250, 0.45)';
      maskCtx.fillStyle = 'rgba(124, 196, 250, 0.85)';
      maskCtx.font = 'bold 10px sans-serif';
      maskCtx.lineWidth = 1;
      maskCtx.setLineDash([4, 4]);
      currentFrameSegments.forEach((seg, idx) => {
        if (idx > 0) {
          maskCtx.beginPath();
          maskCtx.moveTo(seg.x0, 0);
          maskCtx.lineTo(seg.x0, CROP_SIZE);
          maskCtx.stroke();
        }
        maskCtx.fillText('F' + (idx + 1), seg.x0 + 4, 14);
      });
    } else {
      const sw = Math.max(1, Math.min(64, parseInt(document.getElementById('numStripWidth').value || '8', 10)));
      const x0 = Math.max(0, 256 - Math.floor(sw / 2));
      maskCtx.strokeStyle = 'rgba(255, 50, 50, 0.85)';
      maskCtx.lineWidth = 1.5;
      maskCtx.setLineDash([6, 4]);
      maskCtx.strokeRect(x0, 0, sw, CROP_SIZE);
    }
    maskCtx.restore();
  }
}

function updateMaskStats() {
  let c1 = 0, c2 = 0, c3 = 0;
  // Hitung rata-rata Y per kolom X untuk mengukur ketebalan SC (1->2) & ED (2->3)
  const sumY1 = new Float32Array(CROP_SIZE), cnt1 = new Uint16Array(CROP_SIZE);
  const sumY2 = new Float32Array(CROP_SIZE), cnt2 = new Uint16Array(CROP_SIZE);
  const sumY3 = new Float32Array(CROP_SIZE), cnt3 = new Uint16Array(CROP_SIZE);

  for (let y = 0; y < CROP_SIZE; y++) {
    const rowOff = y * CROP_SIZE;
    for (let x = 0; x < CROP_SIZE; x++) {
      const v = maskData[rowOff + x];
      if (v === 1) { c1++; sumY1[x] += y; cnt1[x]++; }
      else if (v === 2) { c2++; sumY2[x] += y; cnt2[x]++; }
      else if (v === 3) { c3++; sumY3[x] += y; cnt3[x]++; }
    }
  }

  let scSum = 0, scCols = 0;
  let edSum = 0, edCols = 0;
  for (let x = 0; x < CROP_SIZE; x++) {
    if (cnt1[x] > 0 && cnt2[x] > 0) {
      scSum += Math.abs((sumY2[x] / cnt2[x]) - (sumY1[x] / cnt1[x]));
      scCols++;
    }
    if (cnt2[x] > 0 && cnt3[x] > 0) {
      edSum += Math.abs((sumY3[x] / cnt3[x]) - (sumY2[x] / cnt2[x]));
      edCols++;
    }
  }
  const scStr = scCols > 0 ? (scSum / scCols).toFixed(1) + 'px' : '-';
  const edStr = edCols > 0 ? (edSum / edCols).toFixed(1) + 'px' : '-';

  document.getElementById('lblMaskStats').textContent =
    `top SC: ${c1} | bot SC: ${c2} | end ED: ${c3}  ||  Tebal SC: ${scStr} | ED: ${edStr}`;
}

function pushUndo() {
  undoStack.push(new Uint8Array(maskData));
  if (undoStack.length > MAX_UNDO) undoStack.shift();
}

function undoStroke() {
  if (undoStack.length === 0) {
    setStatus('Tidak ada riwayat untuk di-undo.');
    return;
  }
  maskData = undoStack.pop();
  isDirty = true;
  renderMaskOverlay();
  updateMaskStats();
  setStatus('Undo berhasil.');
}

function clearMask() {
  if (currentIndex < 0) return;
  pushUndo();
  maskData.fill(0);
  isDirty = true;
  renderMaskOverlay();
  updateMaskStats();
  setStatus('Label direset (Clear).');
}

function stampBrush(cx, cy, cls, size) {
  const r = Math.floor(size / 2);
  const r2 = r * r;
  const xMin = Math.max(0, cx - r);
  const xMax = Math.min(CROP_SIZE - 1, cx + r);
  const yMin = Math.max(0, cy - r);
  const yMax = Math.min(CROP_SIZE - 1, cy + r);

  for (let y = yMin; y <= yMax; y++) {
    const dy = y - cy;
    const rowOff = y * CROP_SIZE;
    for (let x = xMin; x <= xMax; x++) {
      const dx = x - cx;
      if (dx * dx + dy * dy <= r2) {
        maskData[rowOff + x] = cls;
      }
    }
  }
}

function drawBresenhamLine(x0, y0, x1, y1, cls, size) {
  let dx = Math.abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  let dy = -Math.abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  while (true) {
    stampBrush(x0, y0, cls, size);
    if (x0 === x1 && y0 === y1) break;
    let e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

function getCanvasCoords(e) {
  const rect = maskCanvas.getBoundingClientRect();
  const x = Math.floor(((e.clientX - rect.left) / rect.width) * CROP_SIZE);
  const y = Math.floor(((e.clientY - rect.top) / rect.height) * CROP_SIZE);
  return {
    x: Math.max(0, Math.min(CROP_SIZE - 1, x)),
    y: Math.max(0, Math.min(CROP_SIZE - 1, y))
  };
}

let isDrawing = false;
let isPanning = false;
let lastPt = null;
let panStart = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 };
const viewport = document.getElementById('canvasViewport');

maskCanvas.addEventListener('pointerdown', (e) => {
  if (currentIndex < 0) return;
  const penOnly = document.getElementById('chkPenOnly').checked;

  // Jika Mode Khusus Stylus aktif dan layar disentuh jari (touch), gunakan untuk menggeser kanvas!
  if (penOnly && e.pointerType === 'touch') {
    isPanning = true;
    panStart = {
      x: e.clientX,
      y: e.clientY,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop
    };
    maskCanvas.setPointerCapture(e.pointerId);
    return;
  }

  e.preventDefault();
  maskCanvas.setPointerCapture(e.pointerId);
  pushUndo();
  isDrawing = true;
  const pt = getCanvasCoords(e);
  lastPt = pt;
  const drawCls = (e.button === 2) ? 0 : activeClass;
  drawBresenhamLine(pt.x, pt.y, pt.x, pt.y, drawCls, brushSize);
  isDirty = true;
  renderMaskOverlay();
  updateMaskStats();
});

maskCanvas.addEventListener('pointermove', (e) => {
  if (isPanning) {
    viewport.scrollLeft = panStart.scrollLeft - (e.clientX - panStart.x);
    viewport.scrollTop = panStart.scrollTop - (e.clientY - panStart.y);
    return;
  }
  if (currentIndex < 0) return;
  const pt = getCanvasCoords(e);
  const val = maskData[pt.y * CROP_SIZE + pt.x];
  document.getElementById('lblCoords').textContent = `X=${pt.x}, Y=${pt.y} | Mask=${val}`;

  if (!isDrawing || !lastPt) return;
  e.preventDefault();
  const drawCls = (e.buttons === 2) ? 0 : activeClass;
  drawBresenhamLine(lastPt.x, lastPt.y, pt.x, pt.y, drawCls, brushSize);
  lastPt = pt;
  renderMaskOverlay();
});

function stopInteraction() {
  if (isDrawing) {
    isDrawing = false;
    lastPt = null;
    updateMaskStats();
  }
  isPanning = false;
}
maskCanvas.addEventListener('pointerup', stopInteraction);
maskCanvas.addEventListener('pointercancel', stopInteraction);
maskCanvas.addEventListener('contextmenu', (e) => e.preventDefault());

function setActiveClass(cls) {
  activeClass = cls;
  document.getElementById('btnClass1').className = 'tool-btn' + (cls === 1 ? ' active-c1' : '');
  document.getElementById('btnClass2').className = 'tool-btn' + (cls === 2 ? ' active-c2' : '');
  document.getElementById('btnClass3').className = 'tool-btn' + (cls === 3 ? ' active-c3' : '');
  document.getElementById('btnEraser').className = 'tool-btn' + (cls === 0 ? ' active-er' : '');
}

function onBrushChange(val) {
  brushSize = parseInt(val, 10);
  document.getElementById('lblBrushVal').textContent = brushSize + ' px';
}

function onAlphaChange(val) {
  overlayAlpha = parseInt(val, 10) / 100.0;
  document.getElementById('lblAlphaVal').textContent = val + '%';
  renderMaskOverlay();
}

function setZoom(z) {
  zoomLevel = z;
  const stack = document.getElementById('canvasStack');
  stack.style.width = Math.round(CROP_SIZE * z) + 'px';
  stack.style.height = Math.round(CROP_SIZE * z) + 'px';
}

// =========================================================================
// Simpan Langsung ke Hard Disk Laptop
// =========================================================================
function uint8ToBase64(u8) {
  const CHUNK = 0x8000;
  let result = '';
  for (let i = 0; i < u8.length; i += CHUNK) {
    result += String.fromCharCode.apply(null, u8.subarray(i, i + CHUNK));
  }
  return btoa(result);
}

async function saveCurrent(showToast = true) {
  if (currentIndex < 0) return false;
  setStatus('Menyimpan Crop + Strip Tengah + Mask ke Laptop...');
  const payload = {
    index: currentIndex,
    mode: document.getElementById('selCropMode').value,
    custom_x: parseInt(document.getElementById('numStartX').value || '0', 10),
    custom_y: parseInt(document.getElementById('numStartY').value || '0', 10),
    subfolder_mode: document.getElementById('chkSubfolder').checked,
    slice_x: 256,
    strip_width: parseInt(document.getElementById('numStripWidth').value || '8', 10),
    mask_raw_b64: uint8ToBase64(maskData)
  };

  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const out = await res.json();
    if (out.ok) {
      isDirty = false;
      files[currentIndex].labeled = true;
      renderFileList(
        out.stats.labeled_count,
        out.stats.total_files,
        out.stats.labeled_points,
        out.stats.total_points
      );
      setStatus('✓ ' + out.message);
      return true;
    } else {
      alert('Gagal menyimpan: ' + out.message);
      return false;
    }
  } catch (e) {
    alert('Error koneksi saat menyimpan: ' + e.message);
    return false;
  }
}

async function prevImage() {
  if (currentIndex > 0) await selectFileIndex(currentIndex - 1);
}

async function nextImage() {
  if (currentIndex < files.length - 1) await selectFileIndex(currentIndex + 1);
}

async function randomSlice() {
  if (!files || files.length === 0) return;
  // JANGAN Auto-Save saat menekan Ganti Slice Acak, karena slice ini ditolak (strukturnya kurang bagus)
  isDirty = false;
  try {
    const res = await fetch('/api/reroll_point', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: currentIndex })
    });
    const out = await res.json();
    if (out.ok) {
      await fetchStatus(false);
      await loadImage(out.index, true);
      setStatus(out.message);
    }
  } catch (e) {
    setStatus('Gagal memilih sampel acak: ' + e.message);
  }
}

async function switchSiblingPoint(delta) {
  if (document.getElementById('chkAutoSave').checked && isDirty && hasAnyMaskStroke()) {
    await saveCurrent(false);
  }
  try {
    const res = await fetch('/api/switch_point', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delta: delta })
    });
    const out = await res.json();
    if (out.ok) {
      currentIndex = -1;
      await fetchStatus(true);
      setStatus('✓ ' + out.message);
    } else {
      alert(out.message);
    }
  } catch (e) {
    alert('Error pindah titik: ' + e.message);
  }
}

async function loadParentAllPoints() {
  if (document.getElementById('chkAutoSave').checked && isDirty && hasAnyMaskStroke()) {
    await saveCurrent(false);
  }
  try {
    const res = await fetch('/api/load_parent_points', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    const out = await res.json();
    if (out.ok) {
      currentIndex = -1;
      await fetchStatus(true);
      setStatus('✓ ' + out.message);
    } else {
      alert(out.message);
    }
  } catch (e) {
    alert('Error memuat semua titik: ' + e.message);
  }
}

async function onToggleOneSamplePerPoint() {
  const checked = document.getElementById('chkOneSamplePerPoint').checked;
  const res = await fetch('/api/set_dirs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      subfolder_mode: document.getElementById('chkSubfolder').checked,
      one_sample_per_point: checked
    })
  });
  const out = await res.json();
  if (out.ok) {
    currentIndex = -1;
    await fetchStatus(true);
  }
}

async function stitchAllFrames() {
  if (files.length === 0) {
    alert('Pilih Folder Sumber terlebih dahulu.');
    return;
  }
  const fpg = parseInt(document.getElementById('numFramesPerGroup').value || '10', 10);
  setStatus(`Menggabungkan irisan tengah (${fpg > 0 ? 'per ' + fpg + ' frame' : 'semua frame'}) menjadi gambar Transit OCT...`);
  try {
    const res = await fetch('/api/stitch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: document.getElementById('selCropMode').value,
        custom_x: parseInt(document.getElementById('numStartX').value || '0', 10),
        custom_y: parseInt(document.getElementById('numStartY').value || '0', 10),
        slice_x: 256,
        frames_per_group: fpg
      })
    });
    const out = await res.json();
    if (out.ok) {
      currentIndex = -1;
      await fetchStatus(true);
      setStatus('✓ ' + out.message);
      alert(out.message + '\n\nGambar gabungan sudah dimuat di urutan pertama (#1). Anda cukup melabeli 1 gambar ini saja!');
    } else {
      alert('Gagal menggabungkan frame: ' + out.message);
    }
  } catch (e) {
    alert('Error saat menggabungkan frame: ' + e.message);
  }
}

// =========================================================================
// Penjelajah Folder Laptop dari Browser Tablet
// =========================================================================
let modalTarget = 'input';
let currentBrowseParent = '';

async function openFolderModal(target) {
  modalTarget = target;
  document.getElementById('modalTitle').textContent =
    target === 'input' ? 'Pilih Folder Sumber OCT di Laptop' : 'Pilih Folder Output di Laptop';
  document.getElementById('folderModal').style.display = 'flex';
  await browsePath(defaultBrowseDir || '');
}

function closeFolderModal() {
  document.getElementById('folderModal').style.display = 'none';
}

async function browsePath(pathStr) {
  const res = await fetch('/api/browse?path=' + encodeURIComponent(pathStr || ''));
  const data = await res.json();
  if (!data.ok) {
    alert(data.message);
    return;
  }
  document.getElementById('txtFolderPath').value = data.current_path;
  currentBrowseParent = data.parent_path || '';
  document.getElementById('lblFolderImgCount').textContent =
    data.current_path ? `${data.image_count} gambar` : 'Pilih Drive';

  const listEl = document.getElementById('folderList');
  listEl.innerHTML = '';
  data.entries.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'folder-row';
    row.innerHTML = `<span>📁</span> <span>${item.name}</span>`;
    row.onclick = () => browsePath(item.path);
    listEl.appendChild(row);
  });
}

function browseParent() {
  browsePath(currentBrowseParent);
}

async function confirmFolderSelection() {
  const chosen = document.getElementById('txtFolderPath').value.trim();
  if (!chosen) {
    alert('Pilih salah satu folder terlebih dahulu.');
    return;
  }
  const body = {
    subfolder_mode: document.getElementById('chkSubfolder').checked,
    one_sample_per_point: document.getElementById('chkOneSamplePerPoint').checked
  };
  if (modalTarget === 'input') body.input_dir = chosen;
  else body.output_dir = chosen;

  const res = await fetch('/api/set_dirs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const out = await res.json();
  if (!out.ok) {
    alert(out.message);
    return;
  }
  closeFolderModal();
  currentIndex = -1;
  await fetchStatus(true);
}

fetchStatus(true);
</script>
</body>
</html>
"""


MANIFEST_JSON = json.dumps(
    {
        "name": "OCT Labeling App (Remote Laptop)",
        "short_name": "OCT Labeler",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#121418",
        "theme_color": "#121418",
    }
)


class TabletRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def _send_json(self, data: Dict, status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send_html(INDEX_HTML)
            return

        if path == "/manifest.json":
            raw = MANIFEST_JSON.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/status":
            self._send_json(STATE.get_status_dict())
            return

        if path == "/api/browse":
            target_path = qs.get("path", [""])[0]
            try:
                info = list_filesystem_dir(target_path)
                info["ok"] = True
                self._send_json(info)
            except Exception as e:
                self._send_json({"ok": False, "message": str(e)}, status=400)
            return

        if path == "/api/image":
            try:
                idx = int(qs.get("index", ["0"])[0])
                mode = qs.get("mode", ["top_center"])[0]
                cx = int(qs.get("x", ["0"])[0])
                cy = int(qs.get("y", ["0"])[0])
                inc_mask = qs.get("mask", ["1"])[0] == "1"
                payload = STATE.get_image_payload(idx, mode, cx, cy, inc_mask)
                if payload is None:
                    self._send_json(
                        {"ok": False, "message": "Gambar tidak ditemukan"}, status=404
                    )
                else:
                    payload["ok"] = True
                    self._send_json(payload)
            except Exception as e:
                self._send_json({"ok": False, "message": str(e)}, status=500)
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            self._send_json({"ok": False, "message": "JSON tidak valid"}, status=400)
            return

        if path == "/api/set_dirs":
            ok, msg = STATE.set_directories(
                payload.get("input_dir"),
                payload.get("output_dir"),
                payload.get("subfolder_mode"),
                payload.get("one_sample_per_point"),
            )
            self._send_json({"ok": ok, "message": msg})
            return

        if path == "/api/reroll_point":
            idx = int(payload.get("index", 0))
            ok, msg, new_idx = STATE.reroll_point_sample(idx)
            self._send_json({"ok": ok, "message": msg, "index": new_idx})
            return

        if path == "/api/switch_point":
            delta = int(payload.get("delta", 1))
            ok, msg = STATE.switch_sibling_point(delta)
            self._send_json({"ok": ok, "message": msg})
            return

        if path == "/api/load_parent_points":
            ok, msg = STATE.load_parent_all_points()
            self._send_json({"ok": ok, "message": msg})
            return

        if path == "/api/save":
            try:
                ok, msg, stats = STATE.save_crop_and_mask(
                    index=int(payload["index"]),
                    mode=str(payload.get("mode", "top_center")),
                    custom_x=int(payload.get("custom_x", 0)),
                    custom_y=int(payload.get("custom_y", 0)),
                    mask_raw_b64=str(payload["mask_raw_b64"]),
                    subfolder_mode=bool(payload.get("subfolder_mode", True)),
                    slice_x=int(payload.get("slice_x", 256)),
                    strip_width=int(payload.get("strip_width", 8)),
                )
                print(f"[SAVE] {msg}")
                self._send_json({"ok": ok, "message": msg, "stats": stats})
            except Exception as e:
                traceback.print_exc()
                self._send_json({"ok": False, "message": str(e)}, status=500)
            return

        if path == "/api/stitch":
            try:
                ok, msg, idx = STATE.stitch_all_frames(
                    mode=str(payload.get("mode", "top_center")),
                    custom_x=int(payload.get("custom_x", 0)),
                    custom_y=int(payload.get("custom_y", 0)),
                    slice_x=int(payload.get("slice_x", 256)),
                    frames_per_group=int(payload.get("frames_per_group", 10)),
                )
                print(f"[STITCH] {msg}")
                self._send_json({"ok": ok, "message": msg, "index": idx})
            except Exception as e:
                traceback.print_exc()
                self._send_json({"ok": False, "message": str(e)}, status=500)
            return

        self.send_error(404, "Not Found")


def get_local_ips() -> List[str]:
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("127.") and ip not in ips:
            ips.insert(0, ip)
    except Exception:
        pass
    return ips or ["127.0.0.1"]


def main() -> None:
    base_port = int(os.environ.get("OCT_PORT", "8080"))
    server = None
    port = base_port
    for candidate_port in range(base_port, base_port + 10):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", candidate_port), TabletRequestHandler)
            port = candidate_port
            break
        except OSError:
            continue

    if server is None:
        print(f"[ERROR] Tidak dapat membuka port {base_port}..{base_port + 9}")
        sys.exit(1)

    ips = get_local_ips()

    print("=" * 74)
    print("  OCT PREPROCESSING & LABELING SERVER (AKSES DARI TABLET VIA WI-FI)")
    print("=" * 74)
    print("  Dataset tetap berada di Laptop. Tablet hanya menampilkan antarmuka")
    print("  dan semua hasil Crop + Mask langsung tersimpan ke hard disk Laptop!")
    print("-" * 74)
    print("  Pastikan Laptop dan Tablet terhubung ke Wi-Fi / Hotspot yang sama,")
    print("  lalu buka salah satu alamat berikut di Chrome / Browser Tablet Anda:\n")
    for ip in ips:
        print(f"    👉  http://{ip}:{port}")
    print(f"\n  (Jika dibuka di Laptop ini sendiri: http://localhost:{port})")
    print("=" * 74)
    print("  Tekan Ctrl+C untuk menghentikan server.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server dihentikan.")
        server.server_close()


if __name__ == "__main__":
    main()
