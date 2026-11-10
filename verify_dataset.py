import argparse
from pathlib import Path
import sys
import cv2
import numpy as np


EXPECTED_SHAPE = (512, 512)
VALID_CLASSES = {0, 1, 2, 3}


def imread_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def verify_output_directory(output_dir: Path, save_preview: bool = False) -> bool:
    """
    Memeriksa seluruh file *_mask.png pada direktori output (termasuk subfolder masks/ jika ada):
    1. Memastikan ukuran gambar adalah 512x512 piksel grayscale (2D).
    2. Memastikan nilai piksel murni hanya terdiri dari subset {0, 1, 2, 3}:
       - 1: top SC (Batas Paling Atas - Merah)
       - 2: bot SC (Batas Tengah - Kuning)
       - 3: end ED (Batas Bawah - Hijau)
    """
    if not output_dir.exists():
        print(f"[ERROR] Direktori tidak ditemukan: {output_dir}")
        return False

    mask_dir = output_dir / "masks" if (output_dir / "masks").exists() else output_dir
    mask_files = sorted(mask_dir.glob("*_mask.png"))

    if not mask_files:
        print(f"[PERINGATAN] Tidak ditemukan file *_mask.png di {mask_dir}")
        return False

    print(f"Memeriksa {len(mask_files)} file mask di: {mask_dir}\n" + "-" * 72)
    all_valid = True

    preview_dir = output_dir / "debug_previews"
    if save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for mask_path in mask_files:
        mask = imread_gray(mask_path)
        if mask is None:
            print(f"[GAGAL] {mask_path.name}: Tidak dapat dibaca")
            all_valid = False
            continue

        shape_ok = mask.shape == EXPECTED_SHAPE
        unique_vals = set(np.unique(mask).tolist())
        vals_ok = unique_vals.issubset(VALID_CLASSES)

        c1_count = int(np.sum(mask == 1))
        c2_count = int(np.sum(mask == 2))
        c3_count = int(np.sum(mask == 3))

        status = "OK" if (shape_ok and vals_ok) else "INVALID"
        if status != "OK":
            all_valid = False

        print(
            f"[{status}] {mask_path.name:<28} | Shape: {str(mask.shape):<10} | "
            f"Nilai Unik: {sorted(unique_vals)} | topSC(1): {c1_count} px, botSC(2): {c2_count} px, endED(3): {c3_count} px"
        )

        if save_preview and shape_ok:
            # Buat gambar visualisasi berwarna sesuai referensi (Merah, Kuning, Hijau)
            vis = np.zeros((512, 512, 3), dtype=np.uint8)
            vis[mask == 1] = (45, 45, 255)   # BGR: Merah untuk Class 1 (top SC)
            vis[mask == 2] = (0, 230, 255)   # BGR: Kuning untuk Class 2 (bot SC)
            vis[mask == 3] = (64, 230, 0)    # BGR: Hijau untuk Class 3 (end ED)
            out_vis_path = preview_dir / f"{mask_path.stem}_preview.png"
            _, enc = cv2.imencode(".png", vis)
            enc.tofile(str(out_vis_path))

    print("-" * 72)
    if all_valid:
        print("[SUKSES] Semua file mask valid (512x512, nilai piksel murni 0, 1, 2, 3) dan siap untuk PyTorch U-Net!")
    else:
        print("[PERHATIAN] Ditemukan file mask yang tidak sesuai spesifikasi.")

    if save_preview:
        print(f"[INFO] Preview berwarna disimpan di: {preview_dir}")

    return all_valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skrip Debugging & Verifikasi Dataset Masker OCT (Milestone 4)"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path ke folder output yang berisi file *_mask.png",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Simpan visualisasi mask berwarna ke subfolder debug_previews/",
    )
    args = parser.parse_args()
    ok = verify_output_directory(args.output_dir, save_preview=args.preview)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
