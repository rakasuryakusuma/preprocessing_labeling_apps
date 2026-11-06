import argparse
import json
import mimetypes
import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


CROP_SIZE = 512
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def imread_unicode(file_path: Path, flags: int = cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(file_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(file_path: Path, img: np.ndarray) -> bool:
    try:
        ext = file_path.suffix or ".png"
        ok, enc = cv2.imencode(ext, img)
        if not ok:
            return False
        enc.tofile(str(file_path))
        return True
    except Exception:
        return False


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class TabletRequestHandler(BaseHTTPRequestHandler):
    web_root: Path = Path(__file__).parent / "tablet_app"
    input_dir: Path = Path(__file__).parent / "input_oct"
    output_dir: Path = Path(__file__).parent / "output_oct"

    def log_message(self, format, *args):
        pass  # Buat log terminal tetap bersih

    def _send_json(self, data: dict, status: int = 200):
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)

        if route == "/api/status":
            files = []
            if self.input_dir.exists():
                for p in sorted(self.input_dir.iterdir()):
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        if p.name.endswith("_crop.png") or p.name.endswith("_mask.png"):
                            continue
                        mask_p = self.output_dir / f"{p.stem}_mask.png"
                        sub_mask_p = self.output_dir / "masks" / f"{p.stem}_mask.png"
                        files.append(
                            {
                                "name": p.name,
                                "stem": p.stem,
                                "labeled": mask_p.exists() or sub_mask_p.exists(),
                            }
                        )
            self._send_json(
                {
                    "ok": True,
                    "input_dir": str(self.input_dir),
                    "output_dir": str(self.output_dir),
                    "files": files,
                }
            )
            return

        if route == "/api/image":
            name = qs.get("name", [""])[0]
            img_path = self.input_dir / name
            if not img_path.exists() or not img_path.is_file():
                self.send_error(404, "Gambar tidak ditemukan")
                return
            mime, _ = mimetypes.guess_type(str(img_path))
            raw = img_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime or "image/jpeg")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if route == "/api/mask":
            stem = qs.get("stem", [""])[0]
            p1 = self.output_dir / f"{stem}_mask.png"
            p2 = self.output_dir / "masks" / f"{stem}_mask.png"
            cand = p1 if p1.exists() else p2
            if not cand.exists():
                self.send_error(404, "Mask belum ada")
                return
            mask = imread_unicode(cand, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (CROP_SIZE, CROP_SIZE):
                self.send_error(404, "Mask tidak valid")
                return
            raw = mask.astype(np.uint8).tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        # Layani file statis dari folder tablet_app/
        rel = route.lstrip("/") or "index.html"
        target = (self.web_root / rel).resolve()
        if not str(target).startswith(str(self.web_root.resolve())) or not target.is_file():
            target = self.web_root / "index.html"

        mime, _ = mimetypes.guess_type(str(target))
        raw = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            stem = body["stem"]
            sx, sy = body.get("crop_coords", [244, 0])
            mask_vals = np.array(body["mask_values"], dtype=np.uint8).reshape(
                (CROP_SIZE, CROP_SIZE)
            )
            clean_mask = np.clip(mask_vals, 0, 2).astype(np.uint8)

            # Cari file sumber
            src_file = None
            if self.input_dir.exists():
                for p in self.input_dir.iterdir():
                    if p.is_file() and p.stem == stem and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        src_file = p
                        break

            self.output_dir.mkdir(parents=True, exist_ok=True)
            img_sub = self.output_dir / "images"
            msk_sub = self.output_dir / "masks"
            img_sub.mkdir(parents=True, exist_ok=True)
            msk_sub.mkdir(parents=True, exist_ok=True)

            if src_file is not None:
                raw_img = imread_unicode(src_file, cv2.IMREAD_COLOR)
                if raw_img is not None:
                    sx = max(0, min(raw_img.shape[1] - CROP_SIZE, int(sx)))
                    sy = max(0, min(raw_img.shape[0] - CROP_SIZE, int(sy)))
                    crop_bgr = raw_img[sy : sy + CROP_SIZE, sx : sx + CROP_SIZE].copy()
                    imwrite_unicode(self.output_dir / f"{stem}_crop.png", crop_bgr)
                    imwrite_unicode(img_sub / f"{stem}_crop.png", crop_bgr)

            imwrite_unicode(self.output_dir / f"{stem}_mask.png", clean_mask)
            imwrite_unicode(msk_sub / f"{stem}_mask.png", clean_mask)

            print(f"[TERSIMPAN DARI TABLET] {stem}_crop.png & {stem}_mask.png")
            self._send_json({"ok": True})
            return

        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description="Server Sinkronisasi Tablet Redmi Pad 2")
    parser.add_argument("--input", type=str, default="", help="Folder sumber gambar OCT")
    parser.add_argument("--output", type=str, default="", help="Folder output hasil crop & mask")
    parser.add_argument("--port", type=int, default=8080, help="Port server (default: 8080)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    in_dir = Path(args.input) if args.input else (base_dir / "input_oct")
    out_dir = Path(args.output) if args.output else (base_dir / "output_oct")
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    TabletRequestHandler.input_dir = in_dir
    TabletRequestHandler.output_dir = out_dir

    ip = get_local_ip()
    server = HTTPServer(("0.0.0.0", args.port), TabletRequestHandler)

    print("=" * 74)
    print("  SERVER TABLET (REDMI PAD 2) AKTIF!")
    print("=" * 74)
    print(f"  Folder Input OCT  : {in_dir}")
    print(f"  Folder Output     : {out_dir}")
    print("-" * 74)
    print("  CARA BUKA & INSTALL DI REDMI PAD 2:")
    print("  1. Pastikan Redmi Pad 2 & Laptop terhubung ke Wi-Fi / Hotspot yang sama.")
    print(f"  2. Buka Google Chrome di Redmi Pad 2, lalu ketik alamat ini:")
    print(f"\n       http://{ip}:{args.port}\n")
    print("  3. Di Chrome Redmi Pad 2, ketuk tombol 'Install App' (atau menu titik tiga")
    print("     -> 'Tambahkan ke layar utama') agar ter-install di Tablet Anda!")
    print("=" * 74)
    print("  Tekan Ctrl+C untuk menghentikan server.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")


if __name__ == "__main__":
    main()
