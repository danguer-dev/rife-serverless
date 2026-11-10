import os
import cv2
import torch
import base64
import runpod
import requests
import subprocess
import numpy as np

# Carga del modelo Practical-RIFE (se cachea en memoria GPU tras el primer arranque)
from model.RIFE_HDv3 import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.load_model("./train_log", -1)
model.eval()
model.device()

def download_image(url_or_b64):
    if url_or_b64.startswith("http"):
        resp = requests.get(url_or_b64, timeout=10)
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        decoded = base64.b64decode(url_or_b64)
        img_array = np.asarray(bytearray(decoded), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def make_inference(I0, I1, num_frames=8):
    # RIFE requiere dimensiones múltiplos de 32
    h, w, _ = I0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)

    I0_t = torch.from_numpy(np.transpose(I0, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
    I1_t = torch.from_numpy(np.transpose(I1, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
    I0_t = torch.nn.functional.pad(I0_t, padding)
    I1_t = torch.nn.functional.pad(I1_t, padding)

    frames = [I0]
    # Genera puntos de tiempo intermedios uniformes: t in (0, 1)
    timesteps = np.linspace(0, 1, num_frames)[1:-1]
    
    with torch.no_grad():
        for t in timesteps:
            mid = model.inference(I0_t, I1_t, float(t))
            mid_np = (((mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)))[:h, :w]
            frames.append(mid_np)
            
    frames.append(I1)
    return frames

def handler(job):
    job_input = job.get("input", {})
    img_start_raw = job_input.get("image_start")
    img_end_raw = job_input.get("image_end")
    num_frames = int(job_input.get("num_frames", 8))
    fps = int(job_input.get("fps", 2))

    if not img_start_raw or not img_end_raw:
        return {"error": "Faltan las imágenes 'image_start' o 'image_end'"}

    img0 = download_image(img_start_raw)
    img1 = download_image(img_end_raw)

    # Asegurar coincidencia exacta de resolución
    h0, w0 = img0.shape[:2]
    if img1.shape[:2] != (h0, w0):
        img1 = cv2.resize(img1, (w0, h0), interpolation=cv2.INTER_LANCZOS4)

    # Interpolación
    frames = make_inference(img0, img1, num_frames=num_frames)

    # Render a MP4 temporal mediante FFmpeg
    output_video_path = "/tmp/output.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w0}x{h0}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_video_path
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

    # Codificación del resultado
    with open(output_video_path, "rb") as vf:
        video_b64 = base64.b64encode(vf.read()).decode("utf-8")

    return {
        "video_base64": video_b64,
        "frames_rendered": len(frames),
        "fps": fps
    }

runpod.serverless.start({"handler": handler})