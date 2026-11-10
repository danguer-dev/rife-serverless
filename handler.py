import os
import cv2
import shutil
import base64
import runpod
import requests
import subprocess
import numpy as np

def download_image(url_or_b64):
    if url_or_b64.startswith("http://") or url_or_b64.startswith("https://"):
        resp = requests.get(url_or_b64, timeout=15)
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        if "," in url_or_b64:
            url_or_b64 = url_or_b64.split(",")[1]
        decoded = base64.b64decode(url_or_b64)
        img_array = np.asarray(bytearray(decoded), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def handler(job):
    job_input = job.get("input", {})
    img_start_raw = job_input.get("image_start")
    img_end_raw = job_input.get("image_end")
    num_frames = int(job_input.get("num_frames", 8))
    fps = int(job_input.get("fps", 2))

    if not img_start_raw or not img_end_raw:
        return {"error": "Faltan los parametros 'image_start' o 'image_end'"}

    img0 = download_image(img_start_raw)
    img1 = download_image(img_end_raw)

    if img0 is None or img1 is None:
        return {"error": "No se pudieron decodificar las imagenes de entrada"}

    # Igualar resolucion de la segunda imagen con la primera
    h0, w0 = img0.shape[:2]
    if img1.shape[:2] != (h0, w0):
        img1 = cv2.resize(img1, (w0, h0), interpolation=cv2.INTER_LANCZOS4)

    in_dir = "/tmp/rife_in"
    out_dir = "/tmp/rife_out"
    shutil.rmtree(in_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(f"{in_dir}/0.png", img0)
    cv2.imwrite(f"{in_dir}/1.png", img1)

    # Ejecutar rife-ncnn-vulkan con el modelo v4.6
    rife_cmd = [
        "/app/rife-ncnn-vulkan",
        "-i", in_dir,
        "-o", out_dir,
        "-n", str(num_frames),
        "-m", "rife-v4.6",
        "-f", "%08d.png"
    ]
    proc = subprocess.run(rife_cmd, capture_output=True, text=True)

    # Fallback a CPU si el host de contenedor restringe el ICD de Vulkan
    if proc.returncode != 0 or not os.listdir(out_dir):
        rife_cmd.extend(["-g", "-1"])
        subprocess.run(rife_cmd, check=True)

    output_video_path = "/tmp/output.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{out_dir}/%08d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_video_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    with open(output_video_path, "rb") as vf:
        video_b64 = base64.b64encode(vf.read()).decode("utf-8")

    return {
        "video_base64": video_b64,
        "frames_rendered": len(os.listdir(out_dir)),
        "fps": fps
    }

runpod.serverless.start({"handler": handler})
