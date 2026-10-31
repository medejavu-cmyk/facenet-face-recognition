"""Web 端训练集样本识别演示。

运行位置：D:\face_lfw_project
启动命令：python web_face_app.py
浏览器访问：http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import io
import json
import math
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "lfw_home" / "lfw_funneled"
OUTPUT_DIR = ROOT / "outputs_lfw_facenet_50"
CLASSIFIER_PATH = OUTPUT_DIR / "facenet_svm_classifier.pkl"
SUMMARY_PATH = OUTPUT_DIR / "result_summary.json"
HOST = "127.0.0.1"
PORT = 8000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LFW 人脸识别系统</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei UI", "Segoe UI", Arial, sans-serif;
      color: #162033;
      background: #eef2f7;
    }
    header {
      min-height: 78px;
      padding: 16px 28px;
      color: white;
      background: #172033;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .subtitle {
      margin-top: 4px;
      color: #cbd5e1;
      font-size: 13px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(430px, 1fr) minmax(360px, 0.85fr);
      gap: 18px;
      padding: 18px;
      max-width: 1240px;
      margin: 0 auto;
    }
    section {
      background: white;
      border: 1px solid #dbe3ef;
      border-radius: 8px;
      overflow: hidden;
    }
    .panel-title {
      padding: 14px 16px;
      border-bottom: 1px solid #e5eaf2;
      font-size: 16px;
      font-weight: 700;
      background: #fbfdff;
    }
    .selector {
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    label {
      display: grid;
      gap: 7px;
      color: #475569;
      font-size: 13px;
      font-weight: 700;
    }
    select {
      width: 100%;
      height: 42px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 0 10px;
      background: white;
      color: #111827;
      font-size: 14px;
    }
    .preview {
      padding: 0 16px 16px;
    }
    .image-box {
      min-height: 420px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .image-box img {
      max-width: 100%;
      max-height: 560px;
      object-fit: contain;
      display: block;
    }
    .empty {
      color: #64748b;
      font-size: 14px;
      padding: 24px;
      text-align: center;
      line-height: 1.7;
    }
    .controls {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 16px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      color: white;
      background: #2563eb;
    }
    button.secondary { background: #475569; }
    .upload-label {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      border-radius: 6px;
      padding: 9px 14px;
      color: #111827;
      background: #e2e8f0;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .upload-label input { display: none; }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .hint {
      margin-top: 12px;
      line-height: 1.7;
      color: #64748b;
      font-size: 13px;
    }
    .result {
      padding: 18px;
    }
    .badge {
      display: inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      color: #1d4ed8;
      background: #dbeafe;
      font-size: 13px;
      font-weight: 700;
    }
    .match { color: #166534; background: #dcfce7; }
    .mismatch { color: #991b1b; background: #fee2e2; }
    .name {
      margin: 14px 0 8px;
      font-size: 32px;
      font-weight: 800;
      color: #111827;
      min-height: 42px;
      line-height: 1.25;
    }
    .confidence {
      color: #334155;
      font-size: 15px;
      margin-bottom: 16px;
      line-height: 1.8;
    }
    .bars {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }
    .bar-row {
      display: grid;
      gap: 6px;
    }
    .bar-meta {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      color: #334155;
      gap: 12px;
    }
    .bar {
      height: 10px;
      border-radius: 999px;
      background: #e2e8f0;
      overflow: hidden;
    }
    .bar > div {
      height: 100%;
      background: #2563eb;
      width: 0;
      transition: width 180ms ease;
    }
    .info {
      border-top: 1px solid #e5eaf2;
      padding: 16px 18px;
      color: #475569;
      line-height: 1.75;
      font-size: 14px;
      background: #fbfdff;
    }
    .classes {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .chip {
      padding: 6px 9px;
      border-radius: 6px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 700;
    }
    .error {
      color: #b91c1c;
      background: #fee2e2;
      border: 1px solid #fecaca;
      border-radius: 6px;
      padding: 10px 12px;
      display: none;
      margin: 0 16px 14px;
    }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; padding: 12px; }
      .name { font-size: 27px; }
      .image-box { min-height: 320px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>基于 PyTorch 的人脸识别系统</h1>
      <div class="subtitle">选择 LFW 训练集人物样本，使用 FaceNet + SVM 进行身份识别</div>
    </div>
    <div class="subtitle">访问地址：http://127.0.0.1:8000</div>
  </header>

  <main>
    <section>
      <div class="panel-title">训练集样本选择</div>
      <div class="selector">
        <label>
          人物类别
          <select id="personSelect"></select>
        </label>
        <label>
          样本图片
          <select id="imageSelect"></select>
        </label>
      </div>
      <div id="error" class="error"></div>
      <div class="preview">
        <div class="image-box" id="imageBox">
          <div class="empty">请选择人物和样本图片</div>
        </div>
        <div class="controls">
          <button id="predictBtn" disabled>识别当前样本</button>
          <button class="secondary" id="randomBtn" disabled>随机换一张</button>
          <label class="upload-label">
            上传本地照片
            <input id="uploadInput" type="file" accept="image/*" />
          </label>
          <button id="uploadPredictBtn" disabled>识别上传照片</button>
        </div>
        <div class="hint">
          这个网页不使用摄像头，可以从 LFW 训练集/数据集中选择图片，也可以上传本地照片进行识别演示。
          适合放在实训报告的“系统可视化展示”部分。
        </div>
      </div>
    </section>

    <section>
      <div class="panel-title">识别结果</div>
      <div class="result">
        <span class="badge" id="status">等待识别</span>
        <div class="name" id="name">-</div>
        <div class="confidence" id="confidence">真实类别：-<br />置信度：-</div>
        <div class="bars" id="bars"></div>
      </div>
      <div class="info">
        <strong>可视化说明</strong><br />
        页面展示样本图片、真实人物标签、模型预测标签和 Top-3 相似类别。
        如果预测标签与真实标签一致，说明该样本识别正确。
        <div class="classes" id="classes"></div>
      </div>
    </section>
  </main>

  <script>
    const personSelect = document.getElementById("personSelect");
    const imageSelect = document.getElementById("imageSelect");
    const imageBox = document.getElementById("imageBox");
    const predictBtn = document.getElementById("predictBtn");
    const randomBtn = document.getElementById("randomBtn");
    const uploadInput = document.getElementById("uploadInput");
    const uploadPredictBtn = document.getElementById("uploadPredictBtn");
    const errorBox = document.getElementById("error");
    const statusEl = document.getElementById("status");
    const nameEl = document.getElementById("name");
    const confidenceEl = document.getElementById("confidence");
    const barsEl = document.getElementById("bars");
    const classesEl = document.getElementById("classes");

    let people = [];
    let currentImage = null;
    let currentPerson = null;
    let uploadedImage = null;

    function showError(message) {
      errorBox.textContent = message;
      errorBox.style.display = "block";
    }

    function clearError() {
      errorBox.textContent = "";
      errorBox.style.display = "none";
    }

    function renderTop(top) {
      barsEl.innerHTML = "";
      for (const item of top) {
        const row = document.createElement("div");
        row.className = "bar-row";
        const meta = document.createElement("div");
        meta.className = "bar-meta";
        meta.innerHTML = `<span>${item.name}</span><span>${(item.score * 100).toFixed(1)}%</span>`;
        const bar = document.createElement("div");
        bar.className = "bar";
        const fill = document.createElement("div");
        fill.style.width = `${Math.max(2, item.score * 100)}%`;
        bar.appendChild(fill);
        row.appendChild(meta);
        row.appendChild(bar);
        barsEl.appendChild(row);
      }
    }

    async function api(path) {
      const res = await fetch(path);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "请求失败");
      return data;
    }

    async function loadPeople() {
      clearError();
      const data = await api("/people");
      people = data.people || [];
      personSelect.innerHTML = "";
      classesEl.innerHTML = "";

      for (const person of people) {
        const option = document.createElement("option");
        option.value = person.name;
        option.textContent = `${person.name} (${person.count} 张)`;
        personSelect.appendChild(option);

        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = person.name;
        classesEl.appendChild(chip);
      }

      if (people.length > 0) {
        await loadImages(people[0].name);
      }
    }

    async function loadImages(personName) {
      clearError();
      currentPerson = personName;
      const data = await api(`/images?person=${encodeURIComponent(personName)}`);
      imageSelect.innerHTML = "";
      for (const image of data.images || []) {
        const option = document.createElement("option");
        option.value = image;
        option.textContent = image;
        imageSelect.appendChild(option);
      }
      if ((data.images || []).length > 0) {
        await loadImage(personName, data.images[0]);
      }
    }

    async function loadImage(personName, filename) {
      clearError();
      currentPerson = personName;
      currentImage = filename;
      const data = await api(`/image?person=${encodeURIComponent(personName)}&file=${encodeURIComponent(filename)}`);
      imageBox.innerHTML = "";
      const img = document.createElement("img");
      img.src = data.image;
      img.alt = `${personName} - ${filename}`;
      imageBox.appendChild(img);
      predictBtn.disabled = false;
      randomBtn.disabled = false;
      uploadPredictBtn.disabled = !uploadedImage;
      statusEl.textContent = "已选择样本";
      statusEl.className = "badge";
      nameEl.textContent = "-";
      confidenceEl.innerHTML = `真实类别：${personName}<br />置信度：-`;
      barsEl.innerHTML = "";
    }

    async function randomImage() {
      if (!currentPerson) return;
      const data = await api(`/random?person=${encodeURIComponent(currentPerson)}`);
      imageSelect.value = data.file;
      await loadImage(currentPerson, data.file);
    }

    async function predict() {
      if (!currentPerson || !currentImage) return;
      clearError();
      predictBtn.disabled = true;
      statusEl.textContent = "识别中...";
      statusEl.className = "badge";
      try {
        const data = await api(`/predict?person=${encodeURIComponent(currentPerson)}&file=${encodeURIComponent(currentImage)}`);
        nameEl.textContent = data.predicted_name;
        confidenceEl.innerHTML =
          `真实类别：${data.true_name}<br />置信度：${(data.confidence * 100).toFixed(1)}%`;
        statusEl.textContent = data.is_correct ? "识别正确" : "识别不一致";
        statusEl.className = data.is_correct ? "badge match" : "badge mismatch";
        renderTop(data.top);
      } catch (err) {
        statusEl.textContent = "识别失败";
        showError(err.message || "识别失败");
      } finally {
        predictBtn.disabled = false;
      }
    }

    function showUploadedImage(file) {
      clearError();
      const reader = new FileReader();
      reader.onload = () => {
        uploadedImage = reader.result;
        imageBox.innerHTML = "";
        const img = document.createElement("img");
        img.src = uploadedImage;
        img.alt = file.name;
        imageBox.appendChild(img);
        uploadPredictBtn.disabled = false;
        statusEl.textContent = "已上传照片";
        statusEl.className = "badge";
        nameEl.textContent = "-";
        confidenceEl.innerHTML = "真实类别：本地上传照片<br />置信度：-";
        barsEl.innerHTML = "";
      };
      reader.onerror = () => showError("读取上传照片失败");
      reader.readAsDataURL(file);
    }

    async function predictUploaded() {
      if (!uploadedImage) return;
      clearError();
      uploadPredictBtn.disabled = true;
      statusEl.textContent = "识别中...";
      statusEl.className = "badge";
      try {
        const res = await fetch("/upload_predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: uploadedImage })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "识别失败");
        nameEl.textContent = data.predicted_name;
        confidenceEl.innerHTML =
          `真实类别：本地上传照片<br />置信度：${(data.confidence * 100).toFixed(1)}%`;
        statusEl.textContent = "已在 12 个已知类别中找到最相似人物";
        statusEl.className = "badge";
        renderTop(data.top);
      } catch (err) {
        statusEl.textContent = "识别失败";
        showError(err.message || "识别失败");
      } finally {
        uploadPredictBtn.disabled = false;
      }
    }

    personSelect.addEventListener("change", () => loadImages(personSelect.value).catch(err => showError(err.message)));
    imageSelect.addEventListener("change", () => loadImage(personSelect.value, imageSelect.value).catch(err => showError(err.message)));
    randomBtn.addEventListener("click", () => randomImage().catch(err => showError(err.message)));
    predictBtn.addEventListener("click", () => predict());
    uploadInput.addEventListener("change", () => {
      const file = uploadInput.files && uploadInput.files[0];
      if (file) showUploadedImage(file);
    });
    uploadPredictBtn.addEventListener("click", () => predictUploaded());

    loadPeople().catch(err => showError(err.message));
  </script>
</body>
</html>"""


def load_names() -> list[str]:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return list(summary["target_names"])


def person_to_dir(name: str) -> str:
    return name.replace(" ", "_")


def safe_person_dir(name: str) -> Path:
    path = DATA_DIR / person_to_dir(name)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"找不到人物目录：{name}")
    return path


def list_images(name: str) -> list[str]:
    person_dir = safe_person_dir(name)
    return sorted(
        p.name for p in person_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_image_path(name: str, filename: str) -> Path:
    person_dir = safe_person_dir(name)
    path = (person_dir / filename).resolve()
    if person_dir.resolve() not in path.parents:
        raise ValueError("非法图片路径")
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise FileNotFoundError(f"找不到图片：{filename}")
    return path


def image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = center_crop_square(image)
    image = image.resize((160, 160), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = fixed_image_standardization(tensor)
    return tensor.unsqueeze(0)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float64)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    total = exp_scores.sum()
    if not math.isfinite(total) or total <= 0:
        return np.ones_like(scores) / len(scores)
    return exp_scores / total


class FaceRecognizer:
    def __init__(self) -> None:
        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"找不到分类器模型：{CLASSIFIER_PATH}")
        if not SUMMARY_PATH.exists():
            raise FileNotFoundError(f"找不到结果摘要：{SUMMARY_PATH}")

        self.names = load_names()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()
        self.classifier = joblib.load(CLASSIFIER_PATH)

    @torch.no_grad()
    def predict_image(self, image: Image.Image, true_name: str) -> dict:
        tensor = preprocess(image).to(self.device)
        embedding = self.model(tensor)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).cpu().numpy()

        pred = int(self.classifier.predict(embedding)[0])
        decision = self.classifier.decision_function(embedding)
        scores = decision[0] if decision.ndim == 2 else np.asarray(decision)
        probs = softmax(scores)
        order = np.argsort(probs)[::-1]
        top = [
            {"name": self.names[int(idx)], "score": float(probs[int(idx)])}
            for idx in order[:3]
        ]
        predicted_name = self.names[pred]
        return {
            "true_name": true_name,
            "predicted_id": pred,
            "predicted_name": predicted_name,
            "is_correct": predicted_name == true_name,
            "confidence": float(probs[pred]),
            "top": top,
        }

    @torch.no_grad()
    def predict(self, image_path: Path, true_name: str) -> dict:
        image = Image.open(image_path)
        return self.predict_image(image, true_name)


RECOGNIZER = FaceRecognizer()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        print("%s - %s" % (self.address_string(), format % args))

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/":
                self._send_html()
                return

            if path == "/people":
                people = [
                    {"name": name, "count": len(list_images(name))}
                    for name in RECOGNIZER.names
                ]
                self._send_json({"people": people, "device": str(RECOGNIZER.device)})
                return

            if path == "/images":
                person = query.get("person", [""])[0]
                self._send_json({"person": person, "images": list_images(person)})
                return

            if path == "/image":
                person = query.get("person", [""])[0]
                filename = query.get("file", [""])[0]
                image_path = safe_image_path(person, filename)
                self._send_json({
                    "person": person,
                    "file": filename,
                    "image": image_data_url(image_path),
                })
                return

            if path == "/random":
                person = query.get("person", [""])[0]
                images = list_images(person)
                if not images:
                    raise FileNotFoundError("该人物没有可用图片")
                filename = random.choice(images)
                self._send_json({"person": person, "file": filename})
                return

            if path == "/predict":
                person = query.get("person", [""])[0]
                filename = query.get("file", [""])[0]
                image_path = safe_image_path(person, filename)
                self._send_json(RECOGNIZER.predict(image_path, person))
                return

            self._send_json({"error": "Not found"}, 404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path != "/upload_predict":
                self._send_json({"error": "Not found"}, 404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            image_data = payload.get("image", "")
            if not image_data:
                raise ValueError("没有收到上传图片")
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]

            raw = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            result = RECOGNIZER.predict_image(image, "本地上传照片")
            result["note"] = "上传照片不一定属于训练集人物，结果表示在 12 个已知类别中最相似的人。"
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("训练集样本识别网页已启动")
    print(f"请在浏览器打开：http://{HOST}:{PORT}")
    print("按 Ctrl+C 停止服务")
    server.serve_forever()


if __name__ == "__main__":
    main()
