"""
选题4：基于 PyTorch 的人脸识别系统

方法：
1. 使用 sklearn 下载/读取 LFW 数据集。
2. 使用 facenet-pytorch 的 InceptionResnetV1(vggface2) 提取 512 维人脸特征。
3. 使用 SVM 分类器完成人脸身份识别，并保存报告、混淆矩阵和模型文件。

重要修正：
sklearn 的 fetch_lfw_people 默认返回 0~1 的 float32 图像。旧脚本直接 astype(uint8)
会把图像压成接近全黑，导致 FaceNet 特征失效。本脚本会先按像素范围自动缩放到 0~255。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert sklearn LFW float image to a valid 0~255 uint8 RGB image."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def square_pad(pil_image: Image.Image) -> Image.Image:
    """Pad rectangular face crops to square before resizing, avoiding face distortion."""
    width, height = pil_image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(pil_image, ((side - width) // 2, (side - height) // 2))
    return canvas


class LFWDataset(Dataset):
    def __init__(self, images: np.ndarray):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = to_uint8_image(self.images[idx])
        pil_image = Image.fromarray(image).convert("RGB")
        pil_image = square_pad(pil_image).resize((160, 160), Image.BILINEAR)

        arr = np.asarray(pil_image).astype(np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return fixed_image_standardization(tensor)


@torch.no_grad()
def extract_embeddings(
    model: InceptionResnetV1,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = LFWDataset(images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        embeddings = model(batch)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        chunks.append(embeddings.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def plot_confusion_matrix(cm: np.ndarray, names: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_facenet.png", dpi=220)
    plt.close(fig)


def save_embedding_preview(images: np.ndarray, labels: np.ndarray, names: np.ndarray, out_dir: Path) -> None:
    count = min(12, len(images))
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for ax, image, label in zip(axes.ravel(), images[:count], labels[:count]):
        ax.imshow(to_uint8_image(image))
        ax.set_title(str(names[label]), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "lfw_samples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LFW FaceNet + SVM 人脸识别训练脚本")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./outputs_lfw_facenet")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--min_faces", type=int, default=70)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备:", device)
    if torch.cuda.is_available():
        print("GPU名称:", torch.cuda.get_device_name(0))

    print("正在读取 LFW 数据集...")
    lfw = fetch_lfw_people(
        data_home=args.data_dir,
        min_faces_per_person=args.min_faces,
        resize=0.5,
        color=True,
        download_if_missing=True,
    )

    images = lfw.images
    labels = lfw.target
    names = lfw.target_names

    print("图片数量:", len(images))
    print("类别数量:", len(names))
    print("类别名称:", names.tolist())
    print("像素范围:", float(images.min()), "~", float(images.max()))

    save_embedding_preview(images, labels, names, out_dir)

    print("正在加载 FaceNet 预训练模型...")
    model = InceptionResnetV1(pretrained="vggface2").to(device)

    print("正在提取 FaceNet 人脸特征...")
    embeddings = extract_embeddings(model, images, args.batch_size, device)
    print("特征矩阵形状:", embeddings.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    print("正在训练 SVM 分类器...")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced")),
        ]
    )
    param_grid = {
        "svm__C": [0.3, 1, 3, 10, 30, 100, 300],
        "svm__gamma": ["scale", 0.0003, 0.001, 0.003, 0.01, 0.03],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    clf = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    print("最佳参数:", clf.best_params_)
    print("交叉验证最佳准确率:", clf.best_score_)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== 最终测试结果 ==========")
    print("Best Test Accuracy:", round(acc, 4))
    print(report)

    plot_confusion_matrix(cm, names, out_dir)
    joblib.dump(clf, out_dir / "facenet_svm_classifier.pkl")
    np.save(out_dir / "lfw_facenet_embeddings.npy", embeddings)

    result = {
        "topic": "选题4：基于 PyTorch 的人脸识别系统",
        "method": "FaceNet pretrained vggface2 + SVM",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "num_images": int(len(images)),
        "num_classes": int(len(names)),
        "target_names": names.tolist(),
        "pixel_min": float(images.min()),
        "pixel_max": float(images.max()),
        "embedding_shape": list(embeddings.shape),
        "best_params": clf.best_params_,
        "cv_best_score": float(clf.best_score_),
        "test_accuracy": float(acc),
    }

    with open(out_dir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(out_dir / "result_report.txt", "w", encoding="utf-8") as f:
        f.write("选题4：基于 PyTorch 的人脸识别系统\n")
        f.write("方法：FaceNet 预训练模型提取人脸特征 + SVM 分类器\n")
        f.write("关键修正：将 LFW 的 0~1 浮点图像正确还原到 0~255，避免图像近似全黑。\n\n")
        f.write(f"设备: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"图片数量: {len(images)}\n")
        f.write(f"类别数量: {len(names)}\n")
        f.write(f"类别名称: {names.tolist()}\n")
        f.write(f"像素范围: {float(images.min()):.4f} ~ {float(images.max()):.4f}\n")
        f.write(f"特征维度: {embeddings.shape}\n")
        f.write(f"最佳参数: {clf.best_params_}\n")
        f.write(f"交叉验证最佳准确率: {clf.best_score_:.4f}\n")
        f.write(f"测试集准确率: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("\n训练完成，结果保存在:", out_dir.resolve())
    print("重要文件:")
    print("1. facenet_svm_classifier.pkl      分类器模型")
    print("2. lfw_facenet_embeddings.npy      FaceNet 人脸特征")
    print("3. result_report.txt               最终结果报告")
    print("4. result_summary.json             结构化结果")
    print("5. confusion_matrix_facenet.png    混淆矩阵图")
    print("6. lfw_samples.png                 数据样例图")


if __name__ == "__main__":
    main()
