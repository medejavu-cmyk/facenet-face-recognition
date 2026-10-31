# FaceNet 人脸身份识别

作者 medejavu-cmyk 独立完成，含 AI 辅助。使用 FaceNet 预训练模型提取人脸特征，以 SVM 在 LFW 筛选子集上进行封闭集身份分类，提供训练评估与本地网页展示。

## 本地运行

在项目目录使用 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_face_lfw_facenet.py --data_dir ./data --output_dir ./outputs_lfw_facenet_50 --batch_size 64 --min_faces 50
.\.venv\Scripts\python.exe web_face_app.py
```

访问 http://127.0.0.1:8000 。首次训练需要下载 LFW 数据及预训练权重。训练输出目录与网页读取目录统一为 `outputs_lfw_facenet_50`。

原报告记录筛选子集测试准确率 100%，本候选未重新训练验证；此结果不代表真实场景或陌生人物的识别能力。

## 来源与状态

依赖 PyTorch、facenet-pytorch、scikit-learn 等第三方库，采用 VGGFace2 预训练模型；各组件与数据集分别遵循其许可和使用条款。候选不包含人脸照片、特征向量、分类器或训练输出。项目许可证和可复现依赖版本尚待补齐；当前只完成源码整理，未验证安装及运行。
