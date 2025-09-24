# AI Study Project – PyTorch → ONNX → C# / Blazor

## 📌 Why I Started This Project
This project is a personal study initiative to:
- Learn **PyTorch** by building models from scratch (CNNs, R-CNN, DCGAN, GPT-like, etc.).
- Understand the **full ML workflow**: dataset preparation → model training → evaluation → export → deployment.
- Explore **ONNX** as a bridge between research (Python) and enterprise (C# / .NET).
- See how trained AI models can be used in **Blazor applications** and integrated into production systems.

The goal is to move beyond just experimenting in notebooks and actually bring ML into **enterprise-ready environments**.

---

## 🔄 Workflow

1. **Train in Python**
   - Define model, dataset, and trainer in `src/pytorch`.
   - Train locally or in a Jupyter notebook.
   - Save weights as `.pth`.

2. **Export to ONNX**
   - Use custom exporter classes (e.g., `RCNNExportOnnx`).
   - Store exported `.onnx` in external storage (Azure Blob, GitHub Releases, Hugging Face Hub, etc.).

3. **Consume in .NET**
   - During build, CI downloads the ONNX model.
   - Models are copied into `bin/publish/models`.
   - C# backend uses **ONNX Runtime** for inference.
   - Blazor pages call inference APIs and display results.

4. **Deploy**
   - The publish folder contains all DLLs and ONNX models.
   - Deployment is manual or automated (planned with CI/CD).

