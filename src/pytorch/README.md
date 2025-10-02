# ai-study
This repo is for stady

# 🔢 Neural Network Activation & Output Functions

This document provides a comparison of commonly used **activation and output functions** in neural networks, detailing their behavior, usage, performance, and optimization characteristics.

---

## 📊 Comparison Table

| **Function Name** | **Specific Property** (e.g., Sum, Range)       | **When to Use**                                  | **Performance Notes**                          | **Optimization Behavior**                         |
|------------------|--------------------------------------------------|--------------------------------------------------|------------------------------------------------|----------------------------------------------------|
| `softmax`        | Output ∈ (0, 1); **sum = 1**                     | Output layer of **multi-class classification**   | More expensive (exp & sum); vector-level op     | Enables cross-entropy loss; gradients can vanish if values are too sharp |
| `sigmoid`        | Output ∈ (0, 1); not guaranteed to sum to 1     | Output for **binary classification**             | Simple but can saturate                         | Prone to **vanishing gradients** for large |x|     |
| `tanh`           | Output ∈ (–1, 1); **symmetric around 0**        | Hidden layers (RNNs, when zero-centering helps)  | Better than sigmoid for zero-centered outputs   | Also prone to **vanishing gradients**             |
| `ReLU`           | Output ∈ [0, ∞); **not bounded or normalized**  | Default for most hidden layers                   | Fast, sparse activation (0s for negative input) | Not differentiable at 0, but works well in practice |
| `Leaky ReLU`     | Like ReLU, but slope for x<0 (e.g. 0.01x)       | Same as ReLU, helps with **“dying ReLU”**        | Slightly more compute than ReLU                 | Prevents zero gradients for x<0                   |
| `ELU`            | Smooth curve, output ∈ (–1, ∞)                  | Can be used in place of ReLU                     | Smoother than ReLU, but slower                  | Better learning dynamics in some networks         |
| `GELU`           | Gaussian-based, smoother than ReLU              | Used in **Transformers** (e.g. BERT)             | More expensive than ReLU                        | Empirically performs well in large models         |
| `identity`       | Output = input                                  | Useful for **linear output layers**              | No transformation; very fast                    | Only useful when no non-linearity is needed       |
| `log_softmax`    | `log(softmax(x))`; sum(log probs) ≠ 0           | Input to **NLLLoss** for numerical stability     | More stable than separate log + softmax         | Same gradients as softmax with log; faster numerics |

---

## ✅ Quick Tips

- Use **ReLU** in hidden layers unless you have a specific reason to change.
- Use **softmax** only in output layers for **multi-class classification**.
- Use **sigmoid** for **binary classification** or individual probability outputs.
- For deep models like Transformers, prefer **GELU** if supported.
- Use **log_softmax** when followed by `NLLLoss` for **numerical stability**.

---

## 📘 References

- [PyTorch Activation Functions](https://pytorch.org/docs/stable/nn.functional.html)
- [Understanding Activation Functions in Neural Networks – Medium](https://medium.com/@aptrishu/activation-functions-neural-networks-1cbd9f8d91d6)

---

## 🔊 Pre-Trained Audio Models in PyTorch

| Model | Type | Highlights | Best Use Case | Training & Inference Code |
|-------|------|------------|---------------|---------------------------|
| **PANNs** (CNN10 / CNN14 / Wavegram-Logmel-CNN) | CNN | Trained on **AudioSet**, strong embeddings for downstream tasks | General-purpose sound event detection & transfer learning | [GitHub – audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) |
| **AST** (Audio Spectrogram Transformer) | Transformer | High accuracy on ESC-50, AudioSet, FSD50K | Max accuracy on medium/large datasets | [GitHub – YuanGongND/ast](https://github.com/YuanGongND/ast) |
| **PaSST** (Efficient Audio Transformers) | Transformer | Lighter AST variant, efficient for long audios | High accuracy with lower compute demand | [GitHub – kkoutini/PaSST](https://github.com/kkoutini/PaSST) |
| **HTS-AT** (Hierarchical Token-Semantic AT) | Transformer | Hierarchical attention, top leaderboard results | Research & SOTA performance | [GitHub – HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer) |
| **VGGish** | CNN | Classic log-mel feature extractor | Simple feature frontend + custom classifier | [GitHub – torchvggish](https://github.com/harritaylor/torchvggish)<br>[Docs – v-iashin/vggish](https://v-iashin.github.io/video_features/models/vggish/) |
| **Wav2Vec 2.0** | Self-Supervised Encoder | Learns from raw waveforms, powerful transfer learning | Low-label datasets, robust embeddings for speech & sound | [Torchaudio Tutorial](https://docs.pytorch.org/audio/2.3.0/tutorials/speech_recognition_pipeline_tutorial.html)<br>[HF Fine-Tuning Guide](https://huggingface.co/blog/fine-tune-wav2vec2-english) |
| **BC-ResNet** | CNN (compact) | Small footprint, efficient on-device | Edge deployment, low-power devices (KWS, IoT) | [GitHub – bcresnet](https://github.com/Qualcomm-AI-research/bcresnet) |
| **EfficientAT** | CNN (lightweight) | AudioSet-pretrained, optimized for speed | Efficient large-scale tagging, resource-limited training | [GitHub – EfficientAT](https://github.com/fschmid56/EfficientAT) |
| **SSAST** (Self-Supervised AST) | Transformer | Self-supervised AST, transfer friendly | Transfer learning when labeled data is scarce | [GitHub – ssast](https://github.com/YuanGongND/ssast) |
| **SAT** (Streaming Audio Transformers) | Transformer | Low-latency, streaming audio tagging | Real-time audio classification (online/continuous) | [GitHub – SAT](https://github.com/RicherMans/SAT) |


_Last updated: 2025-10-01_

