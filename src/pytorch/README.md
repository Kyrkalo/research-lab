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

_Last updated: 2025-06-07_

