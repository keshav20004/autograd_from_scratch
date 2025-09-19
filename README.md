# ⚡ Autograd from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
A minimal **automatic differentiation engine** built from scratch, inspired by [PyTorch](https://pytorch.org/) and [micrograd](https://github.com/karpathy/micrograd).  

This project demonstrates how to implement **computational graphs** and **backpropagation** without relying on external ML frameworks.

---

## ✨ Features
- Lightweight autograd engine in <200 lines of Python.  
- Dynamically builds computational graphs.  
- Supports basic operations: `+`, `-`, `*`, `/`, `tanh`, `exp`, `pow`.  
- Forward and backward pass with chain rule.  
- Clear, educational, and dependency-free.  

---

## 📂 Project Structure
autograd-from-scratch/
│── value.py # Core Value class
│── demo.py # Example usage
│── README.md # Documentation

---

## ⚙️ Installation
Clone the repo:
```bash
git clone https://github.com/yourusername/autograd-from-scratch.git
cd autograd-from-scratch
from value import Value

# Forward pass
x = Value(2.0)
y = Value(3.0)
z = x * y + y + x.tanh()

print(f"z = {z.data:.4f}")

# Backward pass
z.backward()
print(f"dz/dx = {x.grad:.4f}")
print(f"dz/dy = {y.grad:.4f}")
z = 7.9640
dz/dx = 3.9640
dz/dy = 3.0000
🧩 How It Works

Each operation creates a new Value node with references to its parents.

Calling .backward() recursively applies the chain rule.

Gradients accumulate in each node’s .grad attribute.

Graph example:

     x ---- (*)
          /     \
         /       \
        y         (+) ---- z
        |             \
        └-------------- y

📖 Learning Goals

Understand computational graphs.

Grasp the chain rule in backpropagation.

See how PyTorch/TensorFlow do gradient tracking under the hood.

🔮 Future Improvements

Add more ops (ReLU, sigmoid, softmax).

Implement a simple neural net using this engine.

Graph visualization of computational graph.

📚 References



CS231n Notes on Backprop
