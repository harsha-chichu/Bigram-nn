# 🧠 Bigram Neural Network (Character-Level)

A minimal character-level Bigram Language Model implemented in PyTorch.  
This project is inspired by Andrej Karpathy’s "makemore" series and is designed to teach the fundamentals of language modeling using a simple neural network.

---

## 📚 Description

This notebook builds a neural network that learns to generate names character-by-character using bigram probabilities (i.e., it learns the likelihood of one character following another).

- Trains on a list of names (provided in `names.txt`)
- One linear layer (27x27 for 26 letters + ".")
- Learns to predict the next character given the current one
- Generates new names using sampling

---

## 🛠️ Features

- Character-level tokenizer
- Vocabulary and one-hot encoding
- Manual gradient descent loop with autograd
- Softmax + cross-entropy loss
- Name generation via learned bigram model

---

## 📝 Example Output

Generated names (after training):

```
Ceron.
Loma.
Rax.
Deren.
```

---

## 📁 Project Structure

```
Bigram-nn/
│
├── BigramNN.ipynb         # Main notebook with code and explanations
├── names.txt              # Input data (list of names, one per line)
└── README.md              # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/harsha-chichu/Bigram-nn.git
cd Bigram-nn
```

### 2. Install dependencies

Make sure you have Python 3 and PyTorch installed.

```bash
pip install torch matplotlib
```

### 3. Run the notebook

Launch Jupyter Notebook or VSCode and open:

```bash
BigramNN.ipynb
```

Train the model and try generating names at the end!

---

## 🧪 Sample Code Snippet

```python
xenc = F.one_hot(x, num_classes=vocab_size).float()
logits = xenc @ W
loss = F.cross_entropy(logits, y)
```

---

## 🧠 Concepts Covered

- One-hot encoding
- Language modeling
- Bigram probabilities
- Gradient descent
- PyTorch autograd

---

## 💡 Future Improvements

- Add an embedding layer instead of one-hot
- Introduce a hidden layer (MLP)
- Use batching and GPU support
- Add word-level bigrams
- Train on a larger corpus

---

## 🧑‍💻 Author

**Harsha Vardhan**  
[GitHub Profile](https://github.com/harsha-chichu)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
