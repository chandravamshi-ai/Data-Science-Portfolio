# Bigram Language Model for Shakespeare Text

## Overview

This project demonstrates the training of a bigram language model using a corpus of text from William Shakespeare. The goal is to create a simple neural network that can generate text in the style of Shakespeare by learning the statistical properties of bigrams in the provided dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation and Setup](#installation-and-setup)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training Procedure](#training-procedure)
7. [Text Generation](#text-generation)
8. [Results](#results)

## Introduction

Language modeling is a fundamental task in natural language processing (NLP). This project uses a bigram model to generate text by predicting the next character based on the previous character. The model is trained on the works of William Shakespeare, leveraging the patterns and structures found in the text to generate new sequences that mimic Shakespeare's style.

## Dataset

The dataset used for this project is works of William Shakespeare. It can be downloaded using the following command:

```bash
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Installation and Setup

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install dependencies**:
    ```bash
    pip install torch
    ```

3. **Download the dataset**:
    ```bash
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    ```

## Usage

To train the model and generate text, run the following command:
1. **Run the training script**:
    ```bash
      python bigram.py
    ```

## Model Architecture

The bigram language model is a simple neural network with the following key components:
- **Embedding Layer**: Maps each character to a dense vector representation.
- **Forward Pass**: Generates logits for the next character based on the current context.
- **Softmax**: Converts logits to probabilities for the next character.

### Class Definition

The model is defined as follows:

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

## Training Procedure

The training process involves the following steps:
1. **Preprocessing**: The text is tokenized into characters, and mappings from characters to integers (and vice versa) are created.
2. **Batch Generation**: Batches of sequences are generated for training.
3. **Model Training**: The model is trained using cross-entropy loss to predict the next character based on the current context.
4. **Evaluation**: Periodically, the model's performance is evaluated on a validation set.

### Example Training Loop

```python
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## Text Generation

After training, the model can generate text by starting with an initial context and iteratively predicting the next character.

### Example Text Generation

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
generated_text = decode(generated_indices)
print(generated_text)
```

## Results

