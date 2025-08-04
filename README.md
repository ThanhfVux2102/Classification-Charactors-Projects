# 🧠 MNIST Digit Classifier with TensorFlow

This project demonstrates how to build, train, and evaluate a simple neural network to classify handwritten digits from the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) using TensorFlow and TensorFlow Datasets (TFDS).

---

## 📌 Project Overview

- **Dataset**: MNIST (28x28 grayscale handwritten digit images)
- **Frameworks**: TensorFlow, TensorFlow Datasets
- **Goal**: Train a neural network to achieve high accuracy in digit recognition
- **Approach**: Sequential neural network with one hidden layer (ReLU activation) and softmax output

---

## 📂 Project Structure

| File/Section           | Description                                 |
|------------------------|---------------------------------------------|
| `load_data()`          | Loads MNIST dataset with TFDS               |
| `normalize_img()`      | Scales pixel values to range [0,1]          |
| `model = Sequential()` | Builds a basic dense neural network         |
| `model.fit()`          | Trains the model on the training set        |
| `predictions()`        | Visualizes prediction results with matplotlib |

---

## ⚙️ How it works

### 1. 📥 Load & Preprocess Data
- Download the dataset using `tfds.load()`
- Normalize pixel values to `[0,1]`
- Use `AUTOTUNE`, `.cache()`, `.shuffle()` for optimal pipeline performance

### 2. 🏗️ Model Architecture
```python
Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(128, activation='relu'),
  Dense(10)  # 10 output classes for digits 0–9
])
```

### 3. 🧪 Compile & Train

- Optimizer: `Adam`
- Loss: `SparseCategoricalCrossentropy`
- Metric: `SparseCategoricalAccuracy`

### 4. 🔍 Evaluate & Visualize

A `predictions()` function randomly samples digits from test data and shows:

- The image
- The predicted label vs. true label (color-coded)
