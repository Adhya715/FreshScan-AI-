# 🔍 FreshScan AI

An AI-powered fruit freshness classifier built using **TensorFlow**, **MobileNetV2**, and **Streamlit**.

FreshScan AI classifies an uploaded fruit image into one of six categories:

- 🍎 Apple Fresh
- 🍎 Apple Rotten
- 🍌 Banana Fresh
- 🍌 Banana Rotten
- 🍓 Strawberry Fresh
- 🍓 Strawberry Rotten

The project uses **Transfer Learning** with MobileNetV2 to achieve **99.12% test accuracy**.

---

## Features

- Image preprocessing pipeline using TensorFlow
- Efficient tf.data Dataset pipeline
- Transfer Learning with MobileNetV2
- 6-class image classification
- Streamlit web application
- Confidence score for predictions

---

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- NumPy

---

## Project Structure

```
FreshScan-AI/
│
├── Fruit Freshness Dataset/
│
├── models/
│   └── freshscan_model.keras
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   └── inference/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains images of:

- Fresh Apples
- Rotten Apples
- Fresh Bananas
- Rotten Bananas
- Fresh Strawberries
- Rotten Strawberries

Images are resized to **224 × 224** before training.

---

## Model

Transfer Learning using **MobileNetV2**

Architecture:

- MobileNetV2 (Frozen)
- Global Average Pooling
- Dense (128, ReLU)
- Dropout
- Dense (6, Softmax)

---

## Performance

| Metric | Value |
|---------|--------|
| Test Accuracy | **99.12%** |
| Classes | 6 |
| Image Size | 224×224 |

---

## Running the Project

Clone the repository

```bash
git clone https://github.com/yourusername/FreshScan-AI.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Example

Upload an image

↓

Model predicts

```
Fruit: Banana

Condition: Rotten

Confidence: 99.84%
```

---

## Future Improvements

- Webcam support
- Mobile deployment
- More fruit categories
- Real-time object detection
- Model explainability using Grad-CAM

---

## Author

**Adhya Agrawal**

Engineering Student | AI & Machine Learning Enthusiast

GitHub: https://github.com/Adhya715
