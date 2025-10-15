# Facial Emotion Recognition using Learning Classifier Systems (LCS)

This project implements a **Facial Emotion Recognition (FER)** system using a **Learning Classifier System (LCS)** and compares its performance with conventional machine learning models such as **SVM**, **Random Forest**, and **MLP**.  
The goal is to balance **accuracy** and **explainability**, highlighting how rule-based learning systems like LCS can provide interpretable models for emotion recognition.

---

## 🧠 Abstract

Facial Emotion Recognition (FER) plays a crucial role in affective computing and human-computer interaction. This study employs a **Learning Classifier System (LCS)** to classify facial emotions from grayscale images using **Local Binary Patterns (LBP)** and **Histogram of Oriented Gradients (HOG)** features.  
A combined dataset was created from two public sources—**FER-2013** and **Facial Emotion Recognition Dataset**—to enhance diversity and robustness.  
The proposed LCS achieved interpretable rule sets while maintaining competitive accuracy compared to traditional models.

---

## 📁 Project Structure

```bash
├── train/ # Training dataset (8.6 MB)
├── val/ # Validation dataset (1.9 MB)
├── test/ # Test dataset (2.8 MB)
├── facial_emotion_recognition_lcs.ipynb # Main code file
└── README.md
```


---

## 🧠 Project Overview

### 🔹 Objective
To develop a **Learning Classifier System (LCS)** model for recognizing facial emotions from images and compare its performance with traditional machine learning algorithms in terms of:
- Accuracy  
- F1-score  
- Precision & Recall  
- Interpretability (rule-based reasoning)

### 🔹 Key Contributions
- Creation of a **combined custom dataset** using two publicly available datasets.
- Feature extraction using **Histogram of Oriented Gradients (HOG)** and **Local Binary Patterns (LBP)**.
- Quantitative evaluation using a reproducible and explainable machine learning workflow.
- Demonstration of the interpretability advantages of LCS compared to conventional classifiers.

---

## 📊 Dataset

The dataset used in this project is a **custom combination** of two publicly available facial emotion datasets:

1. [**Facial Emotion Recognition Dataset**](https://www.kaggle.com/datasets/rohulaminlabid/facial-emotion-recognition-dataset) by *Rohul Amin Labid*  
2. [**FER-2013 Dataset**](https://www.kaggle.com/datasets/msambare/fer2013/data) by *M. Sambare*

The two datasets were merged to increase diversity, reduce class imbalance, and improve the robustness of emotion classification.  

Each image is:
- Converted to **grayscale**
- Resized to **48×48 pixels**
- Transformed into **HOG + LBP feature vectors**
- Normalized using `StandardScaler`

**Dataset Split:**
| Split   |  Purpose          | Size    |    Description    |
|:------  |:---------         |:------: |:-------------     |
| `train/`| Model training    | 8.6 MB  | 80% of total data |
| `val/`  | Validation        | 1.9 MB  | 10% of total data |
| `test/` | Final evaluation  | 2.8 MB  | 10% of total data |

**Total size:** ≈ 13.3 MB  
**Classes:** 7 (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)

---

## ⚙️ Methodology

### 1️⃣ Preprocessing
- Conversion to grayscale  
- Image resizing to **48×48 pixels**  
- Feature extraction using:
  - **Local Binary Pattern (LBP)**
  - **Histogram of Oriented Gradients (HOG)**
- Concatenation of features (`np.hstack`)
- Normalization with `StandardScaler`


### 2️⃣ Model Training
 Model Training
- Implemented **Learning Classifier System (XCS variant)** with rule-based evolution.
- Compared with:
  - **Linear SVM**
  - **Random Forest**
  - **MLP Classifier**
- Evaluation using accuracy, F1-score, precision, recall, and confusion matrix.

### 3️⃣ Evaluation
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---


## 🧪 Experimental Results Summary

### 🔸 XCS Learning Progress

Iteration 100  — Accuracy: 0.2185 — Population: 99
Iteration 1000 — Accuracy: 0.3322 — Population: 919
Iteration 2000 — Accuracy: 0.4563 — Population: 1696

### 🔸 Model Performance Comparison

#### **Linear SVM**
| Metric    | Accuracy   | Precision | Recall  | F1-score |
|:-------   |:----------:|:---------:|:-------:|:--------:|
| Overall   | **0.4804** | 0.49      | 0.48    |    0.48  |

#### **Random Forest**
| Metric | Accuracy     | Precision | Recall | F1-score |
|:-------|:------------:|:-------:  |:------:|:--------:|
| Overall | **0.4600**  | 0.46      | 0.46   | 0.45     |

#### **MLP Classifier**
| Metric | Accuracy     | Precision | Recall | F1-score |
|:-------|:------------:|:----------:|:-----:|:--------:|
| Overall | **0.5349**  | 0.54      | 0.53   | 0.54     |

#### **XCS (LCS Model)**
| Metric     | Accuracy          | Interpretability |
|:-------    |:---------------:  |:----------------:|
| **0.4563** | High (rule-based) | 1696 rules       |

---

## 💡 Critical Reflection

1. The **XCS model** successfully evolved interpretable rules for facial emotion recognition.  
2. **HOG + LBP** proved effective in preserving emotional texture and edge cues.  
3. Compared to SVM and MLP, **LCS** provides **better explainability** through IF–THEN rules, though training time was longer.  
4. The model struggled slightly with **subtle emotions** (e.g., Neutral vs. Sad).  
5. Further optimization is needed to enhance **generalization and population stability**.

---

## 🔭 Future Work

1. Explore hybrid approaches combining **XCS with deep learning (CNN-based embeddings)**.  
2. Integrate **genetic algorithm optimizations** (e.g., adaptive mutation or rule compression).  
3. Expand to larger datasets such as **FER+** and **AffectNet** for scalability testing.  
4. Conduct **human-in-the-loop evaluations** on interpretability and trust.  

---


## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/facial-emotion-recognition-lcs.git
cd facial-emotion-recognition-lcs
```

### 1️⃣ Clone the Repository
```bash
pip install opencv-python scikit-image scikit-learn matplotlib seaborn xgboost tqdm
```

### 3️⃣ Run the Notebook
```bash

jupyter notebook facial_emotion_recognition_lcs.ipynb
```
---

## 📚 References
Amin, R. (2023). Facial Emotion Recognition Dataset. Kaggle.
https://www.kaggle.com/datasets/rohulaminlabid/facial-emotion-recognition-dataset

Sambare, M. (2023). FER-2013 Dataset. Kaggle.
https://www.kaggle.com/datasets/msambare/fer2013/data

## 🧾 License & Acknowledgements
1. The datasets are provided under Kaggle Open License for educational and research purposes.
2. Project developed as part of academic coursework on explainable machine learning.
3. Implementation and experimentation carried out by the author using Python (scikit-learn, OpenCV, and XCS framework).

👨‍💻 Author
```bash
Nirankar Jaiswar
Master of Information Technology
Whitireia and WelTec, Wellington
```