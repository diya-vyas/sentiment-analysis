# 🧠 Sentiment Analysis — Amazon Reviews & Beyond

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-green?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>Classify emotions in text — from Amazon product reviews to custom datasets — using classical ML and deep learning approaches.</b>
</p>

---

## 📌 What This Project Does

> **Given a text review, predict whether the sentiment is Positive, Negative, or Neutral.**

This project walks through a complete **NLP pipeline** for sentiment analysis:
- 🔍 Text preprocessing & cleaning
- 📊 Exploratory Data Analysis (EDA)
- 🤖 Model training (VADER, Logistic Regression, LSTM, BERT-based)
- 📈 Evaluation with confusion matrix, ROC curves, and accuracy scores
- 🧪 Real-world testing on Amazon product review data

---

## 🗂️ Project Structure

```
sentiment-analysis/
│
├── 📁 data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Cleaned & tokenized data
│
├── 📁 notebooks/
│   ├── sentiment_analysis.ipynb          # Core analysis notebook
│   ├── amazon_example.ipynb              # Amazon reviews walkthrough
│   └── Untitled0.ipynb                   # Experiments & scratch work
│
├── 📁 src/
│   ├── preprocess.py         # Text cleaning utilities
│   ├── model.py              # Model definitions
│   └── evaluate.py           # Metrics & visualizations
│
├── 📁 results/
│   ├── figures/              # Plots & confusion matrices
│   └── metrics.json          # Saved evaluation results
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook notebooks/
```

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| `Python 3.8+` | Core language |
| `Pandas / NumPy` | Data manipulation |
| `NLTK / spaCy` | Text preprocessing |
| `Scikit-learn` | Classical ML models |
| `TensorFlow / Keras` | Deep learning (LSTM) |
| `Transformers (HuggingFace)` | BERT-based models |
| `Matplotlib / Seaborn` | Visualization |
| `VADER` | Rule-based sentiment scoring |

---

## 📊 Sample Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| VADER (Rule-based) | ~71% | 0.69 |
| Logistic Regression | ~84% | 0.83 |
| LSTM | ~89% | 0.88 |
| BERT (fine-tuned) | ~93% | 0.92 |

> 📌 Results may vary based on dataset split and hyperparameters.

---

## 🗃️ Dataset

This project uses the **Amazon Product Reviews** dataset.

- Source: [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)
- Labels: Positive (⭐⭐⭐⭐⭐), Negative (⭐⭐), Neutral (⭐⭐⭐)
- Preprocessing: lowercasing, punctuation removal, stopword filtering, lemmatization

To use your own dataset, drop a `.csv` with a `text` and `label` column into `data/raw/`.

---

## 💡 Key Concepts Covered

- ✅ Tokenization & stopword removal
- ✅ TF-IDF vectorization
- ✅ Word embeddings (Word2Vec / GloVe)
- ✅ Recurrent Neural Networks (LSTM)
- ✅ Transfer learning with BERT
- ✅ Handling class imbalance
- ✅ Model evaluation & interpretation

---

## 📸 Visualizations

<p align="center">
  <img src="assets/wordcloud.png" width="45%" alt="Word Cloud"/>
  &nbsp;&nbsp;
  <img src="assets/confusion_matrix.png" width="45%" alt="Confusion Matrix"/>
</p>

> 📁 Run the notebooks to auto-generate these plots in the `results/figures/` folder.

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork this repo
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Author

Made with ❤️ by **Diya Vyas**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)

---

<p align="center">⭐ If you found this useful, please star the repo! It helps others discover it.</p>
