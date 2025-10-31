***

# Tweet Sentiment Analysis Using Sentiment140 Dataset

## Table of Contents
- Project Overview
- Dataset
- Workflow
- Requirements
- Model Details
- Results
- How to Run
- Demo Links (Optional)
- References

***

## Project Overview
This project analyzes the sentiment of tweets using the Sentiment140 dataset and compares classic machine learning models with deep learning architectures. The goal is to classify tweets as positive or negative and evaluate the effectiveness of various NLP and modeling approaches.

***

## Dataset
- **Sentiment140**: 1.6 million labeled tweets (positive & negative sentiment).
- Source: Twitter API.

***

## Workflow
1. **Data Preprocessing**
   - Cleaning (removing stopwords, special characters, stemming, tokenization)
2. **Feature Extraction**
   - TF-IDF, GloVe word embeddings
3. **Model Training**
   - Logistic Regression, SVM, Random Forest (ML)
   - LSTM (DL)
4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
5. **Visualization**
   - Word clouds, performance comparison charts

***

## Requirements

- Python 3.7+
- Key Libraries:
  - NumPy
  - Pandas
  - Matplotlib, Seaborn
  - Scikit-learn
  - TensorFlow, Keras, PyTorch
  - Statsmodels
  - NLTK

- For demo/app:
  - Streamlit (optional)
  - pickle (for model saving/loading)

***

## Model Details
- **Classic ML models**: Logistic Regression, SVM, Random Forest
- **Deep Learning**: LSTM (with TensorFlow/Keras/PyTorch)
- **Feature extraction**: TF-IDF Vectorizer, GloVe Embeddings

***

## Results
- Achieved 85% accuracy on test data.
- Comparative analysis of ML vs Deep Learning models.
- Visualizations: tweet length distribution, word cloud.

***

## How to Run

1. Clone this repo.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download Sentiment140 dataset.
4. Run preprocessing and feature extraction scripts.
5. Train and evaluate selected model:
   ```bash
   python train.py
   ```
6. Launch demo app:
   ```bash
   streamlit run app.py
   ```

## Demo Links (Optional)
- Live App: [[Streamlit URL]](https://tweet-sentimeter.streamlit.app/
)<img width="2083" height="228" alt="image" src="https://github.com/user-attachments/assets/c13f566a-c78a-4d25-9e6a-d9b689510072" />

- Model Space: 
https://huggingface.co/spaces/akshit-sukhija26/twitter-sentiment-analysis
<img width="3924" height="228" alt="image" src="https://github.com/user-attachments/assets/b4684c1c-0ad6-4d2b-a928-05f9decd9bb8" />


***

## References

- Sentiment140 Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
- Scikit-learn, TensorFlow, Keras Documentation

***
