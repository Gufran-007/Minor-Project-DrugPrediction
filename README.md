#### 🧠 DrugNet — Deep Learning Drug Effectiveness Prediction

#### 📦 Dataset  : UCI ML Drug Review (161K train / 54K test)

#### ⚙️ Model    : PyTorch feedforward DNN — 5000→512→256→128→2

#### 🔤 Features : TF-IDF vectorization (5,000 vocab features)

#### 📊 Results  : 92.69% test accuracy · F1-score 92.58%

#### 🌐 Deploy   : Flask web app — real-time review prediction

#### 🐍 Stack    : Python 3.12 · PyTorch · scikit-learn · NLTK · Flask

# DrugNet 💊
Predicts whether a drug is effective or not — just from a patient review.

# How to run it
## 1. clone the repo
git clone ```https://github.com/Gufran-007/Minor-Project-DrugPrediction.git```
cd drugnet

## 2. install dependencies
```pip install -r requirements.txt```

## 3. download dataset from Kaggle → put CSVs in /data folder or it is already uploaded, use that
```kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018```

## 4. preprocess (creates the TF-IDF feature matrices)
```python preprocess.py```

## 5. train the model
```python train.py```

## 6. evaluate on test set
```python evaluate.py```

## 7. launch the web app
```python app.py```
#### open → ```http://127.0.0.1:5000```
🐍 Python 3.12
🔥 PyTorch
🌐 Flask
📊 scikit-learn
92.69% accuracy
