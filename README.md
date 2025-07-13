
# 💳 Credit Default Risk Prediction App

A machine learning-powered web app to predict the **probability of credit card default** using the Taiwan Credit Default dataset. Built with **XGBoost**, **FastAPI**, and a clean **HTML/CSS frontend**, the app shows a risk score from **low to high**, enabling smarter credit decisions.

---

## 🚀 Features

- Logistic Regression (baseline), Random Forest, and XGBoost
- Final model (**XGBoost**): 🎯 F1 Score = 0.76, 📊 Accuracy = 0.77
- Full pipeline: Feature Engineering, Label Encoding, Scaling
- GridSearchCV tuning with Stratified K-Fold
- Real-time prediction with **FastAPI**
- Clean and responsive UI with **HTML/CSS**
- Shows both binary result and default **probability score**

---

## 💻 Tech Stack

- Python (scikit-learn, pandas, xgboost, FastAPI)
- Frontend: HTML, CSS
- Deployment ready: FastAPI + Uvicorn
- Version Control: Git & GitHub

---

## 📈 Dataset

- 📂 Dataset: [UCI - Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- 🧮 30,000 records | 23 features | Classification task
- 🎯 Goal: Predict likelihood of a person defaulting on their credit card based on historical data

---

## 🗂️ Folder Structure

```

credit\_default\_app/
├── main.py              # FastAPI backend
├── models/              # Trained ML model files
├── templates/           # HTML files (UI)
├── static/              # CSS styling
├── screenshots/         # web1.png to web4.png (UI previews)
├── requirements.txt     # Required Python packages
├── README.md            # Project documentation

````

---

## 📷 Screenshots

| Web App UI |
|------------|
| ![Web 1](screenshots/web1.png) |
| ![Web 2](screenshots/web2.png) |
| ![Web 3](screenshots/web3.png) |
| ![Web 4](screenshots/web4.png) |

---

## ⚙️ How to Run Locally

1. **Clone the repo**  
```bash
git clone https://github.com/YOUR_USERNAME/credit-default-predictor.git
cd credit-default-predictor
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
uvicorn main:app --reload
```

4. **Open in your browser**

```
http://127.0.0.1:8000
```

---

## 🧠 Model Summary

* **Models trained**: Logistic Regression (baseline), Random Forest, XGBoost
* **Final model**: Tuned XGBoost (via GridSearchCV)
* **Metrics**:

  * F1 Score: 0.76
  * Accuracy: 0.77
  * ROC AUC: strong performance
* **Preprocessing**:

  * Label Encoding, Scaling
  * Feature Engineering for model improvement

---

## 📦 requirements.txt

```text
pandas
scikit-learn
xgboost
fastapi
uvicorn
jinja2
matplotlib
seaborn
```

---

## 👨‍💻 Author

**Trevin Rodrigo**
🎓 Industrial Statistics & Mathematical Finance undergraduate
💡 Aspiring Data Scientist | Machine Learning Enthusiast
📫 [Connect on LinkedIn](https://www.linkedin.com/in/trevin-rodrigo/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

```


