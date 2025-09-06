# 🚕 Predicting Uber Ride Cancellations with Machine Learning
I created an end-to-end machine learning pipeline using XGBoost on a real-world dataset to predict whether a ride will be cancelled before it even starts, using only booking details like pickup, drop, time, and vehicle type etc.

📌 Problem Statement

Task: Binary classification

Target: target_customer_cancelled (1 = cancelled, 0 = not cancelled)

Dataset is imbalanced (~7% cancellations)
👉 Challenge: train a model that detects cancellations without data leakage and remains interpretable.

📊 Results

ROC-AUC: 0.86

Recall for cancellations: 93% (the model catches almost all cancellations)

The dataset was highly imbalanced (~7% cancellations), so I had to carefully handle that in preprocessing to make the model work well.

🛠 Tech Stack

Python: Data analysis & modeling

Pandas, NumPy, Scikit-learn: Preprocessing

XGBoost: Classification model

Flask: Web framework for deployment

HTML/CSS: Frontend form for user input

📌 Next Steps
I plan to deploy it on Streamlit Cloud / Render so others can try it live, and also add features like surge pricing and past customer-driver history.
