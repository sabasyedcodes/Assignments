# Job Recommendation System

## 📌 Project Overview
This project focuses on building a **Job Recommendation System** that suggests relevant job opportunities based on user preferences, skills, and past behavior. We implemented **collaborative filtering, content-based filtering, and a hybrid approach** to improve recommendation quality. Additionally, we developed a **classification model** to categorize jobs using machine learning techniques.

---

## 🗂️ Dataset
We used a dataset containing various job-related attributes such as:
- **Job Title**
- **Job Description**
- **Required Skills**
- **User-Job Interactions** (for collaborative filtering)

---

## 🏗️ Steps Followed

### 1️⃣ Data Preprocessing
- **Cleaning & Handling Missing Data**: Processed missing values and ensured data consistency.
- **Feature Extraction**:
  - **TF-IDF Vectorization**: Converted job descriptions and skills into numerical vectors for content-based filtering.
  - **User-Job Interaction Matrix**: Prepared data for collaborative filtering.

### 2️⃣ Collaborative Filtering
- **Used user-item interaction data** to recommend jobs similar to those liked by other users.
- **Implemented memory-based collaborative filtering** using the Surprise library.

### 3️⃣ Content-Based Filtering
- Recommended jobs based on their descriptions, skills, and qualifications.
- Used **TF-IDF vectors** and **cosine similarity** to find similar jobs.

### 4️⃣ Hybrid Recommendation System
- Combined both **collaborative filtering and content-based filtering** to improve recommendation accuracy.

### 5️⃣ Job Classification Model
- **Extracted Features**: Used **TF-IDF vectors** from job descriptions.
- **Trained a Random Forest Classifier** to categorize jobs.
- **Model Evaluation**:
  - Achieved **98.04% cross-validation accuracy**.
  - Used **classification report (precision, recall, F1-score)** for detailed evaluation.

### 6️⃣ Model Evaluation
- **Checked accuracy & classification report**.
- Ensured the model generalizes well with cross-validation.
- Verified performance across different job categories.

---

## 📊 Results
- **Final Accuracy:** `98.04%`
- **High performance on majority classes**, but **minority classes need improvement**.
- **Balanced trade-off between recall and precision**.

---

## 🚀 Next Steps
- **Fine-tune hyperparameters** (e.g., `n_estimators`, `max_depth`).
- **Balance minority classes** using techniques like oversampling or SMOTE.
- **Deploy the model** for real-world job recommendations.
- **Feature Importance Analysis** to understand which factors influence predictions.

---

## 📌 Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, Surprise, NLTK)
- **Machine Learning** (Collaborative Filtering, Content-Based Filtering, Random Forest)
- **Natural Language Processing (NLP)** (TF-IDF, Cosine Similarity)
- **Evaluation Metrics** (Accuracy, Precision, Recall, F1-score, Cross-Validation)

---

### 🎯 **Conclusion**
This project successfully implemented a **hybrid job recommendation system** and a **classification model** to enhance job search efficiency. The model achieved high accuracy, but further improvements can be made for minority class predictions.

🔹 **Want to improve it further?** Let's explore hyperparameter tuning and feature selection! 🚀

