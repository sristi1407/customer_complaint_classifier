# Customer Complaint Classification

This project focuses on classifying consumer complaints into different product categories using machine learning and NLP techniques.

The goal is to automatically predict the type of complaint (e.g., Credit Card, Mortgage, Loan) based on the issue description, helping improve customer service and complaint routing systems.


## Problem Statement

Given a complaint with:
- Issue  
- Sub-issue  

Predict the **Product category** it belongs to.

This is a **multi-class classification problem** with real-world challenges like:
- Imbalanced dataset  
- Sparse text features  
- Many categories  

---

## Dataset

- Source: Consumer Complaint Database (Kaggle)
- Total records: ~300K+ complaints  
- Target variable: `Product`  
- Input features:
  - `Issue`
  - `Sub-issue`


## Key Insight

The dataset is **highly imbalanced**, where a few categories (like Mortgage and Debt Collection) dominate most of the complaints.

This makes evaluation more challenging and requires careful metric selection.


## Project Pipeline
Raw Data → Cleaning → TF-IDF → Model Training → Evaluation → UI Prediction


### Data Preprocessing
- Removed missing values  
- Combined `Issue` + `Sub-issue`  
- Text cleaning (lowercase, stopword removal)

### Feature Engineering
- Used **TF-IDF Vectorization**
- Limited to **5000 features** for efficiency


## Models Used

We trained and compared multiple models:

- Baseline Model (Majority Class)
- Naive Bayes (default + tuned)
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

## Results

| Model                  | Accuracy |
|------------------------|----------|
| Decision Tree          | ~97.8%   |
| Logistic Regression    | ~97.7%   |
| Random Forest          | ~97.7%   |
| SVM                    | ~96.9%   |
| Naive Bayes (tuned)    | ~96.8%   |
| Baseline               | ~21.6%   |

 **Decision Tree and Logistic Regression performed best overall**

## Evaluation Metrics

We used multiple metrics because the dataset is imbalanced:

- Accuracy  
- Precision  
- Recall  
- F1-score  

## What We Learned

- Real-world datasets are often **imbalanced**
- Accuracy alone can be misleading  
- Some classes are harder to predict than others  

## Future Improvements

- Handle imbalance using:
  - Class weights  
  - SMOTE  
- Try deep learning models:
  - BERT  
  - Transformers  


## User Interface (UI)

We built a simple UI using **Streamlit** to make predictions.

Users can:
- Enter a complaint text  
- Get predicted product category instantly  

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- TF-IDF  
- Streamlit  


## Project Structure
```
customer_complaint_classifier/
│
├── data/                    # Dataset files
├── notebooks/               # Model training notebooks
├── models/                  # Saved models
├── results/                 # Outputs and metrics
├── streamlit_ui/            # UI application
└── README.md
```

## Conclusion

We successfully built a machine learning pipeline that classifies customer complaints with high accuracy (~97–98%).

This project demonstrates how NLP + ML can be used in real-world applications like automated customer support systems.

## Authors

- Sristi Prasad  
- Kunal Ghanwat  
- Dev Patel  