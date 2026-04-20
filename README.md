# Customer Complaint Classification

## Team Members

* Sristi Prasad – [prasad.s@northeastern.edu]
* Kunal Ghanwat – [ghanwat.k@northeastern.edu]
* Dev Patel – [patel.d@northeastern.edu]

##  Objectives and Significance

This project focuses on classifying consumer complaints into product categories using machine learning and natural language processing (NLP).

Given a complaint’s **issue and sub-issue**, the model predicts the **product category** (e.g., Credit Card, Mortgage, Loan).

### Why is this important?

* Automates complaint routing
* Reduces manual effort in customer support
* Improves response time and service quality


## Summary of Main Findings

* Decision Tree achieved the highest accuracy (~97.8%)
* Logistic Regression performed consistently well (~97.7%)
* Naive Bayes worked well despite being simple
* Dataset imbalance affected minority class predictions
* Accuracy alone is not sufficient for evaluation


## Background

### Key Concepts

* **Text Classification**: Assigning categories to text data
* **TF-IDF**: Converts text into numerical features
* **Multi-class Classification**: More than 2 output classes
* **Class Imbalance**: Some categories have more data than others

### Previous Work

Text classification is widely used in NLP applications like spam detection and sentiment analysis.
Traditional models such as **Naive Bayes and Logistic Regression** are commonly used due to their efficiency.
More advanced approaches include **deep learning models (BERT, LSTM)**.


##  Methods and Project Design

### Dataset

* Source: Consumer Complaint Database (Kaggle)
* Records: ~300,000+ complaints
* Input:

  * Issue
  * Sub-issue
* Output:

  * Product category


###  Data Preprocessing

* Removed missing values
* Combined `Issue + Sub-issue`
* Lowercasing and text cleaning
* Removed stopwords


###  Feature Engineering (TF-IDF)

TF-IDF helps convert text into numbers:

**TF-IDF = TF(t,d) × log(N / DF(t))**

Where:

* TF(t,d): frequency of term t in document d
* DF(t): number of documents containing term t
* N: total documents

We limited features to **5000** for efficiency.


###  Models Used

* Baseline Model (Majority Class)
* Naive Bayes (default + tuned)
* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest


###  Evaluation Strategy

We used:

* Accuracy
* Precision
* Recall
* F1-score

Train-test split: **80% training / 20% testing**


##  Results

| Model               | Accuracy |
| ------------------- | -------- |
| Decision Tree       | ~97.8%   |
| Logistic Regression | ~97.7%   |
| Random Forest       | ~97.7%   |
| SVM                 | ~96.9%   |
| Naive Bayes (tuned) | ~96.8%   |
| Baseline            | ~21.6%   |

### Experiments

All models were trained on the same TF-IDF features and evaluated using the same train-test split.

We compared performance across models to identify the best approach for text classification.


##  User Interface (UI)

We built a simple **Streamlit UI**:

* User enters complaint text
* Model predicts product category
* Real-time output



##  Project Structure

```
customer_complaint_classifier/
│
├── data/                  # Dataset files
├── notebooks/             # Model training notebooks
├── models/                # Saved trained models
├── results/               # Outputs and evaluation metrics
├── streamlit_ui/          # Streamlit UI
│   └── app.py
│
└── README.md
```


## Conclusion

We successfully built a machine learning system that classifies customer complaints with high accuracy (~97–98%).

Decision Tree and Logistic Regression performed best due to their ability to capture patterns in text data.

We also learned that:

* Real-world data is often imbalanced
* Multiple evaluation metrics are important
* Simpler models can perform surprisingly well


## Future Work

* Handle imbalance using:

  * Class weights
  * SMOTE
* Try deep learning models:

  * BERT
  * Transformers
* Improve performance on rare categories


##  References

* Scikit-learn Documentation: https://scikit-learn.org
* Kaggle Dataset: Consumer Complaint Database
* TF-IDF: https://en.wikipedia.org/wiki/Tf–idf
* Streamlit: https://streamlit.io

### AI Assistance

We used AI tools (ChatGPT) for:
- Assisting in designing the Streamlit UI form

Example prompts used:
- "Help design a Streamlit UI for text classification"


