# Fake News Detection Using Machine Learning

This repository contains a Python project for detecting fake news using machine learning algorithms. The project covers data exploration, preprocessing, feature extraction, model building, evaluation, and prediction on new data.

## Dataset

- Training dataset: `kaggle_fake_train.csv`
- Testing dataset: `kaggle_fake_test.csv`

The datasets contain news titles with labels indicating whether the news is real (`0`) or fake (`1`).

## Project Overview

The project follows these main steps:

1. **Exploring the Dataset**

   - Checking the shape and columns
   - Displaying the first few rows
   - Visualizing the distribution of real vs fake news
2. **Data Cleaning and Preprocessing**

   - Removing unnecessary columns
   - Handling missing values
   - Text preprocessing:
     - Removing non-alphabetic characters
     - Converting to lowercase
     - Removing stopwords
     - Stemming using PorterStemmer
3. **Feature Extraction**

   - Using CountVectorizer with max 5000 features and n-grams (1,3)
   - Creating input features X and labels y
4. **Model Building**

   - Splitting the dataset into training and testing sets (80%-20%)
   - Algorithms used:
     - Multinomial Naive Bayes
     - Logistic Regression
   - Hyperparameter tuning for best accuracy:
     - alpha for Naive Bayes
     - C for Logistic Regression
5. **Model Evaluation**

   - Accuracy, precision, and recall scores
   - Confusion matrix visualization using Seaborn heatmaps
6. **Predictions**

   - Function fake_news() to predict if a news title is real or fake
   - Testing predictions on random samples from the test dataset

## Libraries Used

- numpy
- pandas
- matplotlib
- seaborn
- nltk
- scikit-learn
- re (regular expressions)

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the notebook in Jupyter:

```python
jupyter notebook Fake_News_Detection.ipynb
```

4. Use the fake_news() function to predict on custom news titles.

## Example Usage

```python
sample_news = "Breaking news: Scientists discover a new planet!"
prediction = fake_news(sample_news)
if prediction:
    print("This is FAKE news!")
else:
    print("This is REAL news.")
```

## Author

**Md. Sameer Sayed**

- AI / Machine Learning Enthusiast
- Email: mdsameersayed0@gmail.com
  Readme yet to be documented
