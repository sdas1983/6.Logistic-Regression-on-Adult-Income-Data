# Logistic Regression on Adult Income Data

This project performs logistic regression analysis on the Adult Income dataset to predict whether an individual's income exceeds $50K/year. The dataset is sourced from the [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult) and the analysis is guided by a [Kaggle Income Predictor](https://www.kaggle.com/code/kartik1trivedi/income-predictor-logistic-regression).

## Table of Contents

- [Data Overview](#data-overview)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results and Visualization](#results-and-visualization)
- [License](#license)

## Data Overview

The dataset consists of two files:
- `adult.data`: Contains the training data.
- `adult.test`: Contains the test data.

### Features
- `age`
- `workclass`
- `fnlwgt`
- `education`
- `education-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`
- `income` (Target Variable)

## Data Cleaning and Preparation

1. **Data Reading**: Load training and test datasets, combining them into a single DataFrame.
2. **Missing Values**: Check for and handle missing values by replacing "?" with NaN and imputing with the most frequent values.
3. **Data Cleaning**: Clean the `income` column by stripping whitespace and trailing periods.
4. **Encoding**: Convert categorical features to numeric labels using `LabelEncoder`.

## Model Training and Evaluation

1. **Feature and Target Selection**: Prepare feature matrix `X` and target vector `y` by excluding unnecessary columns.
2. **Data Splitting**: Split the data into training and testing sets.
3. **Standardization**: Scale features using `StandardScaler`.
4. **Model Training**: Train a `LogisticRegression` model.
5. **Model Evaluation**: Evaluate the model using accuracy scores, confusion matrix, classification report, and ROC curves.

## Results and Visualization

- **Model Scores**: Displays test and train scores.
- **Confusion Matrix**: Shows the performance of the model in classifying the test data.
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **ROC Curve**: Visualizes the ROC curve for evaluating the model's performance.
- **Feature Correlation Heatmap**: Displays the correlation between features in the dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
