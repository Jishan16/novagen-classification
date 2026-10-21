# NovaGen Health Data Classification

A binary classification project on a health and lifestyle dataset. Five models are trained on the same split and compared on accuracy, precision and recall.

## Dataset

`novagen_dataset.csv` holds about 9,550 records with 22 input features and a binary `Target` column. The features cover physiological measurements (age, BMI, blood pressure, cholesterol, glucose level, heart rate), lifestyle factors (sleep hours, exercise hours, water intake, stress level, diet, physical activity), health background (mental health, medical history, allergies) and encoded diet type and blood group. Place the file in the project root next to the notebook.

## Approach

The data was split 80/20 with a fixed random seed (7,640 training and 1,910 test records). Features were standardised for the distance-based and linear models. The tree-based models were trained on the raw features. The models evaluated were Logistic Regression with L2 regularisation, K-Nearest Neighbours (k = 5), Random Forest (200 trees), Gradient Boosting (150 estimators, learning rate 0.1, depth 3) and a soft Voting Classifier combining Logistic Regression, KNN and Random Forest.

## Results

| Model | Accuracy | Recall |
|---|---|---|
| Random Forest | 94.4% | 94.6% |
| Gradient Boosting | 92.5% | 92.8% |
| Voting Classifier | 91.8% | 92.8% |
| K-Nearest Neighbours | 87.2% | 86.9% |
| Logistic Regression | 82.3% | 83.8% |

Random Forest gave the best result, with balanced precision and recall of about 0.94 to 0.95 on both classes. The tree-based models clearly outperformed the linear and distance-based ones, which suggests the relationship between the features and the target is non-linear.

## Installation

```
pip install -r requirements.txt
jupyter lab
```

Open `NovaGen-project.ipynb` and run the cells in order.

## Future Improvements

Hyperparameter tuning with cross-validation, feature importance analysis for the Random Forest, a data-quality check on unusual values such as very low ages, and weighting the voting ensemble towards its stronger members are the planned next steps.
