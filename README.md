
# Rainfall Prediction Using Decision Trees and Random Forest

## Project Overview
This project focuses on predicting whether it will rain tomorrow using weather data from various locations. The models used include **Decision Trees** and **Random Forest** classifiers. The dataset includes features such as temperature, humidity, wind, and more, which are used to predict rainfall.

The objective is to train and evaluate machine learning models that can effectively predict the likelihood of rain the next day.

## Dataset
The dataset used in this project is sourced from the [Weather Dataset (Australia)](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package). It includes the following features:
- Date
- Location
- Min/Max temperature
- Rainfall
- Wind Speed and Direction
- Humidity
- Cloud cover
- Rain today
- Rain tomorrow (target variable)

There are missing values in the dataset, particularly in the `Sunshine`, `Evaporation`, and `Cloud` columns, which are handled appropriately during the preprocessing stage.

## Project Structure
- **Data Preprocessing**: Includes handling missing values, feature engineering, and splitting the data into training, validation, and test sets.
- **Exploratory Data Analysis (EDA)**: Visualizations using `matplotlib` and `seaborn` to understand feature distributions and relationships.
- **Model Training**: Decision Trees and Random Forest models are trained and evaluated for performance using metrics like accuracy and confusion matrix.
- **Model Evaluation**: Comparison of model performance on validation and test data.

## Key Steps
1. **Import Libraries**: Includes Pandas, Numpy, Matplotlib, Seaborn, and Scikit-Learn.
2. **Data Loading**: The dataset is loaded and cleaned to remove unnecessary null values.
3. **Exploratory Data Analysis (EDA)**: Understand data distribution and relationships.
4. **Feature Engineering**: Creating new features and handling categorical variables.
5. **Model Training**:
    - Decision Trees
    - Random Forest
6. **Evaluation**: The models are evaluated using various metrics like accuracy, precision, recall, and F1-score.

## How to Run the Project

### Prerequisites
To run this project locally, you need the following packages installed:

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Code
1. Clone the repository or download the notebook.
2. Ensure the dataset is available in the appropriate directory or modify the path in the code.
3. Run the notebook using Jupyter or any Python IDE to see the results.

## Results
The Random Forest model showed better performance compared to the Decision Tree classifier, achieving higher accuracy and better generalization on unseen test data. The project successfully predicts rainfall for the next day with an acceptable level of accuracy.

## Future Work
- Improve model accuracy by tuning hyperparameters and experimenting with additional models.
- Implement other machine learning models such as Gradient Boosting or XGBoost.
- Integrate real-time weather data to make predictions on new, incoming data.

## License
This project is licensed under the MIT License.

