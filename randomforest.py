import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class StressDetectionModel:
    def __init__(self):
        # Load the dataset
        self.df = pd.read_csv("stress.csv")

        # Define the columns you want to standardize and treat outliers
        self.numerical_columns = ['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sr.1', 'hr']

        # Calculate Z-scores for outlier detection and treatment
        self.z_scores = np.abs(stats.zscore(self.df[self.numerical_columns]))

        # Define a threshold for outlier detection (e.g., 3 standard deviations)
        self.threshold = 3

        # Identify and remove outliers
        self.df_no_outliers = self.df[(self.z_scores < self.threshold).all(axis=1)]

    def visualize_correlation_matrix(self):
        # Calculate and visualize the correlation matrix
        correlation_matrix = self.df_no_outliers[self.numerical_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Matrix')
        plt.show()

class StressDetectionRandomForest:
    def __init__(self):
        self.model = None

    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        numerical_columns = ['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sr.1', 'hr']
        z_scores = np.abs(stats.zscore(df[numerical_columns]))
        threshold = 3
        df_no_outliers = df[(z_scores < threshold).all(axis=1)]
        X = df_no_outliers.drop(columns=['sl'])
        y = df_no_outliers['sl']

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(x_train, y_train)
        y_pred = rf_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Random Forest Accuracy:", accuracy)
        print('\nClassification Report:\n', classification_report(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred)
        print('\nConfusion Matrix:\n', conf_matrix)
        rf_filename = 'random_forest_model.sav'
        with open(rf_filename, 'wb') as rf_model_file:
            pickle.dump(rf_classifier, rf_model_file)

if __name__ == "__main__":
    model = StressDetectionModel()
    model.visualize_correlation_matrix()
    random_forest = StressDetectionRandomForest()
    random_forest.load_data("stress.csv")
