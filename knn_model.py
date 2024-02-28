import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
df = pd. read_csv('stress.csv')

class StressDetectionModel:
    def __init__(self):
        self.df = pd.read_csv("stress.csv")
        self.numerical_columns = ['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sr.1', 'hr']
        self.z_scores = np.abs(stats.zscore(self.df[self.numerical_columns]))
        self.threshold = 3
        self.df_no_outliers = self.df[(self.z_scores < self.threshold).all(axis=1)]

    def visualize_correlation_matrix(self):
        correlation_matrix = self.df_no_outliers[self.numerical_columns].corr()  # Use the dataset without outliers
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Matrix')
        plt.show()

class StressDetectionKNN:
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
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(x_train, y_train)
        y_pred = knn_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("KNN Accuracy:", accuracy)
        print('\nClassification Report:\n', classification_report(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred)
        print('\nConfusion Matrix:\n', conf_matrix)

        knn_filename = 'knn_model.sav'
        with open(knn_filename, 'wb') as knn_model_file:
            pickle.dump(knn_classifier, knn_model_file)

if __name__ == "__main__":
    model = StressDetectionModel()
    model.visualize_correlation_matrix()

    knn = StressDetectionKNN()
    knn.load_data("stress.csv")
