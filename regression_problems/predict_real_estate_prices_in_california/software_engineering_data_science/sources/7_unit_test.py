import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('./data/iris.csv')
        self.X = self.data.drop('target', axis=1)
        self.y = self.data['target']

    def test_scaler(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        self.assertEqual(X_scaled.shape, self.X.shape)

class TestModel(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('./data/iris_feature_engineered.csv')
        self.X = self.data.drop('target', axis=1)
        self.y = self.data['target']

    def test_train_and_predict(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.assertEqual(len(y_pred), len(y_test))

if __name__ == "__main__":
    unittest.main()
