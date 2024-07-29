# sources/_unit_test.py

import unittest

from sources.a_fetch_dataset import DataFetcher
from sources.b_clean_dataset import DataCleaner
from sources.c_feature_engineering import FeatureEngineer
from sources.d_feature_selection import FeatureSelection
from sources.e_train_model import ModelTrainer
from sources.f_generate_evaluation_metrics import ModelEvaluator

class TestDataFetcher(unittest.TestCase):
    def test_fetch_data(self):
        fetcher = DataFetcher()
        data, target = fetcher.fetch_data()
        self.assertIsNotNone(data, "Data should not be None")
        self.assertIsNotNone(target, "Target should not be None")
        self.assertEqual(len(data), len(target), "Data and target lengths should be equal")

class TestDataCleaner(unittest.TestCase):
    def test_clean_data(self):
        fetcher = DataFetcher()
        data, target = fetcher.fetch_data()
        cleaner = DataCleaner(data)
        clean_data = cleaner.clean_data()
        self.assertIsNotNone(clean_data, "Clean data should not be None")
        self.assertGreater(len(clean_data), 0, "Clean data should not be empty")

class TestFeatureEngineer(unittest.TestCase):
    def test_create_features(self):
        fetcher = DataFetcher()
        data, target = fetcher.fetch_data()
        cleaner = DataCleaner(data)
        clean_data = cleaner.clean_data()
        engineer = FeatureEngineer(clean_data)
        engineered_data = engineer.create_features()
        self.assertIn('RoomDensity', engineered_data.columns, "Feature 'RoomDensity' should be in the engineered data")

class TestFeatureSelector(unittest.TestCase):
    def test_select_features(self):
        fetcher = DataFetcher()
        data, target = fetcher.fetch_data()
        cleaner = DataCleaner(data)
        clean_data = cleaner.clean_data()
        engineer = FeatureEngineer(clean_data)
        engineered_data = engineer.create_features()
        selector = FeatureSelection(engineered_data)
        selected_features = selector.selection_feature()
        self.assertIsNotNone(selected_features, "Selected features should not be None")
        self.assertGreater(len(selected_features), 0, "Selected features should not be empty")

class TestModelTrainer(unittest.TestCase):
    def test_train_model(self):
        fetcher = DataFetcher()
        data, target = fetcher.fetch_data()
        cleaner = DataCleaner(data)
        clean_data = cleaner.clean_data()
        engineer = FeatureEngineer(clean_data)
        engineered_data = engineer.create_features()
        selector = FeatureSelection(engineered_data)
        selected_features = selector.selection_feature()
        trainer = ModelTrainer(selected_features, target)
        model, X_val, y_val = trainer.train_model()
        self.assertIsNotNone(model, "Model should not be None")
        self.assertIsNotNone(X_val, "Validation data should not be None")
        self.assertIsNotNone(y_val, "Validation target should not be None")

class TestModelEvaluator(unittest.TestCase):
    def test_evaluate_model(self):
        fetcher = DataFetcher()
        data, target = fetcher.fetch_data()
        cleaner = DataCleaner(data)
        clean_data = cleaner.clean_data()
        engineer = FeatureEngineer(clean_data)
        engineered_data = engineer.create_features()
        selector = FeatureSelection(engineered_data)
        selected_features = selector.selection_feature()
        trainer = ModelTrainer(selected_features, target)
        model, X_val, y_val = trainer.train_model()
        evaluator = ModelEvaluator(model, X_val, y_val)
        mae, r2 = evaluator.evaluate_model()
        self.assertGreaterEqual(r2, 0, "R² should be greater than or equal to 0")
        self.assertLessEqual(mae, 100000, "MAE should be a reasonable value")

class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        fetcher = DataFetcher()
        data, target = fetcher.fetch_data()

        cleaner = DataCleaner(data)
        clean_data = cleaner.clean_data()

        engineer = FeatureEngineer(clean_data)
        engineered_data = engineer.create_features()

        selector = FeatureSelection(engineered_data)
        selected_features = selector.selection_feature()

        trainer = ModelTrainer(selected_features, target)
        model, X_val, y_val = trainer.train_model()

        evaluator = ModelEvaluator(model, X_val, y_val)
        mae, r2 = evaluator.evaluate_model()

        self.assertIsNotNone(data, "Data should not be None")
        self.assertIsNotNone(target, "Target should not be None")
        self.assertIsNotNone(clean_data, "Clean data should not be None")
        self.assertIn('RoomDensity', engineered_data.columns, "Feature 'RoomDensity' should be in the engineered data")
        self.assertIsNotNone(selected_features, "Selected features should not be None")
        self.assertIsNotNone(model, "Model should not be None")
        self.assertIsNotNone(X_val, "Validation data should not be None")
        self.assertIsNotNone(y_val, "Validation target should not be None")
        self.assertGreaterEqual(r2, 0, "R² should be greater than or equal to 0")
        self.assertLessEqual(mae, 100000, "MAE should be a reasonable value")

if __name__ == "__main__":
    unittest.main()