# main/main.py

from sources.a_fetch_dataset import DataFetcher
from sources.b_clean_dataset import DataCleaner
from sources.c_feature_engineering import FeatureEngineer
from sources.d_feature_selection import FeatureSelection
from sources.e_train_model import ModelTrainer
from sources.f_generate_evaluation_metrics import ModelEvaluator
import unittest

def main():
	fetcher = DataFetcher()
	data, target = fetcher.fetch_data()

	cleaner = DataCleaner(data)
	clean_data = cleaner.clean_data()

	engineer = FeatureEngineer(clean_data)
	engineered_data = engineer.create_features()

	selector = FeatureSelection(engineered_data)
	selector_features = selector.selection_feature()

	trainer = ModelTrainer(selector_features, target)
	model, X_val, y_val = trainer.train_model()

	evaluator = ModelEvaluator(model, X_val, y_val)
	mae, r2 = evaluator.evaluate_model()

	print(f'MAE: {mae}, R²: {r2}')

def run_tests():
	unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromModule(__import__('sources.g_unit_test')))

if __name__ == "__main__":
	run_tests()
	print("Model:")
	main()