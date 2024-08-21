# sources/f_generate_evaluation_metrics.py

from sources import mean_absolute_error, r2_score
from sources import os
from sources import pickle

class ModelEvaluator:
	def __init__(self, model, X_val, y_val, output_dir = '../model_trained'):
		self.model = model
		self.X_val = X_val
		self.y_val = y_val
		self.output_dir = output_dir

		os.makedirs(self.output_dir, exist_ok = True)

	def evaluate_model(self):
		y_pred = self.model.predict(self.X_val)
		mae = mean_absolute_error(self.y_val, y_pred)
		r2 = r2_score(self.y_val, y_pred)
		return mae, r2

	def save_model(self, model_name='model_regression_immobile.pkl'):
		filepath = os.path.join(self.output_dir, model_name)
		with open(filepath, 'wb') as file:
			pickle.dump(self.model, file)
		print(f'Modelo salvo em: {filepath}')
