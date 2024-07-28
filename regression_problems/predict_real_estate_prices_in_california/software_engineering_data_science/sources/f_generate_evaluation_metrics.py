# sources/f_generate_evaluation_metrics.py

from sources import mean_absolute_error, r2_score

class ModelEvaluator:
	def __init__(self, model, X_val, y_val):
		self.model = model
		self.X_val = X_val
		self.y_val = y_val

	def evaluate_model(self):
		y_pred = self.model.predict(self.X_val)
		mae = mean_absolute_error(self.y_val, y_pred)
		r2 = r2_score(self.y_val, y_pred)
		return mae, r2
