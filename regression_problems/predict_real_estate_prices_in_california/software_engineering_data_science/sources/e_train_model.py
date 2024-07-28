# sources/e_trai_model.py

from sources import RandomForestRegressor, train_test_split

class ModelTrainer:
	def __init__(self, data, target):
		self.data = data
		self.target = target
		self.model = RandomForestRegressor()

	def train_model(self):
		X_train, X_val, y_train, y_val = train_test_split(self.data, self.target, test_size=0.4, random_state=42)
		self.model.fit(X_train, y_train)
		return self.model, X_val, y_val
