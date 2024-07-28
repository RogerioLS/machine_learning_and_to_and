# sources/d_feature_selection.py

from sources import StandardScaler

class FeatureSelection:
	def __init__(self, data):
		self.data = data

	def selection_feature(self):
		scaler = StandardScaler()
		scaler_data = scaler.fit_transform(self.data)

		return scaler_data
