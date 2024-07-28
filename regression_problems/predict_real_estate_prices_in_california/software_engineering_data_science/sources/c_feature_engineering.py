# sources/c_feature_engineer.py

class FeatureEngineer:
	def __init__(self, data):
		self.data = data

	def create_features(self):
		if self.data is None:
			raise ValueError("Data is not provided for feature engineering.")
		self.data['RoomDensity'] = self.data['AveRooms'] / self.data['AveOccup']

		return self.data
