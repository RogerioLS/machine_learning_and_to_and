# sources/a_fetch_dataset.py

from sources import pd, fetch_california_housing

class DataFetcher:
	def __init__(self):
		self.data = None
		self.target = None
		self.feature_names = None

	def fetch_data(self):
		cali_data = fetch_california_housing()
		self.data = pd.DataFrame(cali_data.data, columns=cali_data.feature_names)
		self.target = cali_data.target
		self.feature_names = cali_data.feature_names
		
		return self.data, self.target
