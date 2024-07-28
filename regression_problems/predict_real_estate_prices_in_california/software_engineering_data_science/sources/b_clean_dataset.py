# sources/b_clean_dataset.pt

from sources import pd

class DataCleaner:
	def __init__(self, data):
		self.data = data

	def has_missing_values(self):

		return self.data.any().any()

	def clean_data(self):
		if self.has_missing_values():
			self.data = self.data.dropna()
		else:
			self.data

		return self.data
