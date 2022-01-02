from abc import ABC, abstractmethod
class Scaler():
	@abstractmethod
	def runScaler(self,ds_dict):
		pass