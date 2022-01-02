from abc import ABC, abstractmethod
class Imputize(ABC):
	@abstractmethod
	def runImputizer(self,ds_dict):
		pass