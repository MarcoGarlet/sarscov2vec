from abc import ABC, abstractmethod
class ImportData(ABC):
	@abstractmethod
	def runImport(self):
		pass