from abc import ABC, abstractmethod
class Classifier(ABC):

	@abstractmethod
	def runClassifier(self):
		pass