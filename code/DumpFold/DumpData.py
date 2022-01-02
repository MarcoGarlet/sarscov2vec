from abc import ABC, abstractmethod
class DumpData(ABC):
	@abstractmethod
	def runDump(self):
		pass
	@abstractmethod
	def runJoin(self):
		pass
