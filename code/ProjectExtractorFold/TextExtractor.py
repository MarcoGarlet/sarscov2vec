
class TextExtractor():

	def runExtractor(self):
		return self.myCEWordExtractionAlg.runCEWordExtraction()

	def setImportData(self,newCEWordExtractionAlg):
		self.myCEWordExtractionAlg = newCEWordExtractionAlg

