#from ImporterFold.ImportData import ImportData
#from CEWordExtraction import CEWordExtraction
#from DumpResult import DumpResult


class Importer():
	def runImport(self):
		return self.myImportDataAlg.runImport()
		
	def setImportData(self,newImportDataAlg):
		self.myImportDataAlg = newImportDataAlg

