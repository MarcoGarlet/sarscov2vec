
class Dumper():
	def runDump(self):
		return self.dumObj.runDump()

	def runJoinRes(self,labels):
		return self.dumObj.runJoin(labels)

	def setImportData(self,newDumpObj):
		self.dumObj = newDumpObj


