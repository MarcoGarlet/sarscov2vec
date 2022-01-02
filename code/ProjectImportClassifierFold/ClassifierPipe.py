
class ClassifierPipe():

	def __init__(self, ds_dict):
		self.ds_dict = ds_dict

	def runClassifierClient(self):
		return self.clas.runClassifier(self.ds_dict)
		
	def runImputizerClient(self):
		self.ds_dict = self.imp.runImputizer(self.ds_dict)

	def runScalerClient(self):
		self.ds_dict = self.scal.runScaler(self.ds_dict)
		

	def setRunClassifierClient(self,newClassAlg):
		self.clas = newClassAlg
		
	def setRunImputizerClient(self,newImpAlg):
		self.imp = newImpAlg

	def setRunScalerClient(self,newScalAlg):
		self.scal = newScalAlg

		