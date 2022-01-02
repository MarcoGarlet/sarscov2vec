from ProjectImportClassifierFold.ClassifierPipe import ClassifierPipe
from ProjectImportClassifierFold.CovidImputize import CovidImputize
from ProjectImportClassifierFold.CovidScaler import CovidScaler
from ProjectImportClassifierFold.CovidClassifier import CovidClassifier



class ClassifierPipeCovid(ClassifierPipe):
	def __init__(self,ds_dict,fname_ftselected='./code/model_SVM_mpro/ft_selected.txt',fname_scaler= './code/model_SVM_mpro/std_scaler.bin',fname_imputize='./code/model_SVM_mpro/imputizer_cv.bin',scaler_dim_fname='./code/model_SVM_mpro/dimension_for_scaler.txt',classifier_fname='./code/model_SVM_mpro/Classifier.sav',titles_imp_fname='./code/model_SVM_mpro/titles.txt'):
		super().__init__(ds_dict)
		self.titles_imp_fname = titles_imp_fname
		self.fname_imputize = fname_imputize
		self.fname_scaler = fname_scaler
		self.scaler_dim_fname = scaler_dim_fname
		self.classifier_fname = classifier_fname
		self.fname_ftselected = fname_ftselected

		print("In ClassifierPipeCovid OK")
		
		self.imp = CovidImputize(self.fname_imputize,self.titles_imp_fname)
		self.scal = CovidScaler(self.fname_scaler, self.scaler_dim_fname)
		self.clas = CovidClassifier(self.classifier_fname, self.fname_ftselected)