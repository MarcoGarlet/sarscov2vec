
from joblib import load


from ProjectImportClassifierFold.Classifier import Classifier





class CovidClassifier(Classifier):

	def __init__(self,classifier_fname,fname_ftselected):
		self.fname_ftselected = fname_ftselected
		self.classifier_fname = classifier_fname
		print("In CovidClassifier Constructor OK")



	def runClassifier(self,ds_dict):
		print("In CovidClassifier runScaler start OK")

		svm = load(self.classifier_fname)

		with open(self.fname_ftselected,'r') as f:
			ft_selected = f.readlines()
		ft_selected = [x.strip() for x in ft_selected]
		ds = CovidClassifier.filter_dataset(ds_dict,ft_selected)

		y_pred=svm.predict(ds)
		
		for i,k in enumerate(ds_dict.keys()):
			ds_dict[k]['is_active']=[y_pred[i]]

		print("In CovidClassifier runScaler end OK")

		return ds_dict




	@staticmethod
	def filter_dataset(mol_dict,new_ft):
		return [[mol_dict[k][k1] for k1 in mol_dict[k].keys() if k1 in new_ft] for k in mol_dict.keys()]

