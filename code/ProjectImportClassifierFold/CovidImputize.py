import math
from joblib import load
from ProjectImportClassifierFold.Imputize import Imputize

class CovidImputize(Imputize):
	
	def __init__(self,fname_imputize,titles_imp_fname):
		self.fname_imputize = fname_imputize
		self.titles_imp_fname = titles_imp_fname
		print("In CovidImputize Constructor OK")


	def runImputizer(self,ds_dict):
		print("In CovidImputize runImputizer start OK")

		imp = load(self.fname_imputize) # this should be exported using sklearn 0.23.2
		with open(self.titles_imp_fname,'r') as f:
			titles_imp = f.readlines()
		assert len(titles_imp) == len(imp.statistics_)

		imp_stat = {titles_imp[i].strip():imp.statistics_[i] for i in range(len(titles_imp))}
		
		for el in ds_dict.keys():
			for c in ds_dict[el].keys():
				if math.isnan(ds_dict[el][c]) or math.isinf(ds_dict[el][c]):
					ds_dict[el][c]=imp_stat[c]

		for el in ds_dict.keys():
			for c in ds_dict[el].keys():
				if math.isnan(ds_dict[el][c]) or math.isinf(ds_dict[el][c]):
					print("NAN/INF values after imputizer {}".format(ds_dict[el][c]))
					exit()

		print("In CovidImputize runImputizer end OK")


		return ds_dict