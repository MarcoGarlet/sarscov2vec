from joblib import load
from ProjectImportClassifierFold.Scaler import Scaler
import math
import numpy as np

class CovidScaler(Scaler):

	def __init__(self,fname_scaler,scaler_dim_fname):
		self.fname_scaler = fname_scaler
		self.scaler_dim_fname = scaler_dim_fname
		print("In CovidScaler Constructor OK")


	def runScaler(self,ds_dict):
		print("In CovidScaler runScaler start OK")

		for el in ds_dict.keys():
			for c in ds_dict[el].keys():
				if math.isnan(ds_dict[el][c]) or math.isinf(ds_dict[el][c]):
					print("NAN/INF values after scaler {}".format(ds_dict[el][c]))
					
		print("In scaler after check NAN values")


		with open(self.scaler_dim_fname,'r') as f:
			ft_sc= eval(f.readline().strip())
		sc=load(self.fname_scaler)
		ds = CovidScaler.filter_dataset(ds_dict,ft_sc)
		

		print("In CovidScaler before transform OK")
		ds = [np.array(x,dtype=np.float64) for x in ds]
		ds = np.array(ds,dtype=np.float64)
		ds = sc.transform(ds)
		print("In CovidScaler after transform OK")


		for el,v_norm in zip(ds_dict.keys(),ds):
			for c,v_el in zip(ds_dict[el].keys(),v_norm):
				ds_dict[el][c] = v_el
		print("In CovidScaler after ds_dict update OK")



		del_v=[]

		for el in ds_dict.keys():
			null_p = False
			for c in ds_dict[el].keys():
				if math.isnan(ds_dict[el][c]) or math.isinf(ds_dict[el][c]):
					print("NAN/INF values after scaler {}".format(ds_dict[el][c]))
					null_p = True 
					break
			if null_p:
				del_v+=[el]

		for k in del_v:
			del ds_dict[k]

				


		print("In CovidScaler runScaler end OK")

		return ds_dict

	@staticmethod
	def filter_dataset(mol_dict,new_ft):
		return [[mol_dict[k][k1] for k1 in mol_dict[k].keys() if k1 in new_ft] for k in mol_dict.keys()]

