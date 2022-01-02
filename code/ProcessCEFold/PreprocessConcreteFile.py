import os
import shutil
import math

from functools import reduce
from ProcessCEFold.PreprocessCE import PreprocessCE

class PreprocessConcreteFile(PreprocessCE):
	def __init__(self,titles_fname,fname,width_slice):
		self.fname = fname
		self.titles_fname = titles_fname
		self.width_slice = width_slice
		


	def runPreprocess(self):
		with open(self.fname,'r') as f:
			self.CER = eval(f.readline().strip())	
		self.width_slice = self.width_slice

		with open(self.titles_fname,'r') as f:
			covid_titles = f.readlines()

		self.covid_titles = {c:t.strip() for c,t in enumerate(covid_titles)}

		self.prepare_smiles()
		return PreprocessConcreteFile.build_sarscov_dataset(self.covid_titles)


	def prepare_smiles(self):

		if os.path.exists("./padel-service/struct"):
			shutil.rmtree("./padel-service/struct")
		if os.path.exists("./padel-service/res"):
			shutil.rmtree("./padel-service/res")


		os.makedirs('./padel-service/struct')
		os.makedirs('./padel-service/res')

		CER_nslices = len(self.CER.keys())//self.width_slice

		print(len(self.CER.keys()))
		CER_list = [(k,self.CER[k][0]) for k in self.CER.keys()]

		CER_lists_sliced = [CER_list[self.width_slice*i:self.width_slice*i+self.width_slice] for i in range(CER_nslices)]

		if len(self.CER.keys())%self.width_slice!=0:
			CER_lists_sliced+=[CER_list[(CER_nslices)*self.width_slice:]]
		
		
		assert CER_list == reduce(lambda x,y:x+y,CER_lists_sliced) 
		assert len(self.CER.keys()) == len(CER_list)

		print(len(CER_lists_sliced))
		
		for n_CER,CER_l in enumerate(CER_lists_sliced):
			if not os.path.exists('./padel-service/struct/struct_'+str(n_CER)): os.makedirs('./padel-service/struct/struct_'+str(n_CER))
			for i,el in enumerate(CER_l):
				with open('./padel-service/struct/struct_'+str(n_CER)+'/'+str(i)+'.smi','w') as f:
					f.write(el[1]+'\t'+el[0]+'\n')



	@staticmethod
	def build_sarscov_dataset(covid_titles):
		
		smi_dirs=[subdir for subdir,_,_ in os.walk('./padel-service/struct/')][1:]
		smi_dirs=[(i,el) for i,el in enumerate(smi_dirs)]
		print(smi_dirs)
		
		items = []
		for smi_dir in smi_dirs:
			items+=[PreprocessConcreteFile.smi(smi_dir)]
		desc_mol = {}
		for ft,mol_desc in items:
			desc_mol.update({el[0]:{ft[i]:el[1:][i] for i in range(len(ft)) if ft[i] in covid_titles.values()} for el in mol_desc})
		print(len(desc_mol.keys()))
		
		return desc_mol

	@staticmethod
	def smi(smi_dir_fname):
		smi_dir = './'+'/'.join(smi_dir_fname[1].split('/')[2:])
		fname = str(smi_dir_fname[0])
		print('[START]\tFile {}'.format(smi_dir))

		command = 'echo "-maxruntime 8000 -threads -1 -2d -3d -file ./res/'+fname+' -dir '+smi_dir+'" | nc padel-server 2323'
		os.system(command)
		os.system('chmod 777 ./padel-service/res/'+fname)
		print('[EXECUTED PADEL]\tfile {}'.format(smi_dir))

		ft=[]
		mol_desc=[]

		with open('./padel-service/res/'+fname,'r') as f:
			for i,l in enumerate(f.readlines()):
				l=l.strip()
				if i==0:
					ft = eval('['+','.join(["'"+el+"'" for el in l.split(',')])+']')					
				else:
					mol_desc+=[eval('['+','.join(['math.nan' if 'Infinity' in el else el if len(el)>0 else 'math.nan' for el in l.split(',')])+']')]
		
		ft = ft[1:] # not consider name
		
		print('[END]\tThread {} done'.format(smi_dir))
		return ((ft,mol_desc))	

