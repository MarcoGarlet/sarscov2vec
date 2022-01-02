from ProcessCEFold.PreprocesserCER import PreprocesserCER
from ProcessCEFold.PreprocessConcreteFile import PreprocessConcreteFile


class PreprocessCEProject(PreprocesserCER):
	def __init__(self, titles_fname='./code/model_SVM_mpro/titles.txt', fname='./results/CER_last_comp',width_slice=5):
		self.titles_fname = titles_fname
		self.fname = fname 
		self.width_slice = width_slice 
		self.preProcObj = PreprocessConcreteFile(titles_fname,fname,width_slice)