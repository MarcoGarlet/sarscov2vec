from gensim.models import word2vec
from sklearn.decomposition import PCA
from adjustText import adjust_text

import matplotlib.pyplot as plt
import os
import numpy as np

from ProjectWE.ImporterFile import ImporterFile
from ProjectWE.WE import WE

class WEclient(WE):

	def __init__(self,method):
		self.method = method

		self.result = []
		self.model = None
		self.Importer = ImporterFile()	



	def runWE(self,CE,words):
		self.CE = CE
		self.words = words
		self.model = self.method(sentences=self.words,min_count=1)

	def runPCA(self):
		X = []
		for k in self.model.wv.index_to_key:
			X+=[self.model.wv[k]]

		pca = PCA(n_components=2)
		self.result = pca.fit_transform(X)
		assert len(X) ==  len(self.result)	

	def export_res(self):

		colors= []

		print(len( self.model.wv.index_to_key))


		print(len(self.model.wv.key_to_index.keys()))

		result_red,result_blue,result_purple = [],[],[]
		labels_res = []
		for el in zip(self.model.wv.key_to_index.keys(),self.result):

			k = el[0]
			r = el[1]
			if k in self.CE.keys():
				if self.CE[k]['is_active']==[0.0]:
					result_blue+=[r]
				
				else:
					result_red+=[r]
					labels_res+=[k]
		
			else:
				result_purple+=[r]

		
	
		

		assert len(result_red)+len(result_blue)+len(result_purple) == len(self.result)


		print(' result red = {}\n result purple = {}\n result blue = {}'.format(len(result_red),len(result_purple),len(result_blue)))

		plt.figure()


		result_purple=np.array(result_purple)
		result_blue=np.array(result_blue)
		result_red=np.array(result_red)

		for data,color,label,alpha in zip([result_purple,result_blue,result_red],['purple','blue','red'],['no CE','inactive CE','active CE'],[0.3,0.6,1]):
			if len(data)>0:
				fig = WEclient.my_scatter(data,color,label,alpha) 
	

		plt.legend()
	
		TEXTS = []
		for p,txt in zip(result_red,labels_res):
			x_i,y_i = p[0],p[1]
			TEXTS.append(plt.text(x_i, y_i, txt, color='black', fontsize=7))
	


		d_names = [int(d,16) for d in os.listdir('./dataset') if d[0]!='.']
		dir_2_download = '{:02x}'.format(max(d_names))
		sd_names = [int(d,16) for d in os.listdir('./dataset'+'/'+dir_2_download) if d[0]!='.']
		sdir_2_download = '{:02x}'.format(max(sd_names)) 

		if not os.path.exists('./PCA_OUT'):
			os.mkdir('./PCA_OUT')

		adjust_text(
		TEXTS, 
		expand_points=(2, 2),
		arrowprops=dict(
		arrowstyle="->", 
		color='black', 
		lw=2
		)#,
		#ax=fig.axes[0]
		)
		plt.savefig('./PCA_OUT/result_{}_{}.png'.format(int(dir_2_download,16),int(sdir_2_download,16)))



	@staticmethod
	def my_scatter(data,color,label,alpha):
		return plt.scatter(
			data[:,0],
			data[:,1],
			color='tab:'+color,
			label=label,
			alpha=alpha, edgecolors='none'
		)