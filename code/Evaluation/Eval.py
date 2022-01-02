from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from gensim.models import word2vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from kneed import KneeLocator


import matplotlib.pyplot as plt
import numpy as np

def get_struct(fname_lists=['CER','classified_CER','words']):
	ret = {}
	for fname in fname_lists:
		with open(fname,'r') as f:
			ret[fname] = eval(f.readline().strip())

	return ret


def show_variance_plot(pca):
	plt.rcParams["figure.figsize"] = (12,6)

	fig, ax = plt.subplots()
	xi = np.arange(1, 101, step=2)
	y = [x for i,x in enumerate(np.cumsum(pca.explained_variance_ratio_)) if i%2==1]
	print(len(xi))
	print(len(y))
	plt.ylim(0.0,1.1)
	plt.plot(xi, y, marker='o', linestyle='--', color='b')

	plt.xlabel('Number of Components')
	plt.xticks(np.arange(1, 101, step=2)) #change from 0-based array index to 1-based human-readable label
	plt.ylabel('Cumulative variance (%)')
	plt.title('The number of components needed to explain variance')

	plt.axhline(y=0.95, color='r', linestyle='-')
	plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

	ax.grid(axis='x')
	plt.savefig('./PCA_res.png')
	plt.close()


def plot_sum_square_dist(K,Sum_of_squared_distances,kn=None):
	plt.figure(1337)
	plt.plot(K, Sum_of_squared_distances, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Sum_of_squared_distances')
	if kn:
		plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	plt.title('Elbow Method For Optimal k')
	plt.savefig('./K_means.png')

def pca_plot(data_rescaled):
	pca = PCA().fit(data_rescaled)
	show_variance_plot(pca)

def plot_k_means(reduced):
	Sum_of_squared_distances = []
	for k in range(1,15):
		km = MiniBatchKMeans(n_clusters=k)
		km = km.fit(reduced)
		Sum_of_squared_distances.append(km.inertia_)
	kn = KneeLocator(range(1,15), Sum_of_squared_distances, curve='convex', direction='decreasing')

	plot_sum_square_dist(range(1,15),Sum_of_squared_distances,kn)




def eval_proj(words,classified_CER):
	#st_dict = get_struct()
	st_dict = {}
	st_dict['words'],st_dict['classified_CER'] = words, classified_CER
	model = word2vec.Word2Vec(sentences=st_dict['words'],min_count=1)
	'''
	a = [x[0] for x in model.wv.most_similar('covid-19', topn=10)] 
	print()
	for x in a:
		print(x+', ',end='')
	print()
	exit()
	'''
	X = []
	CE,active_CE,words = [],[],[]
	active_CE_dict = {}
	for k in model.wv.index_to_key:
		if k in st_dict['classified_CER'].keys():
			if k in st_dict['classified_CER'].keys() and st_dict['classified_CER'][k]['is_active'] == [1.0]:
				active_CE += [k]
				active_CE_dict[k]=st_dict['classified_CER'][k]
			else:
				CE+=[k]
		else:
			words+=[k]
		
		X+=[model.wv[k]]

	print('active CE = {}'.format(active_CE_dict))

	scaler = MinMaxScaler()
	data_rescaled = scaler.fit_transform(X)
	
	pca = PCA(n_components = 0.95)
	pca.fit(data_rescaled)
	reduced = pca.transform(data_rescaled)
	
	plot_k_means(reduced)

	n_clusters = 3
	km = MiniBatchKMeans(n_clusters=n_clusters)
	y_pred = km.fit_predict(reduced)
	
	print(y_pred)
	clust = [[] for i in range(n_clusters)]
	for x,y,k in zip(reduced,y_pred,model.wv.index_to_key):
		clust[int(y)] += [k]


	for i,l in enumerate(clust):
		CEc,aCEc,wc = 0,0,0
		for el in l:
			if el in active_CE:
				aCEc+=1
			elif el in CE:
				CEc+=1
			else:
				wc+=1
		print('Cluster {}\n\taCE = {}\n\tCE = {}\n\tw = {}'.format(i,aCEc,CEc,wc))

		
	#silhouette_avg = silhouette_score(reduced, y_pred)
	error = km.inertia_			
	print('SQE = {}'.format(error))
	return model