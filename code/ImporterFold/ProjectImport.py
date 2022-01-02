from ImporterFold.ImportData import ImportData
import os
from Utils.thread_manager import *
from bs4 import BeautifulSoup
import requests


class ProjectImport(ImportData):
	def __init__(self,folder_location, number_of_dirs, max_nthreads_download, link):
		self.folder_location = folder_location
		self.number_of_dirs = number_of_dirs
		self.max_nthreads_download = max_nthreads_download
		self.link = link

	def runImport(self):
		fnames = self.getFnameList()
		return fnames



	def getFnameList(self):

		if not os.path.exists(self.folder_location):
			os.mkdir(self.folder_location)

		d_names = [int(d,16) for d in os.listdir(self.folder_location) if d[0]!='.']
		print(d_names)
		dir_n = max(d_names) if len(d_names)>0 else 0
		dir_2_download = '{:02x}'.format(max(d_names)) if len(d_names)>0 else '00'

		if not os.path.exists(self.folder_location+'/'+dir_2_download):
			os.mkdir(self.folder_location+'/'+dir_2_download)

		sd_names = [int(d,16) for d in os.listdir(self.folder_location+'/'+dir_2_download) if d[0]!='.']
		subdir_n = max(sd_names) if len(sd_names)>0 else 0
		if len(sd_names)!=0: subdir_n+=1
		
		if subdir_n==0x100:
			subdir_n=0
			dir_n+=1
		if dir_n==0x100:
			print('Dataset Processed')
			exit()


		start = ProjectImport.dir2int(dir_n,subdir_n)
		
		d2_downloads = [ProjectImport.int2dir(i) for i in range(start, start+self.number_of_dirs)]
		set_2_down = [(ProjectImport.format_2bytes(el[0]),ProjectImport.format_2bytes(el[1])) for el in d2_downloads]
		dir_2_download = set([el[0] for el in set_2_down])

		for d in dir_2_download:
			if not os.path.exists(self.folder_location+'/'+d):
				os.makedirs(self.folder_location+'/'+d)

		set_2_down_d = [set_2_down[i*self.max_nthreads_download:i*self.max_nthreads_download+self.max_nthreads_download] for i in range(len(set_2_down)//self.max_nthreads_download)]


		if len(set_2_down)%self.max_nthreads_download != 0:
			set_2_down_d+=[set_2_down[(len(set_2_down)//self.max_nthreads_download)*self.max_nthreads_download:]]

		for sub_2_down in set_2_down_d:
			pool = ThreadPool(len(sub_2_down))
			pool.map(self.runner_get_pdf, sub_2_down)
			pool.wait_completion()
				
		fnames=self.gen_fnames(set_2_down)

		return fnames

	def gen_fnames(self,set_2_down):
		fnames=[]
		for d,sd in set_2_down:
			for f in os.listdir(self.folder_location+'/'+d+'/'+sd):
				if d=='.DS_Store': continue
				fnames+=[self.folder_location+'/'+d+'/'+sd+'/'+f]	
		return fnames

	def runner_get_pdf(self,path_2_download):
		dir_2_download=path_2_download[0]
		subdir_2_download=path_2_download[1]
		print('\t [START]\tDownloading => {}'.format(path_2_download))
		
		url=self.link+'/'+dir_2_download+'/'+subdir_2_download
		folder_location_local=self.folder_location+r'/'+dir_2_download+r'/'+subdir_2_download    
		print(folder_location_local)
		response = requests.get(url)
		soup= BeautifulSoup(response.text, "html.parser") 
		os.mkdir(folder_location_local)

		for l in soup.select("a[href$='.pdf']"):
			#Name the pdf files using the last portion of each link which are unique in this case
			filename = os.path.join(folder_location_local,l['href'].split('/')[-1])
			with open(filename, 'wb') as f:
				f.write(requests.get(url+'/'+l['href']).content)
		
		print('\t [END]\tDownloading => {}'.format(path_2_download))

	@staticmethod
	def format_2bytes(n):
		return '{:02x}'.format(n)
	@staticmethod
	def dir2int(int_dir,int_subdir):
		return (int_dir*0x100)+int_subdir
	@staticmethod
	def int2dir(int_dir_sub):
		return ((int_dir_sub&0xff00)>>2**3,int_dir_sub&0xff)



