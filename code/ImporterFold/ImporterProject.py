from ImporterFold.Importer import Importer
from ImporterFold.ProjectImport import ProjectImport
class ImporterProject(Importer):
	def __init__(self,folder_location=r'./dataset',number_of_dirs= 2,max_nthreads_download= 8,link='https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf'):
		self.folder_location = folder_location
		self.number_of_dirs = number_of_dirs
		self.max_nthreads_download = max_nthreads_download
		self.link = link
		self.myImportDataAlg = ProjectImport(self.folder_location,self.number_of_dirs,self.max_nthreads_download,self.link)
