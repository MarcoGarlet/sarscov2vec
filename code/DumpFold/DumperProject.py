from pathlib import Path

from DumpFold.Dumper import Dumper
from DumpFold.ProjectDump import ProjectDump


class DumperProject(Dumper):
	def __init__(self,labels,struct,directory='./results'):
		self.labels = labels
		self.struct = struct 
		self.directory=directory
		Path(self.directory).mkdir(parents=True, exist_ok=True)
		self.dumObj = ProjectDump(labels,struct,directory)
