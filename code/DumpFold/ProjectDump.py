import os

from DumpFold.DumpData import DumpData
from functools import reduce
from pathlib import Path


class ProjectDump(DumpData):
	def __init__(self,labels,struct,directory):
		assert len(struct) == len(labels)
		self.labels = labels
		self.struct = struct
		self.directory = directory

	def runDump(self):
		for i in range(len(self.struct)):
			print('dump {}'.format(self.labels[i]))
			with open(self.directory+'/'+self.labels[i],'w') as f:
				f.write(str(self.struct[i]))

	def runJoin(self,labels):
		old = {}
		struct = {}

		for l in labels:
			fhandle = Path(self.directory+'/'+l)
			fhandle.touch(exist_ok=True)

		

		for i in range(len(labels)):
			print('updating {}'.format(labels[i]))
			
			with open(self.directory+'/'+labels[i]+'_last_comp','r') as f:
				l = f.readline().strip()
				s = [] if labels[i]=='words' else {}
				struct[i] = eval(l) if len(l)>0 else s
			

			with open(self.directory+'/'+labels[i],'r') as f:
				l = f.readline().strip()
				s = [] if labels[i]=='words' else {}
				old[i] = eval(l) if len(l)>0 else s

			print('len(old[i])={}'.format(len(old[i])))
			print('len(struct[i])={}'.format(len(struct[i])))

			with open(self.directory+'/'+labels[i],'w') as f:
				if labels[i]=='words':			
					f.write(str(old[i]+struct[i]))
				
				else:
					old[i].update(struct[i])
					f.write(str(old[i]))	

			with open(self.directory+'/'+labels[i],'r') as f:
				if labels[i]=='words':
					l=eval(f.readline().strip())
				else:
					l=eval(f.readline().strip())
					l=l.keys()
			
					
				set_o = set() if len(old[i])==0 else set(reduce(lambda x,y: x+y,old[i])) if labels[i]=='words' else set(old[i].keys())
				set_k = set() if len(struct[i])==0 else set(reduce(lambda x,y: x+y,struct[i])) if labels[i]=='words' else set(struct[i].keys()) 
				
				l = set() if len(l)==0 else set(reduce(lambda x,y: x+y,l)) if labels[i]=='words' else set(l)

				print("set_o = {}\nset_k = {}\nl = {}".format(set_o,set_k,l))
				assert set_o.issubset(l) and set_k.issubset(l)

