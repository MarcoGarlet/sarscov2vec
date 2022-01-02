
from ProjectWE.Importer import Importer

class ImporterFile(Importer):
    def __init__(self,CEname='classifiedCER', wordsname='words',docsdict='docsdict'):
        self.CE_fname = CEname
        self.words_fname = wordsname
        self.docsdict_fname = docsdict
    def runImport(self):
        print('Before import file')
        self.CE = ImporterFile.import_file(self.CE_fname)
        print('CE_classified len ={}'.format(len(self.CE)))
        self.words = ImporterFile.import_file(self.words_fname)
        print('words len ={}'.format(len(self.words)))  
        self.docsdict = ImporterFile.import_file(self.docsdict_fname)
        return self.CE,self.words
    
    def query_docsdict(self,w_list):
        return [self.docsdict[el] for el in w_list]
    @staticmethod
    def import_file(fname,directory='./results'):
        with open(directory+'/'+fname,'r') as f:
            line = f.readline().strip()
        return eval(line)
