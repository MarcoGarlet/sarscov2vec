from pymongo import MongoClient
from ProjectWE.Importer import Importer

class ImporterMongo(Importer):
    def __init__(self,CEname='classifiedCER', wordsname='words',docsdict='docsdict'):
        try:
            self.client = MongoClient('mongo-server:27017')
        except:
            print("Error connection mongo in ImporterFiles")
            exit()

        self.db = self.client["IRProject"]
        self.curCE = self.db[CEname]
        self.curwords = self.db[wordsname]
        self.curdocsdict = self.db[docsdict]


    def runImport(self):

        self.CE = list(self.curCE.find({}))
        self.CE = {el['_id']:el['struct'] for el in self.CE}
        print('CE_classified len ={}'.format(len(self.CE)))
        self.words = list(self.curwords.find({},{'sent':1,'_id':0}))
        self.words = [el['sent'].split() for el in self.words]
        #print(self.words)
        print('words len ={}'.format(len(self.words)))
        return self.CE,self.words

    def query_docsdict(self,w_list):
        res = []
        for q in w_list:
            res += [self.curdocsdict.find_one({'_id':q},{'_id':0,'struct':1})['struct']]
        #print('res = {}, type = {}'.format(res,type(res[0])))

        return res


        # query original words from stemmed w_list
        # the mongo server has k:(k,val) as value

    
