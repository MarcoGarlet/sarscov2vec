
import pymongo
from pymongo import MongoClient
from DumpFold.DumpData import DumpData

class ProjectMongoDump(DumpData):
    def __init__(self,labels,struct,directory=None):
        assert len(struct) == len(labels)
        try:
            self.client = MongoClient('mongo-server:27017')
        except:
            print("Error connection mongo in ProjectMongoDump")
            exit()

        self.db = self.client["IRProject"]
        self.labels = [self.db[l.split('_')[0]] for l in labels]
        self.struct = struct

    def runDump(self):
        for i in range(len(self.struct)):
            if type(self.struct[i])==type({}):
                try:
                    print('###DEBUG rundump, in {} self.struct[i].items = {} '.format(self.labels[i],self.struct[i].items()))
                    self.labels[i].insert_many([{'_id':k,'struct':el} for k,el in self.struct[i].items()], ordered=False)
                except pymongo.errors.BulkWriteError:
                    print('Duplicate warning => [ignore]')
            else:
                print([{'sent':' '.join(sent)} for sent in self.struct[i]][:2])
                self.labels[i].insert_many([{'sent':' '.join(sent)} for sent in self.struct[i]])

    def runJoin(self,labels):
        pass

