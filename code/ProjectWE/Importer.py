from abc import ABC, abstractmethod
class Importer(ABC):
    @abstractmethod 
    def runImport(self):
        pass
    @abstractmethod
    def query_docsdict(self,w_list):
        pass


