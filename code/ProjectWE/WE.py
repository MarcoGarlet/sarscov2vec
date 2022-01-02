class WE():
    def runImporter(self):
        return self.Importer.runImport()
    
    def query_docsdict(self,w_list):
        return self.Importer.query_docsdict(w_list)

    def setImporter(self,newImpObj):
        self.Importer = newImpObj
