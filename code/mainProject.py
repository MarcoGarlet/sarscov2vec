from gensim.models import word2vec

from ImporterFold.ImporterProject import ImporterProject
from ProjectExtractorFold.TextExtractorProject import TextExtractorProject
from DumpFold.DumperProject import DumperProject
from DumpFold.ProjectMongoDump import ProjectMongoDump
from ProcessCEFold.PreprocessCEProject import PreprocessCEProject
from ProcessCEFold.PreprocessConcreteMongo import PreprocessConcreteMongo
from ProjectImportClassifierFold.ClassifierPipeCovid import ClassifierPipeCovid
from ProjectWE.WEclient import WEclient
from ProjectWE.ImporterMongo import ImporterMongo
from Evaluation.Eval import eval_proj

def retrieve_words(int_word,model):
	try:
		a = [x[0] for x in model.wv.most_similar(int_word, topn=10)]
	except:
		print('error retrieving value, not in corpus')
		exit()
	return a

if __name__=='__main__':
	
	myProjImporter = ImporterProject()
	handle_list = myProjImporter.runImport()
	myProjExtractor = TextExtractorProject(handle_list)
	sents, CE, docs_dict = myProjExtractor.runExtractor()

	myProjDump = DumperProject(['words_last_comp','CER_last_comp','docsdict_last_comp'],[sents, CE,docs_dict])
	
	myProjDump.setImportData(ProjectMongoDump(['words_last_comp','CER_last_comp','docsdict_last_comp'],[sents, CE,docs_dict])) # delete this to use FS
	
	myProjDump.runDump()
	
	print(" ### First step end")
	
	myPreprocessCEProject = PreprocessCEProject()

	myPreprocessCEProject.setPreprocess(PreprocessConcreteMongo()) # delete this to use FS
	
	sarsCovDataset = myPreprocessCEProject.runPreprocess()
	covidPipe = ClassifierPipeCovid(sarsCovDataset)
	covidPipe.runImputizerClient()
	covidPipe.runScalerClient()
	classifiedDS = covidPipe.runClassifierClient()
	
	myProjDump = DumperProject(['classifiedCER_last_comp'],[classifiedDS])

	myProjDump.setImportData(ProjectMongoDump(['classifiedCER_last_comp'],[classifiedDS])) # delete this to use FS

	myProjDump.runDump()
	
	print(" ### Second step end")

	myProjDump.runJoinRes(['words','CER','classifiedCER','docsdict']) # handle for classified info
	weobj = WEclient(word2vec.Word2Vec)

	weobj.setImporter(ImporterMongo()) # delete this to use FS

	CE,words = weobj.runImporter()
	weobj.runWE(CE,words)
	weobj.runPCA()
	weobj.export_res()
	print(" ### Iteration end")



	print("Evaluation")
	model = eval_proj(words,CE)

	w = retrieve_words('covid-19',model)

	print()
	print(w)
	
	w_t = weobj.query_docsdict(w)
	print(w_t)
	print()
	


	




