import queue
import subprocess
import re
import nltk
import string
import time
import pubchempy as pcp



from ProjectExtractorFold.CEWordExtraction import CEWordExtraction
from functools import reduce
from Utils.thread_manager import *
from urlextract import URLExtract
from nltk.tokenize import word_tokenize
from chemdataextractor import Document
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer




class CEWordExtractionProject(CEWordExtraction):
    nltk.download('punkt')
    nltk.download('stopwords')
    cachedStopWords = stopwords.words("english")+['ns','nq','date','e-mail','et','al','dr','e.g','i.e','n/a'] # avoid thread exceptions
    yet_visited = {}
    porter = PorterStemmer()


    r=queue.Queue()

    def __init__(self,fnames,max_nthreads_analyze=8):

        self.max_nthreads_analyze = max_nthreads_analyze
        self.fnames=fnames
        self.fname_lists = [fnames[i*max_nthreads_analyze:i*max_nthreads_analyze+max_nthreads_analyze] for i in range(len(fnames)//max_nthreads_analyze)]
        if len(fnames)%max_nthreads_analyze != 0:
            self.fname_lists+=[fnames[(len(fnames)//max_nthreads_analyze)*max_nthreads_analyze:]]

    @staticmethod
    def pdf2txt_pdfminer(f):
        result = subprocess.run(['pdf2txt.py',f], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        text = result.stdout.decode()
        err=result.stderr.decode()
        return text,err

    @staticmethod
    def pdf_analysis_runner(fname):
        print('[START] Processing => {}'.format(fname))
        text,err = CEWordExtractionProject.pdf2txt_pdfminer(fname)
        if len(err)==0:
            text = CEWordExtractionProject.text_preprocessing(text)# remove url
            ret_v_set = CEWordExtractionProject.text_normalization(text)
            if len(ret_v_set)>0:
                CEWordExtractionProject.r.put(ret_v_set)
        else:
            print('Error in parsing document, skip {}, err= {}'.format(fname,err))
        print('[END] Processing => {}'.format(fname))



    @staticmethod
    def text_preprocessing(text):
        return CEWordExtractionProject.remove_url(text)
    @staticmethod
    def remove_url(text):	
        extractor = URLExtract()
        urls = extractor.find_urls(text)
        for url in urls:
            text = text.replace(url, '')
        text =  re.sub('doi:(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text, flags=re.MULTILINE)
        return text


    @staticmethod
    def text_normalization(text):
        ret_v_set = CEWordExtractionProject.sentence_tokenizer(text)
        return ret_v_set

    @staticmethod
    def sentence_tokenizer(text):
            
        sentences = [s.lower() for s in sent_tokenize(text) if len(s)>0]
        ret_v = []
        for sentence in sentences:
            tok = word_tokenize(sentence)
            tok = CEWordExtractionProject.token_preprocessing(tok)
            sentence = ' '.join(tok)
            if len(tok)<3: continue
            tmp = CEWordExtractionProject.identify_ce(tok)
            ret_v += [(tok,sentence,tmp)]
        return ret_v

    @staticmethod
    def float_comp(s):
        s1 = ''.join([t if t not in string.punctuation+'e' else ',' for t in s])
        return all([t.isnumeric() for t in s1.split(',')])

    @staticmethod
    def token_preprocessing(tokens):
        tokens = [t for t in tokens if len(t)>2 and not CEWordExtractionProject.is_number(t) and not CEWordExtractionProject.is_number(t[:-1]) and not CEWordExtractionProject.is_number_sym(t) and CEWordExtractionProject.is_printable(t)]
        tokens = [t for t in tokens if '\\' not in t and '/' not in t]
        tokens = [t for t in tokens if  not (len(t)==2 and t[0] in string.ascii_lowercase and t[1]=='.')]
        tokens = [t for t in tokens if t[-1] not in string.punctuation]
        tokens = [t for t in tokens if  not CEWordExtractionProject.float_comp(t)] # check if token is a float/number composition
        tokens = [t for t in tokens if len(set(t))>1] # avoid terms composed by single char
        tokens = [t for t in tokens if t not in CEWordExtractionProject.cachedStopWords  ]
        tokens = [t for t in tokens if 'doi' not in t and 'cid' not in t]
        return tokens

    @staticmethod
    def is_number_sym(s):
        return all([c.isnumeric() or c in string.punctuation+'\xe2' for c in s]) # is not - char is a special char
    @staticmethod
    def is_printable(w):
        return all([c in string.printable for c in w])


    @staticmethod
    def identify_ce(tokens):
        if len(tokens)==1:
            corp = Document(tokens[0]).cems 
        else:
            corp = Document(' '.join(tokens)).cems	
        return [(el,el.text) for el in corp]

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False or s.isnumeric()

    @staticmethod
    def clearQueue():
        CEWordExtractionProject.r = queue.Queue()

    @staticmethod
    def pcp_normalization(tok,sentence,tmp):
        new_sent,ce = [],{}
        found_pcp,found_el = [],[]
        i=0
        if len(tmp)==0:
            return [(CEWordExtractionProject.porter.stem(t),t) for t in word_tokenize(sentence)],{}
        for el in tmp:
            if i%5==0:
                start = time.time()
            if i%5==4:
                end = time.time()
                print('Start sleep for pubchem')
                if 1-(end-start)>0:
                    time.sleep(1-(end-start))
                print('End sleep for pubchem')
            comp = el[1]
            #print('el1 = {}'.format(el[1]))
            if el[0] not in CEWordExtractionProject.yet_visited.keys():
                try:
                    c=pcp.get_compounds(comp, 'name') 
                    i+=1
                    print('[PCP] ok=>\t[{}]'.format(comp))
                except Exception as e:
                    c=False
                    print('[PCP] fail=>\t[{}] for {}'.format(comp,e))
                CEWordExtractionProject.yet_visited[el[0]]=c
            else:
                print('Already visited')
                c=CEWordExtractionProject.yet_visited[el[0]]


            if c and len(c)>0 and c[0].iupac_name!=None:
                found_el+=[el]
                found_pcp+=[c]
        #change sentence pay attention start and end refer to original sent
        new_tok = []
        new_sentence = ''
        start_t = 0
        if len(found_el)>=2:
            print('Interesting out')

        list_res = zip(found_el,found_pcp)
        list_res = sorted(list_res,key=lambda x: x[0][0].start)
        fint = 0
        new_t=[]
        for el,c in list_res:
            
            iupac_n = ''.join(word_tokenize(c[0].iupac_name.replace(' ','-')))
            print('Iupac name = {} : Replace {}'.format(iupac_n,sentence[el[0].start:el[0].end]))
            new_sentence+=sentence[start_t:el[0].start]+iupac_n
            new_t+=word_tokenize(sentence[start_t:el[0].start])+[iupac_n]
            #print('sent[start] = {}'.format(sentence[el[0].start]))
            #print('start_t = {}'.format(start_t))
            #print('new_sent = {}'.format(new_sentence))
            start_t += (el[0].end)
            fint = el[0].end
            nk = iupac_n
            if nk not in ce.keys():	
                ce[nk]=[c[0].isomeric_smiles,iupac_n]
        

        if len(found_el)==0: 
            new_sentence=sentence
            new_t=word_tokenize(sentence)
        else: 
            new_sentence+=sentence[fint:]
            new_t+=word_tokenize(sentence[fint:])


        
        

        print('sent = {}'.format(sentence))
        print('new sent = {}'.format(new_sentence))

        print('len found_el = {}'.format(len(found_el)))

        print(new_t)
            
        assert ''.join(word_tokenize(new_sentence)) == ''.join(new_t)	

        iup_list = []
        for el,c in zip(found_el,found_pcp):
            iupac_n = ''.join(word_tokenize(c[0].iupac_name.replace(' ','-')))
            iup_list+=[iupac_n]
            if iupac_n not in new_t:
                print([(iupac_n,t) for t in new_t])
                print("DEADLY BUG FOR EXTRACT CER")
                exit()

        #tok = token_preprocessing(tok)	
            
        #tokens = [porter.stem(w) for w in tokens]

        # TAG each tokens of a sentence using nltk (call append wtag)
        new_t = [(CEWordExtractionProject.porter.stem(t),t) if t not in iup_list else (t,t)  for t in new_t]
        return new_t,ce

    def runCEWordExtraction(self):
        sentences_for_pcp = []
        CEWordExtractionProject.clearQueue()
        for fname_list in self.fname_lists:
            pool = ThreadPool(len(fname_list))
            pool.map(CEWordExtractionProject.pdf_analysis_runner, fname_list)
            pool.wait_completion()
            items = [CEWordExtractionProject.r.get() for _ in range(CEWordExtractionProject.r.qsize())]
            sentences_for_pcp += items


        sentences_for_pcp = reduce(lambda x,y:x+y, sentences_for_pcp)
        print(sentences_for_pcp)
        docs = []
        CER = {}
        docs_dict = {}
        for tok,sentence,tmp in sentences_for_pcp:
            docs_tot,CE_t = CEWordExtractionProject.pcp_normalization(tok,sentence,tmp)
            print("Complete sentence pcp cycle sentence = {}".format(sentence))
            docs_t = [x[0] for x in docs_tot]
            docs+=[docs_t]
            docs_dict.update({k:el for k,el in set(docs_tot)})
            CER.update(CE_t)

        assert set(CER.keys()).issubset(set(reduce(lambda x,y:x+y,docs)))

        return docs, CER, docs_dict



