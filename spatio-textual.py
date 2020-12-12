#!/usr/bin/env python
# coding: utf-8

# In[11]:


#! pip install gensim
#import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')


# In[12]:


import tracemalloc


# In[13]:


import csv
import numpy as np
from nltk.corpus import wordnet
from sklearn.cluster import SpectralClustering as spcl
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import AgglomerativeClustering as agcl
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


# In[14]:


def California():
 datasetfile='f:/junaid/jcl1.csv'
 with open(datasetfile, newline='', encoding='ANSI') as f:
     reader = csv.reader(f)
     data = list(reader)


 data=np.array(data)
 print(data[0])
 print(len(data))
 textdata=data[1:lentext+1,3]
 spacdata=data[1:lentext+1,1:3]
 return textdata, spacdata.astype('float')

def World():
 datasetfile='f:/junaid/new.csv'
 with open(datasetfile, newline='', encoding='ANSI') as f:
     reader = csv.reader(f)
     data = list(reader)


 data=np.array(data)
 print(data[0])
 print(len(data))
 textdata=data[1:lentext+1,18]
 spacdata=data[1:lentext+1,0:2]
 return textdata, spacdata.astype('float')


# In[15]:


def tokenize_only(sentence):
    return [token for token in sentence.split()]

def tokenize_remove_stopwords(sentence):
    return [token for token in sentence.split() if token not in STOP_WORDS]

def tokenize_lematize_remove_stopwords(sentence):
    return [lemmatizer.lemmatize(token) for token in sentence.split() if token not in STOP_WORDS]

def tfidfvecs():
 vectorizer = TfidfVectorizer(max_features=vecdim,tokenizer=tokenize)
 vectors = vectorizer.fit_transform(textdata)
 return vectors

def wordtovecs():
 wtkn=[[]]*len(textdata)
 for i in range(len(textdata)):
  wtkn[i]=tokenize(textdata[i])
 word2vec = Word2Vec(wtkn, min_count=1,size=vecdim)
 wvctrs=[[]]*len(wtkn)
 for i in range(len(wtkn)):
  wvctrs[i]=np.mean(word2vec.wv[wtkn[i]],axis=0)
 
 return np.array(wvctrs)


# In[16]:


#textdata[0].split() # tokenize

def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compeucsim(indata):
 eucdistcomp=pdist(indata, metric='euclidean')
 spceucdistmat=squareform(eucdistcomp)#/max(eucdistcomp))
 spceucsim=np.ones(np.shape(spceucdistmat))-spceucdistmat #normalize(spceucdistmat,axis=1)
 return spceucsim

def compcosine(indata):
 #normin=normalize(indata,axis=1)
 return cosine_similarity(indata,indata)

def comprbf(indata):
 return rbf_kernel(indata,indata)

def compjacsim(idx):
 indata=textdata[idx]
 jacsimimat=np.zeros((len(indata),len(indata)))
 for i in range(len(indata)):
     for j in range(len(indata)):
         jacsimimat[i,j]=get_jaccard_sim(indata[i],indata[j])
        
 return jacsimimat, indata

def tfidfvecs_cosine(idx):
 indata=tfvectors[idx]
 return compcosine(indata), indata

def wordtovecs_cosine(idx):
 indata=w2vectors[idx]
 return compcosine(indata), indata

def wordtovecs_euc(idx):
 indata=w2vectors[idx]
 return compeucsim(indata), indata


# In[17]:


def agloclust(simimat,ncl):
 distmat=np.ones(np.shape(simimat))-simimat
 return agcl(n_clusters=ncl,affinity='precomputed',linkage = 'average').fit_predict(distmat)

def specclust(simimat,ncl):
 return spcl(n_clusters=ncl,affinity='precomputed').fit_predict(simimat)

def q_silhouette(dists,clusts,points):
 #return silhouette_score(dists,clusts,metric="precomputed")
 return silhouette_score(points,clusts)

def q_silhouette_cosine(dists,clusts,points):
 #return silhouette_score(dists,clusts,metric="precomputed")
 return silhouette_score(points,clusts,metric='cosine')

def q_calinski_harabasz(dists,clusts,points):
 return calinski_harabasz_score(points,clusts)

def q_davies_bouldin(dists,clusts,points):
 return davies_bouldin_score(points,clusts)


# In[18]:


def mainfunc():
 tracemalloc.start()
 start_time = time.time()
 txtsimimat, textembed=textaffinity(np.arange(len(textdata)))
 start_time1 = time.time()
 spcsimimat=spacaffinity(spacdata)
 sptime = time.time() - start_time1
 avgsim=(txtsimimat*(1-spwt))+(spcsimimat*spwt)
 cn=clusteralgo(avgsim,ncln)
 thyb = time.time() - start_time
 current, peak = tracemalloc.get_traced_memory()
 pkmem_hyb=peak / 10**6
 tracemalloc.stop()
 #hybrid clustering ends heirarchial starts using space similarity from above
 pkmem_her=[[]]*(ncls+1)
 tracemalloc.start()
 start_time_spcl = time.time()
 cs=clusteralgo(spcsimimat,ncls)
 spcltime= time.time() - start_time_spcl 
 ther=sptime+spcltime
 current, peak = tracemalloc.get_traced_memory()
 pkmem_her[ncls]=peak / 10**6
 tracemalloc.stop()
 spclst=[[]]*ncls
 spcltx=[[]]*ncls
 scljsim=[[]]*ncls
 ct=[[]]*ncls
 hcl=np.zeros(lentext)
 for i in range(ncls):
     spclst[i]=np.where(cs==np.unique(cs)[i])[0]
     spcltx[i]=textdata[spclst[i]]
     tracemalloc.start()
     start_time2 = time.time()
     scljsim[i], subtextembed_temp=textaffinity(spclst[i])
     subtxtime = time.time() - start_time2
     current, peak = tracemalloc.get_traced_memory()
     pkmem_her[i]=peak / 10**6
     tracemalloc.stop()
     ther=ther+subtxtime 
     try:
         ct[i]=clusteralgo(scljsim[i],nclt)
     except ValueError:
         ct[i]=[0]
         #print(ct[i])
     try:
         hcl[spclst[i]]=ct[i]+(nclt*i)
     except TypeError:
         hcl[spclst[i]]=nclt*i
         #print(nclt*i)

 avgdist=1-avgsim
 txtdist=1-normalize(txtsimimat,axis=1)
 np.fill_diagonal(txtdist, 0)
 spcdist=1-normalize(spcsimimat,axis=1)
 np.fill_diagonal(spcdist, 0)
 #'''
 pm1=jaccard_score(cn, hcl,average='micro') # micro, macro, weighted
 pm2=f1_score(cn, hcl,average='micro') # micro, macro, weighted
 for a in range(len(quality_func_d)):#len(spwt_d)
  quality_func=quality_func_d[a]
  pm3=quality_func(txtdist,hcl,textembed)
  pm4=quality_func(txtdist,cn,textembed)
  pm5=quality_func(spcdist,hcl,spacdata)
  pm6=quality_func(spcdist,cn,spacdata)
  pmhyb=pm4+pm6
  pmher=pm3+pm5
  #print(ncln,nclt,ncls,clusteralgo.__name__,textaffinity.__name__,spacaffinity.__name__,pm1,pm2,pm3,pm3-pm1)
  csvfilepath='f:/junaid/time-mem-qual_vs_samples.csv'
  with open(csvfilepath, 'a', newline='') as csvfile:
      res = csv.writer(csvfile)
      res.writerow(['peak_memory_her_MB']+['peak_memory_hyb_MB']+['dataset']+['tokenize']+['hybrid-spweight']+['time_hybrid']+['time_heirarchy']+['num_samples']+['hybclusters']+['textclusts']+['spaceclusts']+['clustering_algo']+['text_simmilarity']+['space_similarity']+['jacard_overlap']+['f1_overlap']+['unified-quality_hybrid']+['unified-quality_herarchy']+['sil_text_her']+['qual_text-hyb']+['qual_space-her']+['qual_space-hyb']+['quality-metric'])
      res.writerow([np.max(pkmem_her)]+[pkmem_hyb]+[dataload.__name__]+[tokenize.__name__]+[spwt]+[thyb]+[ther]+[lentext]+[ncln]+[nclt]+[ncls]+[clusteralgo.__name__]+[textaffinity.__name__]+[spacaffinity.__name__]+[pm1]+[pm2]+[pmhyb]+[pmher]+[pm3]+[pm4]+[pm5]+[pm6]+[quality_func.__name__])
 #'''
 return


# In[21]:


STOP_WORDS=set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer() 
tokenize_d=[tokenize_only,tokenize_remove_stopwords,tokenize_lematize_remove_stopwords]
vecdim=100 # tfidf/w2vec vectors dimenssion
spacaffinity_d=[compcosine, comprbf, compeucsim]
#textaffinity_d=[compjacsim, tfidfvecs_cosine, wordtovecs_cosine]
textaffinity_d=[wordtovecs_euc, wordtovecs_cosine]
clusteralgo_d=[agloclust, specclust]
ncln_d=[30,60,90]
ncls_d=[5,10,15]
lentext_d=[500,1000,5000,10000]
spwt_d=[0.25,0.5,0.75]
dataload_d=[California,World]
quality_func_d=[q_silhouette,q_silhouette_cosine]#q_calinski_harabasz]#,q_davies_bouldin]

#application oriented evaluation e.g spacial coverage or texutual  summary
# memory evaluation

for a in range(1):#len(spwt_d)
    for b in range(1):#len(ncls_d)
        for c in range(1):#len(ncln_d)
            for d in range(1):#len(spacaffinity_d)
                for e in range(1):#len(textaffinity_d)
                    for f in range(1):#len(clusteralgo_d)
                        for g in range(len(lentext_d)):#len(lentext_d)
                            for h in range(1):#len(tokenize_d)
                             for i in range(len(dataload_d)):#len(dataload_d)
                              tokenize=tokenize_d[0]
                              spwt=spwt_d[1] # weight for spacial info
                              ncls=ncls_d[1]  # number of spac clusters
                              ncln=ncln_d[1] # number of net clusters
                              nclt=int(ncln/ncls)  # number of text clusters
                              spacaffinity=spacaffinity_d[0] 
                              textaffinity=textaffinity_d[0]
                              clusteralgo =clusteralgo_d[0]
                              lentext=lentext_d[g]
                              dataload=dataload_d[i]
                              textdata,spacdata=dataload()                        
                              w2vectors=wordtovecs()
                              tfvectors=tfidfvecs()
                              mainfunc()


# In[20]:


'''
dataload=cali
lentext=12930
#cali,world]
#12931
textdata,spacdata=dataload()                        
import matplotlib.pyplot as plt
for i in range(lentext):
    #plt.scatter(spacdata[i,0],spacdata[i,1])
    plt.scatter(spacdata[i,1],spacdata[i,0]) # for california

#plt.title('world')
plt.title('california')
plt.xlabel("longitude")
plt.ylabel("latitude")
'''

