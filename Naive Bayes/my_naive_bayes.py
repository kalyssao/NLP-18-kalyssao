
# coding: utf-8

# In[59]:


import re
from math import log, exp


# In[60]:


def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


# In[61]:


def check_word(word, bag, aDict, vocab):
    product = 1
    denom = len(bag) + len(vocab)
    
    if(word in vocab):
        if(word in bag):
            num = aDict[word]+1
        else:
            num = 1
        
        wordProb = num/denom
        
        return wordProb
    
    else:
        wordProb = 1
        return wordProb


# In[84]:


def my_naive_bayes(testfile):
    filenames = ['imdb_labelled.txt','yelp_labelled.txt','amazon_cells_labelled.txt']
    
    docSent = []
    docVocab = []
    
    negVocab = []
    posVocab = []
    
    negCount = 0
    posCount = 0
    
    with open('train.txt','w') as outfile:
        for file in filenames:
            with open(file) as infile:
                for line in infile:
                    outfile.write(line)
    
    with open('train.txt','r') as g:
        for line in g:
            line = line.lower()
            line = line.strip('\n')
            line = line.replace('\t',' ')
            
            if('0' in line):
                line = line[0:len(line)-1]
                inter = line.split()
                negVocab = negVocab + inter
                negCount +=1
                
            elif('1' in line):
                line = line[0:len(line)-1]
                inter = line.split()
                posVocab = posVocab + inter
                posCount += 1
                
            line = line[0:len(line)-1]    
            docSent.append(line)
            docVocab = docVocab + line.split()
        
        docVocab = remove_duplicates(docVocab)
        
        total = len(docSent)
        probNeg = negCount/total
        probPos = posCount/total
        
        negDict = {}
        posDict = {}
        
        for word in negVocab:
            if word in negDict:
                negDict[word] = negDict[word] + 1
            else:
                negDict[word] = 1
        
        for word in posVocab:
            if word in posDict:
                posDict[word] = posDict[word] + 1
            else:
                posDict[word] = 1
    
    features = []
    mulArrayNeg = []
    mulArrayPos = []
    
    probArray = []
    results = open('results.txt','w')
    with open(testfile,'r') as k:
        for line in k:
            print(line)
            features = features + line.split()

            prodNeg = 1
            prodPos = 1
            maxDict = {}
            negProbArray = []
            posProbArray = []
            for word in features:
                
                #check word
                negProbArray.append(check_word(word, negVocab, negDict, docVocab))
                posProbArray.append(check_word(word, posVocab, posDict, docVocab))
            
            for i in negProbArray:
                prodNeg *= i
            for i in posProbArray:
                prodPos *= i
            prodNeg = prodNeg * probNeg
            prodPos = prodPos * probPos
            
            maxDict["0"] = prodNeg
            maxDict["1"] = prodPos
            
            maximum = max(maxDict, key=maxDict.get) 
            
            results.write(line.strip('\n') + '\t' + maximum + '\n')            


# In[85]:


my_naive_bayes('test.txt')

