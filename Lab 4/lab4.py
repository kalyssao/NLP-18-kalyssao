#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Text normalisation
import re
from stop_words import get_stop_words

# Uses an existing list of stopwords to trim the test vocabulary
def remove_stopwords(words):
    stop_words = list(get_stop_words('en'))
    new_words = ' '.join([word for word in words.split() if word not in stop_words])

    return new_words

def rename(astring):
    new_name = astring[:-4] + "_normalised.txt"
    return new_name

# Normalises by removing capital letters, punctuation and stopwords
# and writes normalised text to a new file to be used as input for nb/lr
def normalise(textfile):
    newfilename = rename(textfile)
    outfile = open(newfilename, "w+")
    with open(textfile, "r") as file_to_normalise:
        for line in file_to_normalise:
            
            # Remove capital letters and punctuation
            line = line.lower()
            line = re.sub(r'[^\w\s]','',line)
            line = remove_stopwords(line)
            

            outfile.write(line)
            outfile.write('\n')

    file_to_normalise.close()
    outfile.close()


# In[2]:


# Initialising a dataframe in pandas containing the training vocabulary
import pandas as pd

df = pd.DataFrame()
filelist = ['amazon_cells_labelled.txt','yelp_labelled.txt','imdb_labelled.txt']

for item in filelist:
    frame = pd.read_csv(item, delimiter='\t',header=None, names=['sentiment','rating'])
    df = df.append(frame)


# In[3]:


# Defining x and y for use with count vectorizer
X = df.sentiment
y = df.rating


# In[4]:


# Importing the function for splitting data into test & train
from sklearn.model_selection import train_test_split

# Splitting the data into 80% training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# In[5]:


# Instantiating and fitting the unnormalised and normalised
# count vectorizers, which tokenizes the entered data to build a vocabulary
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(lowercase=False)
vect_norm = CountVectorizer()

# Learning with the training data (normalised and unnormalised)
vect.fit(X_train)
vect_norm.fit(X_train)

X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)


# In[6]:


# Importing metrics functionality, which will be used soon
from sklearn import metrics


# In[7]:


# Importing the logistic regression function
from sklearn.linear_model import LogisticRegression

# Instantiate the classifier
log_reg = LogisticRegression(solver='lbfgs')

# The model will learn the relationship between the input 
# and the observation when fit is called on the data
log_reg.fit(X_train_dtm, y_train)

# Testing the model using the remaining test data
lr_predicted = log_reg.predict(X_test_dtm)


# In[8]:


# Importing the naive bayes function
from sklearn.naive_bayes import MultinomialNB

# Instantiating the gaussian naive bayes classifier
naiveBayes = MultinomialNB()

# Training the model
naiveBayes.fit(X_train_dtm, y_train)

# Testing the model
nb_predicted = naiveBayes.predict(X_test_dtm)


# In[9]:


# General Overview of lr and nb
# For each line in a text file, append to a list and convert 
# the list to a matrix. Then, pass the matrix as an input to the trained model.
# Write the result to an output textfile whose name changes
# depending on the version specified and move to the next sentiment


# In[10]:


def lr(textfile, version):
    test_list = []
    infile = open(textfile, "r")
    
    outfile = open("results-lr-u.txt","w") if version == "u" else open("results-lr-n.txt","w")
    
    for sentiment in infile:
        test_list.append(sentiment)

        if(version == "n"):
            test_list_dtm = vect_norm.transform(test_list)
        
        test_list_dtm = vect.transform(test_list)
        
        result = log_reg.predict(test_list_dtm)
        outfile.write(str(result[0]))
        outfile.write('\n')
        
        test_list = []

    infile.close()
    outfile.close()


# In[11]:


def nb(textfile, version):
    test_list = []
    infile = open(textfile, "r")
    
    outfile = open("results-nb-u.txt","w") if version == "u" else open("results-nb-n.txt","w")
    for sentiment in infile:
        test_list.append(sentiment)

        if(version == "n"):
            test_list_dtm = vect_norm.transform(test_list)
        
        test_list_dtm = vect.transform(test_list)
        
        result = naiveBayes.predict(test_list_dtm)
        outfile.write(str(result[0]))
        outfile.write('\n')
        
        test_list = []
    infile.close()
    outfile.close()


# In[12]:


# System arguments check
import sys


# In[13]:


# If required length, check if NB or LR
# then, check if normalised or unnormalised
# If normalised, call normalise function on text file and pass result to 
# the specified model name of text file specified in commandline

if(len(sys.argv) != 4):
    print("Check that you have all your aguments")

else:
    inter = str(sys.argv[3])
    normalised_file = inter[:-4] + "_normalised.txt"

    if(sys.argv[1] == "nb"):
        if(sys.argv[2] == "u"):
            print("Unnormalised naive bayes")
            nb(sys.argv[3],"u")
            
        else:
            print("Normalised naive bayes")
            normalise(sys.argv[3])
            
            nb(normalised_file, "n")
    
    else:
        if(sys.argv[2] == "u"):
            print("Unnormalised logistic regression")
            lr(sys.argv[3],"u")
        
        else:
            print("Normalised logistic regression")
            normalise(sys.argv[3])
            lr(sys.argv[3],"n")

