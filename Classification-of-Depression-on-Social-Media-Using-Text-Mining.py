#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv


# In[2]:


tweet_data=[]
a=[]
b=[]
for lines in open(r'C:\Users\LENOVO\Documents\Github\Classification-of-Depression-on-Social-Media-Using-Text-Mining\data\tweetdata.txt','r'):
    try:
        tweet=json.loads(lines)
        a.append(tweet['id_str'])
        b.append(tweet['text'])
    except:
        continue


# In[3]:


data=pd.DataFrame({'Id':a,'Text':b})


# In[4]:


data.head()


# In[5]:


import re
import string
for i in range(len(data)):
    re_emoji=re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    data['Text'][i]=re.sub(re_emoji,'',data['Text'][i])
    data['Text'][i]=data['Text'][i].translate(str.maketrans('','',string.punctuation))
    data['Text'][i]=data['Text'][i].lower()


# In[6]:


data.head()


# In[7]:


dict_file=pd.read_csv(r'C:\Users\LENOVO\Documents\Github\Classification-of-Depression-on-Social-Media-Using-Text-Mining\data\dictionary.tsv',delimiter='\t',header=None)


# In[8]:


dict_file.head()


# In[9]:


from nltk.tokenize import word_tokenize
id=[]
t=[]
c=1
print(len(data['Text']))
for i in range(len(data['Text'])):
    text=word_tokenize(data['Text'][i])
    sum=0
    words=0
    for j in range(len(text)):
        for k in range(len(dict_file)):
            if text[j]==dict_file[2][k]:
                if dict_file[5][k]=='positive':
                    sum+=1
                    words+=1
                elif dict_file[5][k]=='negative':
                    sum-=1;
                    words+=1
                else:
                    words+=1
    print("Yes", c)
    c+=1
    if words!=0:
        new=sum/words
        if new>=0.2:
            new=1
        elif (new<0.2) and (new>-0.5):
            new=0
        elif new<=-0.5:
            new=-1
        else:
            print('****')
    id.append(data['Id'][i])
    t.append(new)
d=pd.DataFrame({'id':id,'sentiment':t})
d.to_excel(r'output.xlsx',index=False)


# In[10]:


new_data=d.copy()


# In[11]:


new_data.head()


# In[12]:


new_data.shape


# In[13]:


x=data['Text']
y=new_data['sentiment']


# In[14]:


print(x)


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(stop_words='english')
X=vectorizer.fit_transform(x)


# In[16]:


f=X.toarray()


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(f,y,test_size=0.2,random_state=0)


# In[18]:


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(x_train,[int(r) for r in y_train])
y_pred=nb.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))


# In[19]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,[int(r) for r in y_train])
y_pred=dtree.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))


# In[20]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=2,random_state=0)
rf=rf.fit(x_train, [int(i) for i in y_train])
y_pred=rf.predict(x_test)
print(y_pred)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))


# In[ ]:




