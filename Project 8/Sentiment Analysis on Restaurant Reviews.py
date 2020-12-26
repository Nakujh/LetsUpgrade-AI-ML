#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')


# In[2]:


import numpy as np
import pandas as pd
import pickle 


# In[3]:


dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)
dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.dtypes


# In[6]:


dataset.groupby('Liked').size()


# In[7]:


dataset.isnull().sum()


# In[8]:


dataset['Review'][0]


# In[9]:


# Removing Numbers and Punctuations
import re
review = re.sub( '[^a-zA-Z]', ' ', dataset['Review'][0] )
print(review)


# In[10]:


review = review.lower()
review


# In[11]:


import nltk
# nltk.download('stopwords')


# In[12]:


from nltk.corpus import stopwords


# In[13]:


stopwords.words('english')


# In[14]:


len(stopwords.words('english'))


# In[15]:


review


# In[16]:


review = review.split()
review


# In[17]:


# List comprehension to remove stopwords
review1 = [ word for word in review if not word in set(stopwords.words('english')) ]
review1


# In[18]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review1 = [ ps.stem(word) for word in review1 ]
review1


# In[19]:


review2 = ' '.join(review1)
review2


# In[20]:


corpus1 = []
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3)
print(review2)
corpus1.append(review2)
print(corpus1)
X = cv.fit_transform(corpus1)
print(X.toarray())


# In[21]:


dataset.shape


# In[22]:


dataset.tail()


# In[23]:


import re
import nltk
# nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)
    print(review)
    corpus.append(review)


# In[24]:


type(review)


# In[25]:


corpus


# In[26]:


type(corpus)


# In[27]:


corpus_dataset = pd.DataFrame(corpus)
corpus_dataset.head()


# In[28]:


corpus_dataset['corpus'] = corpus_dataset
corpus_dataset.head()


# In[29]:


corpus_dataset = corpus_dataset.drop([0], axis=1)
corpus_dataset.head()


# In[30]:


corpus_dataset.to_csv("corpus_dataset.csv")


# In[31]:


type(corpus_dataset)


# In[32]:


# Create a Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)


# In[33]:


X = cv.fit_transform(corpus).toarray()
X[0]


# In[34]:


X


# In[35]:


dataset.head()


# In[36]:


cv.get_feature_names()


# In[37]:


len(cv.get_feature_names())


# In[38]:


y = dataset.iloc[:,1].values
y


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[40]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[41]:


y_pred = classifier.predict(X_test)


# In[42]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[43]:


confusion_matrix(y_test,y_pred)


# In[44]:


accuracy_score(y_test,y_pred)


# In[45]:


Review = "nice service"
input1 = [Review]

input_data = cv.transform(input1).toarray()

input_pred = classifier.predict(input_data)

if input_pred[0]==1:
    print("Review is Positive")
else:
    print("Review is Negative")


# Review = "long waiting time"
# input1 = [Review]
# 
# input_data = cv.transform(input1).toarray()
# 
# input_pred = classifier.predict(input_data)
# 
# if input_pred[0]==1:
#     print("Review is Positive")
# else:
#     print("Review is Negative")

# Review = input("Enter a sentence?")
# input1 = [Review]
# 
# input_data = cv.transform(input1).toarray()
# 
# input_pred = classifier.predict(input_data)
# 
# if input_pred[0]==1:
#     print("Review is Positive")
# else:
#     print("Review is Negative")

# # Model Deployment

# In[46]:


# Here, we will save the 'lr' model to disk as 'model.pkl'

pickle.dump(classifier,open('RestaurantReviewSentimentAnalyzer.pkl','wb'))
# Dump this model by the name "model.pkl" in the systems HDD and while doing this
# write this file using "write bytes" mode.


# In[47]:


# Lets now try to load the same model by reading it from the system
# and using it for prediction

model2 = pickle.load(open("RestaurantReviewSentimentAnalyzer.pkl","rb"))


# In[50]:


Review = input("Enter a sentence?") 

input1 = [Review]

input_data = cv.transform(input1).toarray()

input_pred = model2.predict(input_data)

if input_pred[0]==1: 
    print("Review is Positive") 
else: 
    print("Review is Negative")

