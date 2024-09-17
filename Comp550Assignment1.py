import re
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ShuffleSplit,GridSearchCV,train_test_split

#nltk.download('punkt_tab')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

with open('facts.txt','r',encoding='utf-8') as f:
    facts = f.read().split('\n')

with open('fakes.txt','r',encoding='utf-8') as f:
    fakes = f.readlines()
data = facts + fakes
#print(data)

#define models


def text_lemma(s):
    words = nltk.word_tokenize(s)
    result = []
    for w in words:
        result.append(lemmatizer.lemmatize(w))

    return ' '.join(result)

#preprocessing
#extract labels from sentences
features = []
targets = []
lemmatizer = WordNetLemmatizer()
for l in data:

    if '[FACT]' in l:
        targets.append(1)
        tmp = re.sub(r'\[FACT\]','',l).strip()

    if '[FAKE]' in l:
        targets.append(0)
        tmp = re.sub(r'\[FAKE\]', '', l).strip()

    tmp = re.sub(r'\.', '', re.sub(r'\,', '', re.sub(r'\"', '', tmp))).strip().lower() #remove punctuations and convert to lower case
    tmp = text_lemma(tmp).strip()
    features.append(tmp)
#print("\n")
#print(features)

#Vectorization(TF-IDF)
v = TfidfVectorizer(ngram_range=(1,3),stop_words='english')
X = v.fit_transform(features)
Y = np.array(targets)
#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#SVM hyperparameters tuning
def hyperpara_tuning_SVM(X,Y):
    params = [
        {'C':[0.001,0.1,1,10,100,1000], 'kernel':['linear']},
        {'C':[0.001,0.1,1,10,100,1000], 'gamma':[0.01,0.001,0.0001],'kernel':['rbf']},
        {'C': [0.01, 0.1, 1, 10,100,1000], 'degree': [2, 3, 4], 'kernel': ['poly']}
    ]
    clf = SVC(random_state=42)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(clf,params,cv=cv,n_jobs=-1,scoring='accuracy')
    grid_search.fit(X,Y)
    print("SVM")
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    return grid_search.best_estimator_ #use this on the actual training sets for training: grid_search.best_estimator_.fit(X,Y)

best_SVM = hyperpara_tuning_SVM(X_train,Y_train)
best_SVM.fit(X_train,Y_train)

test_score = best_SVM.score(X_test,Y_test)
print("Best Score on test sets",test_score)



