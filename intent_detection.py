import numpy as np 
import pandas as pd 
from IPython.display import display
import re
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
import string
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import nltk
import ssl
import pickle
stopwords = stopwords.words('english')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def get_intent(input):
    
    #necessary nltk and spacy package for the upcoming cleanup part
    nltk.download('stopwords')
    nlp = spacy.load('en_core_web_sm')
    punctuations = string.punctuation

    #clean text using spacy (lemma, lowercasing, removing stopwords)
    def cleanup_text(docs, logging=False):
        texts = [] 
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print(" %d out of %d documents." % (counter, len(docs)))
            counter += 1
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc] #Lemmatization and lowercasing
            tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations] #removing stopwords
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)

        # further cleaning the text to apply a word2vec model
    def cleanup_text_word2vec(docs, logging=False):
        sentences = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents" % (counter, len(docs)))
            doc = nlp(doc, disable=['tagger'])
            doc = " ".join([tok.lemma_.lower() for tok in doc])
            doc = re.split("[\.?!;] ", doc) #splitting text into sentences and words
            doc = [re.sub("[\.,;:!?]", "", sent) for sent in doc] #substitute punctuations 
            doc = [sent.split() for sent in doc]
            sentences += doc
            counter += 1
        return sentences #list of lists of words for the w2v

    # load the model from disk
    vectorizer = TfidfVectorizer()
    filename_vecfit = 'vec_fit.sav'
    vec_fit = pickle.load(open(filename_vecfit, 'rb'))

    # Define function to create word vector representation of a given cleaned piece of text by averaging the tf-idf vectors
    # take in cleaned piece of text
    def create_average_vec(doc): 
        #initialize empty average vector of same length as number of features in vec_fit
        average = np.zeros(111, dtype='float32')
        num_words = 0.
        #iterate over each word in doc and add tf-idf vector to average vector
        for word in doc.split():
            average = np.add(average, vec_fit.transform([word]).toarray()) 
            num_words += 1.
        if num_words != 0.:
            #divide average vector by number of words to get average vector repr. of text
            average = np.divide(average, num_words)
        return average #return converted cleaned data to use as input for SVC Model

    # load the model from disk
    filename_model = 'intent_detection_model.sav'
    loaded_model = pickle.load(open(filename_model, 'rb'))

    cleanup = cleanup_text([input], logging=True)
    intent_categories = ['PlacementTeam', 'TeamPlacement', 'YearHost', 'bye', 'firstPlace', 'greeting', 'thankYou', 'year(avg)Goals', 'yearGoals', 'yearMatches']

    # converting the cleaned data to vector
    cleanup_vec = np.zeros((1, 111), dtype="float32")  # 19579 x 300
    print(cleanup_vec)
    for i in range(len(cleanup)):
        cleanup_vec[i] = create_average_vec(cleanup[i])
    # predict category of new input with the trained model    
    y = loaded_model.predict([cleanup_vec[0]])

    #get category of input in original form (not in encoded form)
    intent = intent_categories[int(y)]
    return intent

# Code auf Basis von: https://www.kaggle.com/code/taranjeet03/intent-detection-svc-using-word2vec/notebook#)