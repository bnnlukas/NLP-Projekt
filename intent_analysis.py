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

data = {'query':['Which Team became the 6th place in 2006?','Which Team was world champion in the year 2010?', 'Who won in 1990', 'Who became the world champion in 1945',
        'Which place did germany become in 1986', 'Which place became germany in 2014?','Which place was morroco in 2022?',
        'Who hosted the world cup in the year 2018?','Which nation was the host of the world cup in 1950?','Where was the world cup 1958',
        'in average, how many goals wher scored in 1998?','How many goals where scored 1962 on average', 'averagd Goals scored 1986',
        'How many goals were scored 1950?','How much goals in 2010?','Number of overall goals in 2022',
        'How many matches were played in 1930?', 'Number of matches played in 1954','Amount of Matches in the 2010 Worldcup', 
        'won','winner','champion','world champion'
        ],


       'category':['PlacementTeam','PlacementTeam','PlacementTeam','PlacementTeam',
       'TeamPlacement','TeamPlacement','TeamPlacement',
       'YearHost','YearHost','YearHost',
       'year(avg)Goals','year(avg)Goals', 'year(avg)Goals',
       'yearGoals','yearGoals','yearGoals',
       'yearMatches','yearMatches','yearMatches', 
       'firstPlace','firstPlace', 'firstPlace','firstPlace']}
stopwords = stopwords.words('english')
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def get_intent(input):
    df = pd.DataFrame(data)
    
    nltk.download('stopwords')
    nlp = spacy.load('en_core_web_sm')
    punctuations = string.punctuation

    # Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
    def cleanup_text(docs, logging=False):
        texts = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents." % (counter, len(docs)))
            counter += 1
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)
    train_cleaned = cleanup_text(df['query'], logging=True)

        # Define function to preprocess text for a word2vec model
    def cleanup_text_word2vec(docs, logging=False):
        sentences = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents" % (counter, len(docs)))
            # Disable tagger so that lemma_ of personal pronouns (I, me, etc) don't getted marked as "-PRON-"
            doc = nlp(doc, disable=['tagger'])
            # Grab lemmatized form of words and make lowercase
            doc = " ".join([tok.lemma_.lower() for tok in doc])
            # Split into sentences based on punctuation
            doc = re.split("[\.?!;] ", doc)
            # Remove commas, periods, and other punctuation (mostly commas)
            doc = [re.sub("[\.,;:!?]", "", sent) for sent in doc]
            # Split into words
            doc = [sent.split() for sent in doc]
            sentences += doc
            counter += 1
        return sentences
    train_cleaned_word2vec = cleanup_text_word2vec(df['query'], logging=True)
    corpus = [" ".join(i) for i in train_cleaned_word2vec]

    vectorizer = TfidfVectorizer()
    vec_fit = vectorizer.fit(corpus)

    text = "login is so cool"
    Y = vec_fit.transform([text])

    # Define function to create word vectors given a cleaned piece of text.
    def create_average_vec(doc):
        average = np.zeros(len(vectorizer.get_feature_names()), dtype='float32')
        num_words = 0.
        for word in doc.split():
            average = np.add(average, vec_fit.transform([word]).toarray())
            num_words += 1.
        if num_words != 0.:
            average = np.divide(average, num_words)
        return average
    
    # Create word vectors
    train_cleaned_vec = np.zeros((df.shape[0], len(vectorizer.get_feature_names())), dtype="float32")  # 19579 x 300
    for i in range(len(train_cleaned)):
        train_cleaned_vec[i] = create_average_vec(train_cleaned[i])
    
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(df['category'])

    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
    svclassifier = SVC(probability = True)
    svclassifier.fit(train_cleaned_vec, y_train)
    kfold = KFold(n_splits=5, shuffle=True)
    grid = GridSearchCV(svclassifier, param_grid, cv=kfold, refit=True, verbose=3)
    grid.fit(train_cleaned_vec, y_train)

    cleanup = cleanup_text([input], logging=True)
    intent_categories = list(Encoder.classes_)

    # converting the cleaned data to vector
    cleanup_vec = np.zeros((1, len(vectorizer.get_feature_names())), dtype="float32")  # 19579 x 300
    for i in range(len(cleanup)):
        cleanup_vec[i] = create_average_vec(cleanup[i])
    y = grid.predict([cleanup_vec[0]])

    intent = intent_categories[int(y)]

    return intent