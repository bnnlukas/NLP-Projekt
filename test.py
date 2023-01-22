input = 'Who was the champion of the worldcup in 2010?'
from intent_analysis import vectorizer
from intent_analysis import cleanup_text
from intent_analysis import Encoder
from intent_analysis import create_average_vec
from intent_analysis import filename
import pickle
import numpy as np
print('test')
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

cleanup = cleanup_text([input], logging=True)
intent_categories = list(Encoder.classes_)
# converting the cleaned data to vector
cleanup_vec = np.zeros((1, len(vectorizer.get_feature_names())), dtype="float32")  # 19579 x 300
for i in range(len(cleanup)):
    cleanup_vec[i] = create_average_vec(cleanup[i])
y = loaded_model.predict([cleanup_vec[0]])

intent = intent_categories[int(y)]

print(intent)

