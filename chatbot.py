import numpy as np
import nltk
import string
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open("C:\\Users\\hsrak\\OneDrive\\Desktop\\April\\NLP_Class\\chatbot\\New Folder\\chatboat.txt", 'r', errors='ignore')
raw_doc = f.read()
raw_doc
# raw_doc = raw_doc.lower() #Converting entire text to lowercase
# nltk.download('punkt') #Using the Punkt tokenizer
# nltk.download('wordnet') #Using the wordnet dictionary
# nltk.download('omw-1.4')
# sentence_tokens = nltk.sent_tokenize (raw_doc)
# word_tokens = nltk.word_tokenize(raw_doc)


# Convert entire text to lowercase
raw_doc = raw_doc.lower()

# Download necessary NLTK resources
nltk.download('punkt')  # Using the Punkt tokenizer
nltk.download('wordnet')  # Using the WordNet dictionary
nltk.download('omw-1.4')  # Using the Open Multilingual WordNet

# Tokenize sentences
sentence_tokens = nltk.sent_tokenize(raw_doc)

# Tokenize words
word_tokens = word_tokenize(raw_doc)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_word_tokens = [word for word in word_tokens if word not in stop_words]

# Print filtered word tokens
print("Word tokens after removing stop words:")
print(filtered_word_tokens)
sentence_tokens[:1]
sentence_tokens
word_tokens[:5]
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens (tokens):
 return [lemmer.lemmatize (token) for token in tokens]
remove_punc_dict = dict((ord (punct), None) for punct in string.punctuation)

def LemNormalize(text):
 return LemTokens (nltk.word_tokenize (text.lower().translate (remove_punc_dict)))
greet_inputs = ('hello', 'hi', 'whas sup', 'how are you?')
greet_responses = ('Hi', 'Hey', 'Hey There!', 'There there!!')
def greet (sentence):
 for word in sentence.split():
    if word. lower() in greet_inputs:
       return random.choice (greet_responses)
    
    def response(user_response):
    robo1_response =''
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity (tfidf [-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
     robo1_response = robo1_response + "I am sorry. Unable to understand you!"
     return robo1_response
    else:
     robo1_response = robo1_response+ sentence_tokens[idx]
     return robo1_response
flag = True
print('Hello! I am the Retreival Learning Bot. Start typing your text after greeting to talk to me. For ending convo type bye!')

while flag:
    user_response = input().lower()
    
    if user_response!= 'bye':
        if user_response in ['thank you', 'thanks']:
            flag = False
            print('Bot: You are welcome..')
        else:
            if greet(user_response) is not None:
                print('Bot ' + greet(user_response))
            else:
                print(user_response)
                sentence_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('Bot: ', end='')
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print('Bot: Goodbye!')