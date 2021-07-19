import numpy as np
import pickle
import pandas as pd
import streamlit as st
import tweepy
import pandas as pd
import re
import emoji
import nltk
import datetime
import spacy
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

pickle_in=open("topic_modelling.pkl","rb")

topic_modelling=pickle.load(pickle_in)

def Predict_Topics(cv_arr,vocab_tf_idf):
   result=pd.DataFrame()
      

   # Implementation of LDA:
    
# Create object for the LDA class 
# Inside this class LDA: define the components:
   from sklearn.decomposition import LatentDirichletAllocation
   lda_model = LatentDirichletAllocation(n_components = 10, max_iter = 20, random_state = 20)

# fit transform on model on our count_vectorizer : running this will return our topics 
   X_topics = lda_model.fit_transform(cv_arr)

# .components_ gives us our topic distribution 
   topic_words = lda_model.components_
#  Define the number of Words that we want to print in every topic : n_top_words
   n_top_words = 10

   for i, topic_dist in enumerate(topic_words):
    
    # np.argsort to sorting an array or a list or the matrix acc to their values
    #np.argsort() is used to sort the index based on probablity
    sorted_topic_dist = np.argsort(topic_dist)
    
    # Next, to view the actual words present in those indexes we can make the use of the vocab created earlier
    # Creating vocabulary array which will represent all the corpus 
    #vocab_tf_idf = tf_idf_vectorizer.get_feature_names()
    topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]
    
    # so using the sorted_topic_indexes we ar extracting the words from the vocabulary
    # obtaining topics + words
    # this topic_words variable contains the Topics  as well as the respective words present in those Topics
    topic_words = topic_words[:-n_top_words:-1]
    print ("Topic", str(i+1), topic_words)
    result1=result.append({'topics':str(i+1),'topics_words':topic_words},ignore_index=True)
    result=result1.copy()
   return result 

def main():

    import numpy as np
    import pickle
    import pandas as pd
    import streamlit as st
    import tweepy
    import pandas as pd
    import re
    import emoji
    import nltk
    import datetime
    import spacy
    from nltk.corpus import stopwords 
    from nltk.stem.wordnet import WordNetLemmatizer
    import string

    st.title("Streamlit (Topic_Modelling App)")
    html_temp="""    <div style="background-color:tomato;padding:10px">
    <h1 style="color:white;text-align:center;">Topic_Modelling</h1>
    </div>"""
    st.markdown(html_temp,unsafe_allow_html=True)

#Creating search Box for search
    
    Date1 = st.sidebar.date_input('start date', datetime.date.today()-datetime.timedelta(days=7))
    Date2 = st.sidebar.date_input('end date', datetime. date.today())
    
    
    # set variables for keys and tokens to access the Twitter API
    mykeys = open('API Twitter.txt','r').read().splitlines()
    api_key = mykeys[0]
    api_key_secret = mykeys[1]
    access_token = mykeys[2]
    access_token_secret = mykeys[3]

    auth = tweepy.OAuthHandler(consumer_key = api_key, consumer_secret = api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)

#Featching the data from twitter
    search_words="news"
    date_since=Date1
    data_until=Date2
    tweets = tweepy.Cursor(api.search,q=search_words,lang="en",tweet_mode='extended',
                           since=date_since,until=data_until,result_type="recent").items(300)
              
              
 # Collect tweets
    tweets = tweepy.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(300)

# Iterate and print tweets
    s=[] 
    for tweet in tweets:
     s.append(tweet.text)
    print(s)  
    df=pd.DataFrame({'tweet':s})
    import nltk       
    words = set(nltk.corpus.words.words())
    
    tweet=np.array(df.tweet)
    cleaned_tweet=[]
    for i in df.tweet:
       no_punc_text = i.translate(str.maketrans('', '', string.punctuation))
       no_punc_text=re.sub("(RT)?(ht)?", "", no_punc_text) # to remove RT and ht word
       no_punc_text1=re.sub("[\W\d]", " ", no_punc_text) #to remove not word character and numbers
       no_punc_text2=re.sub("[^a-zA-Z]", " ", no_punc_text1) #to remove forien language word character
       no_punc_text2=" ".join(w for w in nltk.wordpunct_tokenize(no_punc_text2) \
         if w.lower() in words or not w.isalpha())
       cleaned_tweet.append(no_punc_text2)
    df['cleaned_tweet']=cleaned_tweet
    
    df1=df.copy() 
    corpus=df1.cleaned_tweet.unique()
    
    # import vectorizers
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# import numpy for matrix operation
    import numpy as np

# import LDA from sklearn
    from sklearn.decomposition import LatentDirichletAllocation
    
    #nltk.download('wordnet')
    # Lemmatize with POS Tag
    from nltk.corpus import wordnet
    import nltk
    #nltk.download('averaged_perceptron_tagger')

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
    
    # Apply Preprocessing on the Corpus

# stop loss words 
    stop = set(stopwords.words('english'))
    stop.update(["new","news",'via','take','first','one','say','time','big','see','come','good','another','today','make','get','great','could','like','make','set','end','dont'])
# punctuation 
    exclude = set(string.punctuation) 

# lemmatization
    lemma = WordNetLemmatizer() 

# One function for all the steps:
    def clean(doc):
    
    # convert text into lower case + split into words
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    
    # remove any stop words present
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
    
    # remove punctuations + normalize the text
        normalized = " ".join(lemma.lemmatize(word,get_wordnet_pos(word)) for word in punc_free.split())  
        return normalized

# clean data stored in a new list
    clean_corpus = [clean(doc).split() for doc in corpus]
    
    corpus1=[]
    for i in clean_corpus:
        doc=[]
        #j=i.split()
        for z in i:
            #print(len(z))
            if len(z)>2:
                doc.append(z)
        #print(doc)
        doc=" ".join(doc)
        doc1=doc.split()
        #print(doc1)
        corpus1.append(doc1)
    clean_corpus=corpus1
    
    abc = []  #to create single list
    for i in clean_corpus:
        abc.append(' '.join(i))                
                    
    abc2=" ".join(abc)
    
    nlp = spacy.load('en_core_web_sm')                
    one_block = abc2
    doc_block = nlp(one_block)
    
    #collecting 'PROPN','X','NOUN','ADJ' words                 
    final_corpus = [token.text for token in doc_block if token.pos_ in ('PROPN','X','NOUN','ADJ')]
    imp_words = set(final_corpus)

    # to remove the meaningless words 
    #doc=[]
    corpus1=[]
    for i in clean_corpus:
        doc=[]
        #j=i.split()
        for z in i:
            #print(len(z))
            if z in imp_words:
                doc.append(z)
        #print(doc)
        doc=" ".join(doc)
        doc1=doc.split()
        #print(doc1)
        corpus1.append(doc1)
    new_clean_corpus=corpus1                
    
    
    # Converting text into numerical representation
    tf_idf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)

# Converting text into numerical representation
    cv_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    
    # Array from TF-IDF Vectorizer 
    tf_idf_arr = tf_idf_vectorizer.fit_transform(clean_corpus)

# Array from Count Vectorizer 
    cv_arr = cv_vectorizer.fit_transform(clean_corpus)
    # Materialize the sparse data
    data_dense = cv_arr.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
    print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
    
    # Creating vocabulary array which will represent all the corpus 
    vocab_tf_idf = tf_idf_vectorizer.get_feature_names()
    
    # Creating vocabulary array which will represent all the corpus 
    vocab_cv = cv_vectorizer.get_feature_names()
    
    result=""
    if st.button("Search"):
      result=Predict_Topics(cv_arr,vocab_tf_idf)  
    st.success(st.write(result))
    
if __name__ == "__main__":
    main()








