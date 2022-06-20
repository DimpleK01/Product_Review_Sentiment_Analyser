import flask
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from importlib_metadata import method_cache

with open(f'model/sentiment_analysis.pkl', 'rb') as f:
    model = pickle.load(f)


with open(f'model/sentiment_analysis_tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

app= flask.Flask(__name__, template_folder='templates')

def preprocess(text):
    #remove special characters
        text=str.lower(text) 
        text= ''.join([i for i in text if not i.isdigit()])
        pattern=r'[^a-zA-z0-9\s]'
        text=re.sub(pattern,'',text) 

        #tokenizer
        text= text.split()

        #stemming
        porter = SnowballStemmer("english", ignore_stopwords=False)
        text= [porter.stem(word) for word in text]

        #lemmatizing
        lemmatizer = WordNetLemmatizer()
        text= [lemmatizer.lemmatize(word, pos = "a") for word in text]

        #stop word removal
        stop_words = stopwords.words('english')
        text = [word for word in text if not word in stop_words ] 

        #convert to tfidf vector
        text_vector = tfidf.transform(text)
        return text_vector

        

def prediction(text):
    
    preprocessed_text= model.predict(preprocess(text))
    List=list(preprocessed_text)
    sentiment = ''
    

    if List.count("positive")== List.count("negative"):
        sentiment = 'neutral'
    else :
        sentiment=max(set(List), key = List.count)
    
    return sentiment


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        review = flask.request.form['review']
        pred = prediction(review)
        return flask.render_template('main.html', original_input={'review': review}, result=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
