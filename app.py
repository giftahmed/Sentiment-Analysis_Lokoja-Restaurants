# Importing essential libraries
from flask import Flask, request, render_template
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')


# Load the Random Forest  classifier model and CountVectorizer object from disk
filename = 'restaurant-sentiment-rf-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))


def predict_review(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if not message == "":
            if predict_review(message):
                return render_template('index.html', result = 1, message = message) #positive
            else:
                return render_template('index.html', result = 0, message = message) #negative
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)