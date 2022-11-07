# Importing essential libraries
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier


# Loading the dataset
#df = pd.read_csv('dfall.tsv', delimiter='\t', quoting=3)
df = pd.read_csv('dfcor.csv')

# Importing essential libraries for performing Natural Language Processing on 'Restaurant_Reviews.tsv' dataset
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Cleaning the reviews
corpus = []
for i in range(0,df.shape[0]):

  # Cleaning special character from the reviews
  review = re.sub(pattern='[^a-zA-Z]',repl=' ', string=df['Review'][i])

  # Converting the entire review into lower case
  review = review.lower()

  # Tokenizing the review by words
  review_words = review.split()

  # Removing the stop words
  review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

  # Stemming the words
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review_words]

  # Joining the stemmed words
  review = ' '.join(review)

  # Creating a corpus
  corpus.append(review)
  
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=15000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))


# Model Building

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=29)

# Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import MultinomialNB
#classifier = MultinomialNB(alpha=0.1)


# creating the model
rf = RandomForestClassifier(random_state=1234) 
rf.fit(X_train, y_train)

# predicting the test set results
y_pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Creating a pickle file for the Multinomial Naive Bayes model
#filename = 'restaurant-sentiment-rf-model.pkl'
pickle.dump(rf, open('restaurant-sentiment-rf-model.pkl', 'wb'))

#pickle.dump(rf, open(filename, 'wb'))



