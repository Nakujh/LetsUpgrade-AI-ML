import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('RestaurantReviewSentimentAnalyzer.pkl', 'rb'))
vectorizer = pickle.load(open("SentimentVectorizer.pkl", 'rb'))  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    input_data   = vectorizer.transform(int_features).toarray()
    input_pred   = model.predict(input_data)

    if input_pred[0]==1: 
        output = "Review is Positive" 
    else: 
        output = "Review is Negative"

    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)