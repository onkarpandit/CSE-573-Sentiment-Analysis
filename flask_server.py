from flask import Flask, request
from main import linear_models_sentiment_analysis, model_training

app = Flask(__name__)

@app.route("/", methods=['POST'])
def sentiment_analysis():
    input_json = request.get_json(force=True)
    input_text = input_json['text']
    prediction = linear_models_sentiment_analysis(input_text)
    return prediction

