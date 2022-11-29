from flask import Flask, request, json
from main import linear_models_sentiment_analysis, model_training
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=['POST'])
def sentiment_analysis():
    input_json = request.get_json(force=True)
    input_text = input_json['text']
    print('received', input_text)
    prediction = linear_models_sentiment_analysis(input_text)
    print('returning ', prediction)
    return json.dumps(prediction)

