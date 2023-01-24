from flask import Flask, render_template, request, jsonify

from chat import get_response
from model_training import train_model

app = Flask(__name__)

@app.get('/')
def index_get():
    # Beim Neustarten der Web-Applikation wird das Model neu trainieren
    train_model()
    return render_template('base.html')
    

@app.post('/get_answer')
def get_answer():
    text = request.get_json().get('message')
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True, port=5000)