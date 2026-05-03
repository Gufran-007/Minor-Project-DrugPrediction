from flask import Flask, request, jsonify, render_template_string, make_response
import torch
import numpy as np
import pickle
from model import DrugNet

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DrugNet(input_dim=5000)
model.load_state_dict(torch.load(
    'drug_model.pth',
    map_location=device,
    weights_only=True
))
model.to(device)
model.eval()

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print(f"✅ Model loaded on {device}")

HTML = open('templates/index.html').read()

@app.route('/')
def index():
    resp = make_response(render_template_string(HTML))
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    return resp

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    review = data.get('review', '').strip()

    if not review:
        return jsonify({'error': 'No review provided'}), 400

    features = vectorizer.transform([review])
    tensor   = torch.FloatTensor(features.toarray()).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        pred    = torch.argmax(probs, dim=1).item()
        conf    = probs[0][pred].item() * 100

    return jsonify({
        'prediction': 'Good Drug' if pred == 1 else 'Bad Drug',
        'label':      pred,
        'confidence': round(conf, 1),
        'bad_score':  round(probs[0][0].item() * 100, 1),
        'good_score': round(probs[0][1].item() * 100, 1),
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)