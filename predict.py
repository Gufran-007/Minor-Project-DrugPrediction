import torch
import numpy as np
import pickle
from model import DrugNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DrugNet(input_dim=5000)
model.load_state_dict(torch.load(
    'drug_model.pth',
    map_location=device,
    weights_only=True
))
model.to(device)
model.eval()

try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Vectorizer loaded.\n")
except FileNotFoundError:
    print("❌ ERROR: vectorizer.pkl not found.")
    print("   You need to save your TF-IDF vectorizer during preprocessing.")
    print("   Add this to your preprocess.py after fitting the vectorizer:")
    print()
    print("       import pickle")
    print("       with open('vectorizer.pkl', 'wb') as f:")
    print("           pickle.dump(vectorizer, f)")
    print()
    exit()

def predict_drug_review(review_text: str) -> dict:
    """
    Takes a raw drug review string.
    Returns a dict with prediction, confidence, and scores.
    """

    features = vectorizer.transform([review_text])
    features_tensor = torch.FloatTensor(features.toarray()).to(device)

    with torch.no_grad():
        outputs = model(features_tensor)
        probs   = torch.softmax(outputs, dim=1)
        pred    = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    return {
        'prediction':  'Good Drug ✅' if pred == 1 else 'Bad Drug ❌',
        'label':       pred,
        'confidence':  confidence,
        'bad_score':   probs[0][0].item() * 100,
        'good_score':  probs[0][1].item() * 100,
    }

def print_result(review: str, result: dict):
    print("─" * 55)
    display = review if len(review) <= 80 else review[:77] + "..."
    print(f"  Review     : \"{display}\"")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.1f}%")
    print(f"  Bad  score : {result['bad_score']:.1f}%")
    print(f"  Good score : {result['good_score']:.1f}%")
    print("─" * 55)

print("=" * 55)
print("         DrugNet — Prediction Demo")
print("=" * 55)

sample_reviews = [
    "This medication completely changed my life. The pain is gone "
    "and I feel like myself again. Highly recommend it.",

    "Terrible side effects. I felt nauseous every single day and "
    "had to stop taking it after one week. Did nothing for my condition.",

    "It works okay I guess. Some days are better than others. "
    "Not sure if it's really helping.",

    "After years of suffering, this drug finally gave me relief. "
    "No side effects at all. My doctor is very pleased with my progress.",

    "Made my anxiety so much worse. Heart palpitations, insomnia, "
    "and constant headaches. Absolute nightmare.",
]

print("\n📋 Running predictions on sample reviews...\n")
for review in sample_reviews:
    result = predict_drug_review(review)
    print_result(review, result)

print("\n" + "=" * 55)
print("  INTERACTIVE MODE — Type your own review")
print("  (type 'quit' to exit)")
print("=" * 55 + "\n")

while True:
    user_input = input("Enter a drug review: ").strip()
    if user_input.lower() in ('quit', 'exit', 'q'):
        print("Exiting. Goodbye!")
        break
    if not user_input:
        print("Please enter some text.\n")
        continue
    result = predict_drug_review(user_input)
    print_result(user_input, result)
    print()