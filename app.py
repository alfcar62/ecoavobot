from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # permette richieste da domini diversi

# Carica intents.json
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# Prepara dataset (patterns e tag)
patterns = []
tags = []
responses = {}
for intent in intents:
    tag = intent["tag"]
    responses[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(tag)

# TF-IDF per rappresentare i patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

def classify_intent(user_message, threshold=0.3):
    """Ritorna (intent, confidence) per un messaggio utente"""
    vec = vectorizer.transform([user_message])
    sims = cosine_similarity(vec, X)[0]  # similarità con tutti i patterns
    best_idx = sims.argmax()
    best_score = sims[best_idx]

    if best_score < threshold:
        return None, best_score

    return tags[best_idx], best_score

def generate_response(intent_tag):
    if intent_tag in responses:
        return random.choice(responses[intent_tag])
    return "Non ho capito bene, puoi riformulare?"

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "EcoAvoBot è attivo!"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"answer": "Per favore scrivi qualcosa."})

    intent, confidence = classify_intent(user_message)

    if intent is None:
        return jsonify({"answer": "Non ho capito bene, puoi riformulare?"})

    answer = generate_response(intent)
    return jsonify({
        "intent": intent,
        "confidence": round(float(confidence), 2),
        "answer": answer
    })

if __name__ == "__main__":
    app.run(debug=True)
