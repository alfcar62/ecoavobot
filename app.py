from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__, static_folder=".")
CORS(app)  # abilita richieste cross-origin se usi index.html separato

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
    sims = cosine_similarity(vec, X)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]

    if best_score < threshold:
        return None, best_score

    return tags[best_idx], best_score

def generate_response(intent_tag):
    if intent_tag in responses:
        return random.choice(responses[intent_tag])
    return "Non ho capito bene, puoi riformulare?"

# ✅ ROTTA PER SERVIRE IMMAGINI DALLA CARTELLA /img
@app.route('/img/<path:filename>')
def serve_images(filename):
    """Serve file statici dalla cartella img/"""
    try:
        return send_from_directory('img', filename)
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404

# ✅ ENDPOINT DI DEBUG PER VERIFICARE I FILE
@app.route("/debug-files")
def debug_files():
    """Endpoint per verificare che i file siano accessibili"""
    import os
    img_files = []
    if os.path.exists('img'):
        img_files = os.listdir('img')
    
    return jsonify({
        "img_folder_exists": os.path.exists('img'),
        "images_in_img_folder": img_files,
        "logo_exists": "logoavogreen.png" in img_files,
        "current_directory": os.getcwd()
    })

# ✅ route per servire la home page
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ✅ route di test
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Avogreen-Bot è attivo!"})

# ✅ route della chat
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
    port = int(os.environ.get("PORT", 5000))  # Render assegna una porta
    app.run(host="0.0.0.0", port=port)