from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # Usiamo similarity invece di ML
from flask_cors import CORS
import re
import json
import random
import os

app = Flask(__name__)
CORS(app)

# Preprocessamento del testo
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\']', ' ', text)
    return ' '.join(text.split())

# Carica intents.json
def load_intents():
    try:
        with open("intents.json", "r", encoding="utf-8") as f:
            intents_data = json.load(f)
            intents = intents_data["intents"]
        print(f"✅ Caricati {len(intents)} intents")
        return intents
    except FileNotFoundError:
        print("❌ ERRORE: File intents.json non trovato!")
        return []

intents = load_intents()

# Prepara i dati per similarity matching
patterns_data = []
for intent in intents:
    for pattern in intent["patterns"]:
        patterns_data.append({
            "text": preprocess_text(pattern),
            "intent": intent["tag"],
            "response": random.choice(intent["responses"])
        })

# Crea vettori TF-IDF per similarity matching
if patterns_data:
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    pattern_texts = [item["text"] for item in patterns_data]
    pattern_vectors = vectorizer.fit_transform(pattern_texts)
    print("✅ Vettorizzatore preparato per similarity matching")
else:
    vectorizer = None
    pattern_vectors = None

def find_best_match(user_message):
    if vectorizer is None:
        return None, 0.0, "Errore: nessun dato di training"
    
    processed_msg = preprocess_text(user_message)
    user_vector = vectorizer.transform([processed_msg])
    
    best_similarity = 0.0
    best_match = None
    
    for i, pattern_vector in enumerate(pattern_vectors):
        similarity = cosine_similarity(user_vector, pattern_vector)[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = patterns_data[i]
    
    return best_match, best_similarity

def generate_response(intent_tag):
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "Mi dispiace, non ho capito. Puoi riformulare la domanda?"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data_req = request.get_json()
        if not data_req or "message" not in data_req:
            return jsonify({"answer": "Non ho ricevuto alcun messaggio."})
        
        user_message = data_req["message"].strip()
        print(f"📩 Ricevuto: '{user_message}'")
        
        if len(user_message) < 2:
            return jsonify({"answer": "Il messaggio è troppo breve."})
        
        # Usa similarity matching invece di ML classification
        best_match, similarity = find_best_match(user_message)
        
        print(f"🔍 Similarità: {similarity:.3f}")
        
        if similarity < 0.3:
            return jsonify({"answer": "Non ho capito bene, puoi essere più specifico riguardo a riciclo, energia, acqua o mobilità sostenibile?"})
        
        return jsonify({
            "intent": best_match["intent"],
            "confidence": round(float(similarity), 2),
            "answer": best_match["response"]
        })
        
    except Exception as e:
        print(f"❌ Errore in chat(): {e}")
        return jsonify({"answer": "Si è verificato un errore. Riprova più tardi."})

@app.route("/test")
def test():
    return jsonify({
        "status": "online",
        "patterns_loaded": len(patterns_data),
        "message": "✅ Server funzionante"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    print(f"🚀 Avvio server su porta {port}")
    app.run(debug=debug, port=port, host='0.0.0.0')