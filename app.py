from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import json
import random
import os
from math import sqrt
from collections import Counter

app = Flask(__name__)
CORS(app)

# === IMPLEMENTAZIONE MANUALE TF-IDF SEMPLIFICATA ===
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\']', ' ', text)
    return ' '.join(text.split())

def tokenize(text):
    return text.split()

def compute_tf(text):
    tokens = tokenize(text)
    total_words = len(tokens)
    tf_dict = {}
    for word in tokens:
        tf_dict[word] = tf_dict.get(word, 0) + 1 / total_words
    return tf_dict

def compute_idf(documents):
    n_docs = len(documents)
    idf_dict = {}
    for doc in documents:
        tokens = set(tokenize(doc))
        for token in tokens:
            idf_dict[token] = idf_dict.get(token, 0) + 1
    
    for token, count in idf_dict.items():
        idf_dict[token] = 1 + math.log(n_docs / (count + 1))
    
    return idf_dict

def compute_tfidf_vector(text, idf_dict):
    tf_dict = compute_tf(text)
    vector = {}
    for word, tf_val in tf_dict.items():
        vector[word] = tf_val * idf_dict.get(word, 0)
    return vector

def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in set(vec1) | set(vec2))
    norm1 = sqrt(sum(val ** 2 for val in vec1.values()))
    norm2 = sqrt(sum(val ** 2 for val in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

# === FINE IMPLEMENTAZIONE TF-IDF ===

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

# Prepara i dati
patterns_data = []
all_documents = []

for intent in intents:
    for pattern in intent["patterns"]:
        processed_pattern = preprocess_text(pattern)
        patterns_data.append({
            "text": processed_pattern,
            "intent": intent["tag"],
            "response": random.choice(intent["responses"])
        })
        all_documents.append(processed_pattern)

# Calcola IDF una volta sola
idf_dict = compute_idf(all_documents) if all_documents else {}

def find_best_match(user_message):
    if not patterns_data:
        return None, 0.0
    
    processed_msg = preprocess_text(user_message)
    user_vector = compute_tfidf_vector(processed_msg, idf_dict)
    
    best_similarity = 0.0
    best_match = None
    
    for pattern in patterns_data:
        pattern_vector = compute_tfidf_vector(pattern["text"], idf_dict)
        similarity = cosine_similarity(user_vector, pattern_vector)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = pattern
    
    return best_match, best_similarity

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
        
        best_match, similarity = find_best_match(user_message)
        
        print(f"🔍 Similarità: {similarity:.3f}")
        
        if similarity < 0.2:  # Soglia bassa per matching semplice
            # Fallback: cerca parole chiave
            user_text = user_message.lower()
            for intent in intents:
                for pattern in intent["patterns"]:
                    if any(word in user_text for word in pattern.lower().split()[:3]):
                        return jsonify({
                            "intent": intent["tag"],
                            "confidence": 0.5,
                            "answer": random.choice(intent["responses"])
                        })
            
            return jsonify({"answer": "Non ho capito bene, puoi essere più specifico?"})
        
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
        "message": "✅ Server funzionante senza scikit-learn!"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    print(f"🚀 Avvio server su porta {port}")
    app.run(debug=debug, port=port, host='0.0.0.0')