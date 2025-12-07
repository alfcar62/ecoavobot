from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import json
import random
import os

app = Flask(__name__)
CORS(app)

# Preprocessamento semplice
def preprocess_text(text):
    if not text:
        return ""
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
    except json.JSONDecodeError as e:
        print(f"❌ ERRORE nel file JSON: {e}")
        return []

intents = load_intents()

# Matching semplice basato su parole chiave
def find_best_response(user_message):
    if not intents:
        return "Errore: nessun intent caricato", "errore", 0.0
    
    processed_msg = preprocess_text(user_message)
    user_words = set(processed_msg.split())
    
    print(f"🔍 Analizzo: '{user_message}' -> '{processed_msg}'")
    
    # Prima cerca match esatti con i pattern
    for intent in intents:
        for pattern in intent["patterns"]:
            processed_pattern = preprocess_text(pattern)
            if processed_msg == processed_pattern:
                print(f"✅ Match esatto: '{processed_msg}' = '{processed_pattern}'")
                return random.choice(intent["responses"]), intent["tag"], 0.9
    
    # Poi cerca parole in comune
    best_score = 0.0
    best_response = "Non ho capito bene, puoi essere più specifico?"
    best_intent = "unknown"
    
    for intent in intents:
        for pattern in intent["patterns"]:
            processed_pattern = preprocess_text(pattern)
            pattern_words = set(processed_pattern.split())
            
            # Calcola similarità semplice
            common_words = user_words.intersection(pattern_words)
            if common_words:
                score = len(common_words) / max(len(user_words), 1)
                print(f"📊 Intent: {intent['tag']}, Score: {score:.2f}, Common: {common_words}")
                
                if score > best_score:
                    best_score = score
                    best_intent = intent["tag"]
                    best_response = random.choice(intent["responses"])
    
    print(f"🎯 Miglior match: {best_intent} (score: {best_score:.2f})")
    return best_response, best_intent, best_score

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data_req = request.get_json()
        if not data_req:
            return jsonify({"answer": "Non ho ricevuto alcun messaggio."})
        
        user_message = data_req.get("message", "").strip()
        print(f"📩 Ricevuto: '{user_message}'")
        
        if not user_message:
            return jsonify({"answer": "Il messaggio è vuoto."})
        
        if len(user_message) < 2:
            return jsonify({"answer": "Il messaggio è troppo breve."})
        
        response, intent, confidence = find_best_response(user_message)
        
        if confidence < 0.1:
            # Fallback per saluti semplici
            user_lower = user_message.lower()
            if any(word in user_lower for word in ["ciao", "salve", "buongiorno", "hello", "hey"]):
                response = "Ciao! Sono EcoAvoBot, il tuo assistente ambientale! 🌍"
                intent = "saluto"
                confidence = 0.8
        
        return jsonify({
            "intent": intent,
            "confidence": round(float(confidence), 2),
            "answer": response
        })
        
    except Exception as e:
        print(f"❌ Errore in chat(): {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": "Si è verificato un errore. Riprova più tardi."})

# 🔥 SERVE IL FRONTEND
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route("/test")
def test():
    return jsonify({
        "status": "online",
        "intents_loaded": len(intents),
        "message": "✅ Server funzionante!",
        "patterns": sum(len(intent["patterns"]) for intent in intents)
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "EcoAvoBot"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    print(f"🚀 Avvio server su porta {port}")
    print(f"📊 Intents caricati: {len(intents)}")
    app.run(debug=debug_mode, port=port, host='0.0.0.0')