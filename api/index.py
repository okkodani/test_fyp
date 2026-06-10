"""
Neural Network Based Text Mining for Mental Health Analysis
Backend API — Flask + BERT (HuggingFace Transformers) + Firebase Firestore
Author: Muhammad Danial (297801)
Dataset labels: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder
"""

import os, json, datetime, uuid, hashlib, re, numpy as np
from collections import defaultdict
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

import firebase_admin
from firebase_admin import credentials, firestore

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
FRONTEND_FOLDER = os.path.join(BASE_DIR, '..', 'frontend')

# Search multiple likely locations for the model
_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, '..', 'models', 'best_bert_mental_health'),
    os.path.join(BASE_DIR, 'models', 'best_bert_mental_health'),
    os.path.join(BASE_DIR, '..', 'best_bert_mental_health'),
    os.path.join(BASE_DIR, 'best_bert_mental_health'),
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if os.path.isdir(p)), _MODEL_CANDIDATES[0])

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='/')
CORS(app)
app.config["JWT_SECRET_KEY"]           = "mh-bert-secret-key-2024"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(hours=8)
jwt = JWTManager(app)

# ── Firebase ──────────────────────────────────────────────────────────────────
db = None
FIREBASE_READY = False
try:
    cred_path = os.path.join(BASE_DIR, 'firebase-credentials.json')
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    db = firestore.client()
    FIREBASE_READY = True
    print("✅ Firebase connected!")
except Exception as e:
    print(f"⚠️  Firebase Error: {e}")
    print("   → Place 'firebase-credentials.json' in the backend folder.")

# ── BERT Model ────────────────────────────────────────────────────────────────
tokenizer         = None
bert_model        = None
REAL_MODEL_LOADED = False

# These are the EXACT labels from Combined_Data.csv
DATASET_LABELS = {
    0: "Normal",
    1: "Depression",
    2: "Suicidal",
    3: "Anxiety",
    4: "Bipolar",
    5: "Stress",
    6: "Personality disorder"
}
ID2LABEL = DATASET_LABELS.copy()

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"⏳ Loading BERT model from: {MODEL_PATH}")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    bert_model.eval()
    # Override with model's own labels (from training)
    ID2LABEL = dict(bert_model.config.id2label)
    REAL_MODEL_LOADED = True
    print(f"✅ BERT Model loaded! Labels: {list(ID2LABEL.values())}")
except Exception as e:
    print(f"⚠️  BERT model not found at: {MODEL_PATH}")
    print(f"   Error: {e}")
    print("   → Using dataset-accurate keyword fallback until model is placed correctly.")


# ── Dataset-Accurate Keyword Fallback ─────────────────────────────────────────
# Built from analysis of Combined_Data.csv patterns
KEYWORD_RULES = [
    # (pattern, label, weight)  — evaluated in order, highest total wins
    # ── Suicidal (highest priority — safety first) ───────────────────────────
    (r"kill myself|want to die|end my life|suicid|no reason to live|better off dead|kill|"
     r"take my life|don't want to be alive|wish i was dead|self.harm|hurt myself", "Suicidal", 3.0),

    # ── Anxiety ───────────────────────────────────────────────────────────────
    (r"anxious|anxiety|panic attack|heart racing|can'?t breathe|nervous|"
     r"trouble sleeping|sleep(ing)? problem|restless|out of tune|confused mind|"
     r"constantly worry|overthink|health anxiety|racing heart|sweating|trembl|"
     r"scared|terrified|dread|phobia|on edge|can'?t relax|hyperventilat|"
     r"stomach ache|nausea from stress|chest tight|palpitation|startl", "Anxiety", 2.0),

    # ── Depression ────────────────────────────────────────────────────────────
    (r"hopeless|worthless|empty inside|nothing matters|can'?t get out of bed|"
     r"no energy|lost interest|depressed|miserable|dark thoughts|lonely|"
     r"isolated|don'?t care anymore|numb|feel nothing|crying all the time|"
     r"exhausted all the time|no motivation|pointless|feel like a burden|"
     r"can'?t enjoy|anhedonia|low mood|feeling down", "Depression", 2.0),

    # ── Bipolar ───────────────────────────────────────────────────────────────
    (r"bipolar|manic episode|mania|hypomania|mood swing|extreme high|extreme low|"
     r"euphoric|grandiose|racing thoughts|impulsive|reckless spending|"
     r"diagnosed with bipolar|manic and depressive|episode", "Bipolar", 2.0),

    # ── Stress ───────────────────────────────────────────────────────────────
    (r"overwhelmed|burned out|burnout|too much pressure|can'?t cope|"
     r"deadlines|workload|stressed out|tension headache|irritable|"
     r"frustrated with work|exam stress|financial stress|overworked|"
     r"deal with stress|breaking point", "Stress", 2.0),

    # ── Personality Disorder ──────────────────────────────────────────────────
    (r"personality disorder|borderline|bpd|avpd|npd|splitting|"
     r"fear of abandonment|unstable relationships|identity disturbance|"
     r"chronic emptiness|self.image|impulsiv", "Personality disorder", 2.0),

    # ── Normal / Healthy ─────────────────────────────────────────────────────
    (r"feel great|doing well|feeling good|happy and grateful|excited|"
     r"content|peaceful|motivated|thriving|positive outlook|"
     r"healthy routine|enjoying life|feeling positive", "Normal", 1.5),
]

def keyword_fallback(text: str) -> dict:
    """Dataset-accurate keyword classifier as fallback when BERT model is unavailable."""
    t       = text.lower()
    scores  = defaultdict(float)
    labels  = list(DATASET_LABELS.values())

    for pattern, label, weight in KEYWORD_RULES:
        matches = len(re.findall(pattern, t))
        if matches:
            scores[label] += weight * matches

    # Base noise so all categories have some probability
    for lbl in labels:
        scores[lbl] += 0.05

    total = sum(scores.values())
    probs = {lbl: round((scores[lbl] / total) * 100, 2) for lbl in labels}

    predicted  = max(probs, key=probs.get)
    confidence = probs[predicted]

    # Risk level
    if predicted in ("Suicidal",):
        risk = "Critical"
    elif predicted == "Normal":
        risk = "Low"
    elif confidence >= 60:
        risk = "High"
    else:
        risk = "Moderate"

    # Escalate if explicit suicidal language even if not top label
    if re.search(r"kill myself|want to die|end my life|suicid|better off dead", t):
        predicted, risk = "Suicidal", "Critical"
        confidence = max(confidence, 85.0)

    return {
        "predicted_label": predicted,
        "confidence":      confidence,
        "probabilities":   probs,
        "severity_score":  min(100, int(confidence * 1.1)) if predicted != "Normal" else max(0, 30 - int(confidence / 5)),
        "risk_level":      risk,
        "is_real_ai":      False,
    }


def score_text(text: str) -> dict:
    if REAL_MODEL_LOADED:
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        probs        = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
        if not isinstance(probs, list):
            probs = [probs]
        predicted_id    = int(np.argmax(probs))
        predicted_label = ID2LABEL[predicted_id]
        confidence      = probs[predicted_id]

        ll = predicted_label.lower()
        if "suicid" in ll:
            risk = "Critical"
        elif "normal" in ll:
            risk = "Low"
        elif confidence >= 0.70:
            risk = "High"
        else:
            risk = "Moderate"

        return {
            "predicted_label": predicted_label,
            "confidence":      round(confidence * 100, 2),
            "probabilities":   {ID2LABEL[i]: round(p * 100, 2) for i, p in enumerate(probs)},
            "severity_score":  int(confidence * 100) if "normal" not in ll else max(0, 30 - int(confidence * 30)),
            "risk_level":      risk,
            "is_real_ai":      True,
        }
    else:
        return keyword_fallback(text)


def get_recommendation(label: str, risk: str) -> dict:
    ll = label.lower()
    if "suicid" in ll:
        rec = {
            "immediate":    "⚠️ Please reach out to someone RIGHT NOW. You are not alone.",
            "professional": "Contact a crisis counsellor or go to the nearest emergency department.",
            "coping":       ["Call Befrienders KL: 03-7627 2929 (24hr)", "Tell a trusted person how you feel", "Remove access to means of self-harm", "Stay in a safe, non-isolated place"],
            "resources":    ["Befrienders KL: 03-7627 2929 (24hr, free)", "Talian Kasih: 15999", "MMHA: 03-2780 6803", "Hospital Emergency (999)"],
        }
    elif "depress" in ll:
        rec = {
            "immediate":    "Reach out to a trusted friend or family member today.",
            "professional": "Cognitive Behavioural Therapy (CBT) with a licensed psychologist is highly effective.",
            "coping":       ["Maintain a daily routine", "Light physical activity (even a 10-min walk)", "Journaling your thoughts", "Limit alcohol and social media"],
            "resources":    ["MMHA: 03-2780 6803", "Befrienders KL: 03-7627 2929", "Talian Kasih: 15999"],
        }
    elif "anx" in ll:
        rec = {
            "immediate":    "Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Repeat 4 times.",
            "professional": "Exposure therapy and CBT are proven treatments for anxiety.",
            "coping":       ["Box breathing (4-4-4-4)", "Grounding: name 5 things you can see", "Limit caffeine", "Regular sleep schedule", "Light exercise"],
            "resources":    ["MMHA: 03-2780 6803", "Talian Kasih: 15999", "Campus / university counsellor"],
        }
    elif "stress" in ll:
        rec = {
            "immediate":    "Step away from the stressor for 5 minutes. Take slow, deep breaths.",
            "professional": "Stress management counselling or mindfulness-based therapy.",
            "coping":       ["Eisenhower Matrix for task priority", "Pomodoro technique for study/work", "Regular short breaks", "Physical exercise", "Talk to a trusted person"],
            "resources":    ["MMHA: 03-2780 6803", "University counsellor", "Headspace / Calm app"],
        }
    elif "bipolar" in ll:
        rec = {
            "immediate":    "Contact your psychiatrist if you feel an episode coming on.",
            "professional": "Medication management with a psychiatrist is typically essential for Bipolar Disorder.",
            "coping":       ["Mood tracking app (e.g. Daylio)", "Maintain regular sleep/wake times", "Avoid alcohol", "Build a crisis plan with your doctor"],
            "resources":    ["MMHA: 03-2780 6803", "Hospital psychiatry department", "NAMI: nami.org"],
        }
    elif "personality" in ll:
        rec = {
            "immediate":    "Practice the TIPP skill: Temperature, Intense exercise, Paced breathing, Progressive relaxation.",
            "professional": "Dialectical Behaviour Therapy (DBT) is the gold-standard treatment.",
            "coping":       ["DBT distress tolerance skills", "Emotion regulation diary", "Mindfulness practice", "Consistent therapy attendance"],
            "resources":    ["MMHA: 03-2780 6803", "DBT Malaysia resources", "Befrienders KL: 03-7627 2929"],
        }
    else:  # Normal
        rec = {
            "immediate":    "You appear to be in a healthy mental state. Keep it up!",
            "professional": "Periodic mental wellness check-ins are great for prevention.",
            "coping":       ["Regular exercise", "Balanced diet", "Quality sleep (7-9 hrs)", "Mindfulness practice", "Maintain social connections"],
            "resources":    ["Headspace / Calm app", "Campus wellbeing programme"],
        }

    if risk == "Critical" and "suicid" not in ll:
        rec["urgent"] = "⚠️ URGENT: Please seek immediate professional help — call MMHA 03-2780 6803 or 999."
    return rec


# ── Helpers ───────────────────────────────────────────────────────────────────
def require_firebase():
    if not FIREBASE_READY:
        return jsonify({"error": "Database not connected. Check firebase-credentials.json."}), 500
    return None

def identity():
    return json.loads(get_jwt_identity())

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'MindBERT_Mental_Health_System.html')

@app.route('/api/status')
def status():
    return jsonify({
        "firebase_connected": FIREBASE_READY,
        "bert_model_loaded":  REAL_MODEL_LOADED,
        "model_path_checked": MODEL_PATH,
        "model_labels":       list(ID2LABEL.values()),
        "fallback_active":    not REAL_MODEL_LOADED,
    })

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def login():
    err = require_firebase()
    if err: return err
    data     = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = hashlib.sha256(data.get("password", "").encode()).hexdigest()
    doc = db.collection("users").document(username).get()
    if not doc.exists or doc.to_dict().get("password") != password:
        return jsonify({"error": "Invalid credentials"}), 401
    user  = doc.to_dict()
    token = create_access_token(identity=json.dumps({
        "id": user["id"], "username": user["username"],
        "role": user["role"], "name": user["name"],
    }))
    return jsonify({"token": token, "user": {k: v for k, v in user.items() if k != "password"}})

@app.route("/api/auth/me", methods=["GET"])
@jwt_required()
def me():
    err = require_firebase()
    if err: return err
    doc = db.collection("users").document(identity()["username"]).get()
    if not doc.exists: return jsonify({"error": "User not found"}), 404
    return jsonify({k: v for k, v in doc.to_dict().items() if k != "password"})

# ── Analyze ───────────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
@jwt_required()
def analyze():
    err = require_firebase()
    if err: return err
    ident = identity()
    text  = request.get_json(force=True).get("text", "").strip()
    if len(text) < 5:
        return jsonify({"error": "Please provide at least 5 characters."}), 400

    result         = score_text(text)
    recommendation = get_recommendation(result["predicted_label"], result["risk_level"])

    rid    = str(uuid.uuid4())
    record = {
        "id": rid, "user_id": ident["id"], "username": ident["username"],
        "user_name": ident["name"],
        "text_snippet": text[:120] + ("..." if len(text) > 120 else ""),
        "text_length":  len(text),
        "timestamp":    datetime.datetime.utcnow().isoformat(),
        "result":       result, "recommendation": recommendation,
    }
    db.collection("history").document(rid).set(record)
    return jsonify({"analysis_id": rid, "timestamp": record["timestamp"],
                    "result": result, "recommendation": recommendation})

# ── History ───────────────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
@jwt_required()
def history():
    err = require_firebase()
    if err: return err
    ident = identity()
    docs  = db.collection("history").stream() if ident["role"] == "admin" \
            else db.collection("history").where("user_id", "==", ident["id"]).stream()
    records = sorted([d.to_dict() for d in docs],
                     key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify({"records": records, "total": len(records)})

@app.route("/api/history/<record_id>", methods=["GET"])
@jwt_required()
def get_record(record_id):
    err = require_firebase()
    if err: return err
    ident = identity()
    doc   = db.collection("history").document(record_id).get()
    if not doc.exists: return jsonify({"error": "Record not found"}), 404
    record = doc.to_dict()
    if ident["role"] == "admin" or record["user_id"] == ident["id"]:
        return jsonify(record)
    return jsonify({"error": "Forbidden"}), 403

# ── Admin ─────────────────────────────────────────────────────────────────────
@app.route("/api/admin/stats", methods=["GET"])
@jwt_required()
def admin_stats():
    err = require_firebase()
    if err: return err
    ident = identity()
    if ident["role"] != "admin": return jsonify({"error": "Admin access required"}), 403
    records = [d.to_dict() for d in db.collection("history").stream()]
    users_n = len(list(db.collection("users").where("role", "==", "user").stream()))
    cat_counts, risk_counts = defaultdict(int), defaultdict(int)
    for r in records:
        res = r.get("result", {})
        cat_counts[res.get("predicted_label", "Unknown")] += 1
        risk_counts[res.get("risk_level", "Unknown")] += 1
    criticals = [r for r in records if r.get("result", {}).get("risk_level") == "Critical"]
    return jsonify({
        "total_analyses": len(records), "total_users": users_n,
        "avg_confidence": round(float(np.mean([r["result"]["confidence"] for r in records])), 2) if records else 0,
        "avg_severity":   round(float(np.mean([r["result"].get("severity_score", 0) for r in records])), 2) if records else 0,
        "category_distribution": dict(cat_counts),
        "risk_distribution":     dict(risk_counts),
        "critical_cases":        criticals,
    })

@app.route("/api/admin/users", methods=["GET"])
@jwt_required()
def list_users():
    err = require_firebase()
    if err: return err
    if identity()["role"] != "admin": return jsonify({"error": "Admin access required"}), 403
    users = [{k: v for k, v in d.to_dict().items() if k != "password"}
             for d in db.collection("users").stream()]
    return jsonify({"users": users})

@app.route("/api/admin/users", methods=["POST"])
@jwt_required()
def create_user():
    err = require_firebase()
    if err: return err
    if identity()["role"] != "admin": return jsonify({"error": "Admin access required"}), 403
    data     = request.get_json(force=True)
    username = data.get("username", "").strip()
    if not username: return jsonify({"error": "Username required"}), 400
    if db.collection("users").document(username).get().exists:
        return jsonify({"error": "Username already exists"}), 400
    new_user = {
        "id": f"user-{str(uuid.uuid4())[:8]}", "username": username,
        "password": hashlib.sha256(data.get("password", "pass123").encode()).hexdigest(),
        "role": data.get("role", "user"), "name": data.get("name", username),
        "email": data.get("email", ""), "created_at": datetime.date.today().isoformat(),
    }
    db.collection("users").document(username).set(new_user)
    return jsonify({"message": "User created", "user": {k: v for k, v in new_user.items() if k != "password"}}), 201

@app.route("/api/admin/users/<username>", methods=["DELETE"])
@jwt_required()
def delete_user(username):
    err = require_firebase()
    if err: return err
    if identity()["role"] != "admin": return jsonify({"error": "Admin access required"}), 403
    doc = db.collection("users").document(username).get()
    if not doc.exists: return jsonify({"error": "User not found"}), 404
    if doc.to_dict().get("role") == "admin": return jsonify({"error": "Cannot delete admin"}), 403
    db.collection("users").document(username).delete()
    return jsonify({"message": f"User '{username}' deleted"})

# ── Seed Demo Data ────────────────────────────────────────────────────────────
def seed_firebase_data():
    if not FIREBASE_READY: return
    users_ref = db.collection("users")
    if not list(users_ref.limit(1).stream()):
        print("🌱 Seeding demo users...")
        for u in [
            {"id":"admin-001","username":"admin","password":hashlib.sha256(b"admin123").hexdigest(),"role":"admin","name":"Dr. Ahmad Rizal","email":"admin@mhbert.edu.my","created_at":"2024-01-01"},
            {"id":"user-001","username":"user1","password":hashlib.sha256(b"user123").hexdigest(),"role":"user","name":"Muhammad Danial","email":"danial@student.edu.my","created_at":"2024-03-01"},
            {"id":"user-002","username":"user2","password":hashlib.sha256(b"user123").hexdigest(),"role":"user","name":"Siti Aisyah","email":"aisyah@student.edu.my","created_at":"2024-03-15"},
        ]:
            users_ref.document(u["username"]).set(u)
        print("   ✓ Demo users created.")

    hist_ref = db.collection("history")
    if not list(hist_ref.limit(1).stream()):
        print("🌱 Seeding demo history...")
        demos = [
            ("user-001","user1","Muhammad Danial","I feel so hopeless lately. Nothing seems to matter and I feel completely empty inside.","2026-05-20T09:30:00"),
            ("user-002","user2","Siti Aisyah","trouble sleeping, confused mind, restless heart. All out of tune","2026-05-23T11:00:00"),
            ("user-001","user1","Muhammad Danial","I have been having panic attacks. My heart races and I cannot breathe when I go outside.","2026-05-25T14:00:00"),
            ("user-002","user2","Siti Aisyah","Today was a great day! I went for a walk and feel really grateful and content.","2026-05-28T20:00:00"),
        ]
        for uid, uname, uname2, text, ts in demos:
            result = score_text(text)
            rec    = get_recommendation(result["predicted_label"], result["risk_level"])
            rid    = str(uuid.uuid4())
            hist_ref.document(rid).set({
                "id":rid,"user_id":uid,"username":uname,"user_name":uname2,
                "text_snippet":text[:120],"text_length":len(text),
                "timestamp":ts,"result":result,"recommendation":rec,
            })
        print("   ✓ Demo history created.")

seed_firebase_data()

if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  🧠 MindScope Mental Health")
    print(f"  Firebase : {'✅ Connected' if FIREBASE_READY else '❌ Not connected'}")
    print(f"  BERT     : {'✅ Loaded — ' + str(list(ID2LABEL.values())) if REAL_MODEL_LOADED else '⚠️  Using keyword fallback'}")
    if not REAL_MODEL_LOADED:
        print(f"  Model path expected: {MODEL_PATH}")
        print("  → Train model in notebook then copy to that folder.")
    print("  URL      : http://localhost:5000")
    print("=" * 55)
    print()
    app.run(debug=True, port=5000)
