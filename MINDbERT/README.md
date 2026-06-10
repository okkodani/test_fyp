# 🧠 MindBERT — Neural Network Based Text Mining for Mental Health Analysis
**Author:** Muhammad Danial (297801)  
**Subject:** STIZK3993 — Data Science Project  
**Institution:** UUM (Universiti Utara Malaysia)

---

## 📌 Project Overview

This project implements a **BERT-based deep learning system** to analyze text for mental health indicators. The system classifies text into 6 categories using a fine-tuned `bert-base-uncased` transformer model.

### Mental Health Categories
| Label | Code | Description |
|-------|------|-------------|
| Depression | DEP | Persistent sadness, hopelessness, loss of interest |
| Anxiety | ANX | Excessive worry, panic attacks, nervousness |
| Stress | STR | Burnout, overwhelm, work/life pressure |
| PTSD | PTSD | Trauma-related flashbacks, nightmares |
| Bipolar Disorder | BPD | Extreme mood swings, manic episodes |
| Normal / Healthy | NRM | Positive, healthy mental state |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MindBERT System                       │
├────────────────────────┬────────────────────────────────┤
│    Frontend (React)    │      Backend (Flask API)       │
│  ┌──────────────────┐  │  ┌──────────────────────────┐ │
│  │ Admin Dashboard  │◄─┼──┤ JWT Authentication       │ │
│  │ User Portal      │  │  │ BERT Inference Engine    │ │
│  │ Analysis Results │  │  │ Analysis History API     │ │
│  │ History Viewer   │  │  │ User Management API      │ │
│  └──────────────────┘  │  └──────────────────────────┘ │
└────────────────────────┴────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  BERT Model Layer  │
                    │ bert-base-uncased  │
                    │  + Linear Head    │
                    │  (6 classes)      │
                    └───────────────────┘
```

---

## 👤 User Roles

### Admin Role
- View system-wide analytics dashboard
- See all users' analysis records
- Manage users (add, delete)
- Monitor critical/high-risk cases
- View model performance statistics

### User Role
- Submit text for mental health analysis
- View personal analysis history
- Receive detailed recommendations and coping strategies
- Access mental health resources

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- CUDA GPU (recommended for BERT training)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
# API runs on http://localhost:5000
```

### Training the BERT Model (Optional)
```bash
cd notebooks
jupyter notebook bert_training.ipynb
```

### Demo Credentials
| Role  | Username | Password  |
|-------|----------|-----------|
| Admin | admin    | admin123  |
| User  | user1    | user123   |
| User  | user2    | user123   |

---

## 🧠 BERT Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | bert-base-uncased |
| Total Parameters | 110M |
| Hidden Size | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Max Token Length | 256 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 4 |
| Test Accuracy | ~88.4% |
| F1 Score | ~0.882 (weighted) |

---

## 📁 Project Structure

```
mental_health_bert/
├── backend/
│   ├── app.py              # Flask REST API
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── MindBERT_System.html # React SPA (standalone)
├── notebooks/
│   └── bert_training.ipynb # BERT training notebook
├── models/
│   └── best_bert_mental_health/ # Saved model weights
├── data/
│   ├── class_distribution.png
│   ├── training_curves.png
│   └── confusion_matrix.png
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | /api/auth/login | No | Login, get JWT token |
| GET | /api/auth/me | JWT | Get current user info |
| POST | /api/analyze | JWT | Analyze text with BERT |
| GET | /api/history | JWT | Get analysis history |
| GET | /api/history/:id | JWT | Get specific record |
| GET | /api/admin/stats | Admin | Get system statistics |
| GET | /api/admin/users | Admin | List all users |
| POST | /api/admin/users | Admin | Create new user |
| DELETE | /api/admin/users/:id | Admin | Delete user |

---

## ⚠️ Disclaimer

This system is an **educational research tool** for STIZK3993. It is **NOT** a substitute for professional mental health diagnosis or treatment. Always consult a qualified mental health professional.

**Emergency Resources (Malaysia):**
- 🆘 Emergency: **999**
- 💬 MMHA Helpline: **03-2780 6803**
- 📞 Befrienders KL: **03-7627 2929**
- 📱 Talian Kasih: **15999**
