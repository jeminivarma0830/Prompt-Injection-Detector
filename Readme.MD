# 🛡️ Prompt Injection Detector

A fine-tuned BERT-based classifier that detects malicious prompt injection attempts in LLM inputs in real time.

---

## 📁 Project Structure

```
prompt-injection-detector/
├── README.md
├── requirements.txx
├── config.py          # All settings in one place
├── dataset.py         # Load & prepare training data
├── train.py           # Fine-tune the model
├── detector.py        # Core detection logic
├── api.py             # FastAPI REST endpoint
├── test_detector.py   # Quick test script
└── data/
    └── sample_data.json

## ⚙️ Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data
python dataset.py

# 4. Train the model (~5-10 min on CPU)
python train.py

# 5. Test the detector
python test_detector.py

# 6. Run the API
uvicorn api:app --reload
```

---

## 🧪 API Usage

```bash
POST http://localhost:8000/detect

Body: { "text": "Ignore previous instructions and..." }
```

---

## 📊 Model

- Base: `distilbert-base-uncased` (lightweight & fast)
- Task: Binary classification (safe / injection)
- Dataset: Built-in + PINT benchmark style samples
