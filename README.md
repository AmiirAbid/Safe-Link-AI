# IDS Flask API

Simple production-ready Flask API to serve the intrusion detection model pipeline.

## Project structure
flask_app/
│── app.py
│── model_loader.py
│── ids_pipeline.pkl ← (add this file yourself)
│── model_metadata.json ← (optional)
│── requirements.txt
│── .gitignore
│── README.md

## Python / dependencies
This project is designed to run with **Python 3.10** and the exact dependency versions below to match training/export environment:

numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
Flask==3.0.3
joblib==1.4.2
gunicorn==23.0.0

Install dependencies:

```bash
pip install -r requirements.txt
```