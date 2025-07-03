# English-Thai Name Matching API

REST API for matching English and Thai names using AI.

## Installation

### 1. Activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt

```

### 3. Run the API:
```bash
python main.py
```

### 4. Deploy the API:

- Using Gunicorn
```bash
gunicorn --bind 0.0.0.0:4000 wsgi:app
```

- Using Docker
```bash
docker build -t name-matching-api .
docker run -p 4000:4000 name-matching-api
```

### 5. API Documentation

You can access the API documentation at `http://localhost:4000/docs`.
