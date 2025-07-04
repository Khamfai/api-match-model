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
gunicorn --bind 0.0.0.0:3000 wsgi:app
```
- Run Gunicorn Backgroud Service
```bash
./start_server.sh &
```
- Stop Gunicorn Service
```bash
pkill -f gunicorn
```

* Using Docker
```bash
docker build -t name-matching-api .
docker run -p 3000:3000 name-matching-api
```

### 5. API Documentation

You can access the API documentation at `http://localhost:3000/docs`.
