from src.app import app

# Gunicorn looks for 'application' by default
application = app

if __name__ == "__main__":
    app.run()
