bind = "0.0.0.0:4000"
workers = 1  # Use only 1 worker to avoid memory issues
worker_class = "sync"  # Use sync worker instead of async
worker_connections = 1000
timeout = 120  # Increase timeout for model processing
keepalive = 2
max_requests = 1000  # Restart workers after 100 requests to prevent memory leaks
max_requests_jitter = 100
preload_app = True  # Preload the app to share model across workers
# Remove or comment out the worker_tmp_dir line for macOS
# worker_tmp_dir = "/dev/shm"  # This doesn't exist on macOS
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Specify the WSGI module and variable
wsgi_module = "app:app"


# bind = "0.0.0.0:4000"
# workers = 2
# worker_class = "sync"
# worker_connections = 1000
# timeout = 30
# keepalive = 2
# max_requests = 1000
# max_requests_jitter = 100
# preload_app = True
# accesslog = "-"
# errorlog = "-"
# loglevel = "info"

# # Specify the WSGI module and variable
# wsgi_module = "app:app"
