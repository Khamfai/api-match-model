bind = "0.0.0.0:4000"
workers = 1  # Keep single worker to avoid multiprocessing issues
worker_class = "sync"
worker_connections = 1000
timeout = 300  # Increase timeout for model processing
keepalive = 2
max_requests = 100  # Reduce to restart workers more frequently
max_requests_jitter = 10
preload_app = True

# Add these lines to handle memory and multiprocessing issues
worker_tmp_dir = "/tmp"  # Use /tmp instead of /dev/shm on macOS
max_worker_memory = 2048  # Limit worker memory (MB)
worker_rlimit_nofile = 1024

accesslog = "-"
errorlog = "-"
loglevel = "info"

# Specify the WSGI module and variable
wsgi_module = "app:app"
