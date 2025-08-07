import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == '__main__':
    from app import app

    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    Path('cache').mkdir(exist_ok=True)

    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
