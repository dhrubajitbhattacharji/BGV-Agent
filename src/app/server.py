import os
import uvicorn
from dotenv import load_dotenv
from app.api import get_app


def serve():
    load_dotenv()
    app = get_app()
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8003"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
