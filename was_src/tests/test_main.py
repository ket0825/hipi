# test/test_main.py

from fastapi.testclient import TestClient
from main import app
from dotenv import load_dotenv

load_dotenv(".env")

client = TestClient(app)

def test_read_root():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}
    
# Run the tests with the command:
# pytest tests/test_main.py
# or pytest -v (by pytest.ini)
