import pytest
from main import create_app
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv(".env")
client = TestClient(create_app())

@pytest.mark.asyncio
async def test_insert_image():
    res = client.post("/api/v1/img/", files={"file": ("test.jpg", open("test.jpg", "rb"))})
    assert res.status_code == 200
    assert res.json() == {"message": "jpg"}
    
    res = client.post("/api/v1/img/", files={"file": ("test.png", open("test.png", "rb"))})
    assert res.status_code == 200
    assert res.json() == {"message": "png"}    

@pytest.mark.asyncio
async def test_insert_bulk_images():
    res = client.post("/api/v1/img/bulk")
    assert res.status_code == 200
    assert res.json() == {"message": "Insert bulk images"}        

@pytest.mark.asyncio
async def test_query_image():
    res = client.post("/api/v1/img/query")
    assert res.status_code == 200
    assert res.json() == {"message": "Query image"}    