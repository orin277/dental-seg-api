from fastapi.testclient import TestClient
import numpy as np
import io
import cv2
import pytest

from app.main import app
from app.core.config import settings
from app.neural_networks.models.model_loader import ModelLoader



@pytest.fixture(scope="session")
def client():
    app.state.model = app.state.tooth_seg_models = ModelLoader.load_tooth_ensemble(
        settings.all_tooth_seg_model_paths,
        settings.DEVICE
    )
    app.state.caries_seg_models = ModelLoader.load_caries_ensemble(
        settings.all_caries_seg_model_paths,
        settings.DEVICE
    )
    with TestClient(app) as c:
        yield c


def test_predict_tooth_seg_with_correct_file(client):
    img = np.ones((1024, 1024), dtype=np.uint8)
    is_success, buffer = cv2.imencode(".jpg", img)
    io_buf = io.BytesIO(buffer)

    files = {"file": ("test.png", io_buf, "image/png")}

    response = client.post("/predict/tooth-segmentation", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "mask_path" in data


def test_predict_tooth_seg_with_incorrect_file(client):
    io_buf = io.BytesIO("print('test')".encode("utf-8"))
    files = {"file": ("test.py", io_buf, "text/plain")}

    response = client.post("/predict/tooth-segmentation", files=files)
    assert response.status_code == 415


def test_predict_tooth_seg_with_big_file(client):
    io_buf = io.BytesIO(b"\0" * (20*1024*1024))
    files = {"file": ("test.png", io_buf, "image/png")}

    response = client.post("/predict/tooth-segmentation", files=files)
    assert response.status_code == 413


def test_predict_tooth_seg_with_not_transferred_file(client):
    response = client.post("/predict/tooth-segmentation", files={"file": None})
    assert response.status_code == 400



def test_predict_caries_seg_with_correct_file(client):
    img = np.ones((1024, 1024), dtype=np.uint8)
    is_success, buffer = cv2.imencode(".jpg", img)
    io_buf = io.BytesIO(buffer)

    files = {"file": ("test.png", io_buf, "image/png")}

    response = client.post("/predict/caries-segmentation", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "mask_path" in data


def test_predict_caries_seg_with_incorrect_file(client):
    io_buf = io.BytesIO("print('test')".encode("utf-8"))
    files = {"file": ("test.py", io_buf, "text/plain")}

    response = client.post("/predict/caries-segmentation", files=files)
    assert response.status_code == 415


def test_predict_caries_seg_with_big_file(client):
    io_buf = io.BytesIO(b"\0" * (20*1024*1024))
    files = {"file": ("test.png", io_buf, "image/png")}

    response = client.post("/predict/tooth-segmentation", files=files)
    assert response.status_code == 413


def test_predict_caries_seg_with_not_transferred_file(client):
    response = client.post("/predict/caries-segmentation", files={"file": None})
    assert response.status_code == 400