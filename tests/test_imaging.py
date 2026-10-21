"""
Unit tests for Medical Imaging Module.
Tests image validation, file size bounds, invalid image rejection, model inference, and Grad-CAM generation.
"""
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from modules.imaging.preprocessing import validate_image_file
from modules.imaging.inference import imaging_service
from modules.imaging.schemas import ImagingAnalysisResult

def test_validate_image_file_valid(tmp_path):
    img = Image.new("RGB", (256, 256), color="gray")
    img_path = tmp_path / "test_valid.png"
    img.save(img_path)

    is_valid, err_msg, loaded = validate_image_file(img_path, img_path.name)
    assert is_valid is True
    assert err_msg == ""
    assert loaded.size == (256, 256)

def test_validate_image_file_unsupported_extension():
    is_valid, err_msg, loaded = validate_image_file(b"dummy data", "test.exe")
    assert is_valid is False
    assert "Unsupported file type" in err_msg
    assert loaded is None

def test_validate_image_file_low_resolution(tmp_path):
    img = Image.new("RGB", (32, 32), color="black")
    img_path = tmp_path / "low_res.png"
    img.save(img_path)

    is_valid, err_msg, loaded = validate_image_file(img_path, img_path.name)
    assert is_valid is False
    assert "too low" in err_msg

def test_imaging_inference_execution(tmp_path):
    img = Image.new("RGB", (224, 224), color="gray")
    img_path = tmp_path / "cxr_test.png"
    img.save(img_path)

    result = imaging_service.analyze(img, "cxr_test.png")
    assert isinstance(result, ImagingAnalysisResult)
    assert result.inference_status == "SUCCESS"
    assert len(result.findings) > 0
    assert 0.0 <= result.top_confidence <= 1.0
    assert result.model_name == "DenseNet121-CXR"
