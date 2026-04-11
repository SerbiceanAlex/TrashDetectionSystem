"""
Tests for backend.config — verify settings load correctly.
"""

import os
import pytest


def test_settings_loads():
    from backend.config import settings
    assert len(settings.SECRET_KEY) >= 32
    assert settings.ALGORITHM == "HS256"
    assert settings.OTP_LENGTH == 6
    assert settings.MAX_UPLOAD_MB == 20


def test_settings_db_url():
    from backend.config import settings
    assert "sqlite" in settings.db_url


def test_settings_model_paths():
    from backend.config import settings
    assert settings.detector_path.name == "best.pt"
    assert settings.classifier_path.name == "best.pt"
    assert "detect" in str(settings.detector_path)
    assert "classify" in str(settings.classifier_path)


def test_settings_max_upload_bytes():
    from backend.config import settings
    assert settings.max_upload_bytes == 20 * 1024 * 1024
