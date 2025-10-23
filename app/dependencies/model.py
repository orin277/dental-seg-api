from app.neural_networks.dataset.tooth_region_extractor import ToothRegionExtractor
from fastapi import Depends

from app.services.model_service import ModelService
from app.core.config import settings
from app.validators.input_validator import InputValidator


def get_input_validator() -> InputValidator:
    return InputValidator()

def get_tooth_region_extractor() -> ToothRegionExtractor:
    return ToothRegionExtractor(settings.CARIES_IMG_SIZE, 10)

def get_model_service(
    validator: InputValidator = Depends(get_input_validator),
    tooth_extractor: ToothRegionExtractor = Depends(get_tooth_region_extractor)
) -> ModelService:
    return ModelService(
        settings.TOOTH_IMG_SIZE, settings.CARIES_IMG_SIZE, 
        validator, tooth_extractor, settings.DEVICE
    )