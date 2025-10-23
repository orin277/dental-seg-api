from app.neural_networks.dataset.tooth_region_extractor import ToothRegionExtractor
from app.validators.input_validator import InputValidator
from torchvision.transforms.functional import to_pil_image
import numpy as np

from app.utils.image_utils import convert_image_bool_to_uint, denormalize_image, get_mask_path, load_image_from_buffer
from app.neural_networks.dataset.segmentation_transform import SegmentationTransform
from app.neural_networks.predictor.model_predictor import ModelPredictor
from app.exceptions.model import FailedToDecodeImageException



class ModelService:
    def __init__(
        self, 
        tooth_img_size: int,
        caries_img_size: int,
        validator: InputValidator, 
        tooth_extractor: ToothRegionExtractor, 
        device: str = 'cuda'
    ):
        self.device = device
        self.tooth_img_size = tooth_img_size
        self.caries_img_size = caries_img_size
        self.validator = validator
        self.tooth_extractor = tooth_extractor

    async def predict_teeth(self, file, models):
        img = await self._load_image(file)
        img = self._transform_image(img, self.tooth_img_size)
        
        pred_mask = ModelPredictor.predict_sample_using_models(models, img, self.device)
        return self._save_mask(pred_mask)
    
    async def predict_caries(self, file, tooth_models, caries_models):
        img = await self._load_image(file)
        img = self._transform_image(img, self.tooth_img_size)

        tooth_mask = ModelPredictor.predict_sample_using_models(tooth_models, img, self.device)

        cropped_img, crop_params = self.tooth_extractor.extract_inference(
            img.squeeze(0).numpy(), 
            tooth_mask.cpu().numpy().astype(np.uint8)
        )
        cropped_img = denormalize_image(cropped_img)
        cropped_img = self._transform_image(cropped_img, self.caries_img_size)
        
        pred_mask = ModelPredictor.predict_cropped_sample_using_models(
            caries_models, cropped_img, 
            crop_params, self.tooth_extractor, self.device
        )

        return self._save_mask(pred_mask)
    
    async def _load_image(self, file):
        self.validator.check_file_existence(file)
        self.validator.check_image_format(file)

        contents = await file.read()
        self.validator.check_file_size(contents)

        img = load_image_from_buffer(contents)
        if img is None:
            raise FailedToDecodeImageException()
        return img
    
    def _transform_image(self, img, size):
        transform = SegmentationTransform(size)
        return transform.transform_image(img)
    
    def _save_mask(self, mask):
        mask = mask.cpu().numpy()
        mask = to_pil_image(convert_image_bool_to_uint(mask))
        path = get_mask_path()
        mask.save(path)
        return path