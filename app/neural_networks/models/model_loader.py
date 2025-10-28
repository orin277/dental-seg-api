from app.neural_networks.models.dense_unet import DenseUnetModel
from app.neural_networks.models.residual_unet import ResidualUnetModel
from app.neural_networks.models.unet import UnetModel
from app.neural_networks.models.unet_3plus import Unet3PlusModel
import torch


from app.neural_networks.models.attention_unet import AttentionUnetModel


class ModelLoader:
    @staticmethod
    def load_unet_model(path, device):
        model = UnetModel()
        return ModelLoader._load_model(model, path, device)
    
    @staticmethod
    def load_attention_unet_model(path, device):
        model = AttentionUnetModel()
        return ModelLoader._load_model(model, path, device)
    
    @staticmethod
    def load_res_unet_model(path, device):
        model = ResidualUnetModel()
        return ModelLoader._load_model(model, path, device)
    
    @staticmethod
    def load_unet_3plus_model(path, device):
        model = Unet3PlusModel()
        return ModelLoader._load_model(model, path, device)
    
    @staticmethod
    def load_dense_unet_model(path, device):
        model = DenseUnetModel()
        return ModelLoader._load_model(model, path, device)
    
    @staticmethod
    def load_tooth_ensemble(model_paths, device):
        model1 = ModelLoader.load_dense_unet_model(model_paths[0], device)
        model2 = ModelLoader.load_unet_3plus_model(model_paths[1], device)
        model3 = ModelLoader.load_attention_unet_model(model_paths[2], device)
        return [model1, model2, model3]
    
    @staticmethod
    def load_caries_ensemble(model_paths, device):
        model1 = ModelLoader.load_dense_unet_model(model_paths[0], device)
        model2 = ModelLoader.load_unet_3plus_model(model_paths[1], device)
        model3 = ModelLoader.load_attention_unet_model(model_paths[2], device)
        return [model1, model2, model3]
    
    @staticmethod
    def _load_model(model, path, device):
        checkpoint = torch.load(path, weights_only=True, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model.to(device)