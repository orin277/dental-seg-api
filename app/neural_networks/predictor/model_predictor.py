import torch



class ModelPredictor:
    @staticmethod
    def predict_sample(model, image, device='cuda'):
        image = ModelPredictor._add_dims_for_image(image)

        model.eval()
        with torch.no_grad():
            image = image.to(device)
            image = model(image).squeeze(1)
            image = torch.sigmoid(image) > 0.5

        return image.squeeze(0)

    @staticmethod
    def predict_dataset(model, data, device='cuda'):
        predicted_masks = []

        model.eval()

        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(x).squeeze(1)
                pred = torch.sigmoid(pred) > 0.5
                predicted_masks.extend(pred)

        return torch.stack(predicted_masks).to(device)

    @staticmethod
    def predict_dataset_using_models(models, data, device='cuda'):
        predicted_masks = []

        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            predicted_masks.extend(ModelPredictor.predict_batch_for_models(models, x, device))

        return torch.stack(predicted_masks).to(device)

    @staticmethod
    def predict_batch_using_models(models, x, device='cuda'):
        preds = []

        with torch.no_grad():
            for model in models:
                model.eval()
                output = model(x.to(device))
                output = torch.sigmoid(output).squeeze(1)
                preds.append(output)

        batch_preds = torch.stack(preds, dim=0)
        return ModelPredictor._get_final_mask(batch_preds, len(models))

    @staticmethod
    def predict_sample_using_models(models, image, device='cuda'):
        image = ModelPredictor._add_dims_for_image(image)

        preds = []
        for i in range(len(models)):
            models[i].eval()
            with torch.no_grad():
                output = models[i](image.to(device)).squeeze(1).squeeze(0)
                output = torch.sigmoid(output) > 0.5
                preds.append(output)

        return ModelPredictor._get_final_mask(torch.stack(preds, dim=0), len(models))
    
    @staticmethod
    def predict_cropped_sample_using_models(models, image, crop_params, tooth_extractor, device='cuda'):
        image = ModelPredictor._add_dims_for_image(image)

        preds = []
        for i in range(len(models)):
            models[i].eval()
            with torch.no_grad():
                output = models[i](image.to(device)).squeeze(1)
                new_output = tooth_extractor.restore_full_mask(output.squeeze(0).cpu().numpy(), crop_params)
                output = torch.from_numpy(new_output).to(device)
                output = torch.sigmoid(output) > 0.5
                preds.append(output)

        return ModelPredictor._get_final_mask(torch.stack(preds, dim=0), len(models))

    @staticmethod
    def _get_final_mask(preds, count_models):
        preds = (preds > 0.5).int()
        votes = preds.sum(dim=0)
        threshold = count_models // 2 + (count_models % 2 > 0)
        return (votes >= threshold).int()


    @staticmethod
    def _add_dims_for_image(image):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        return image