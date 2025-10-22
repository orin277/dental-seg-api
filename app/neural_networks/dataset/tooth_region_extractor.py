import cv2
import numpy as np


class ToothRegionExtractor:
    def __init__(self, target_size = 256, padding = 10):
        self.target_size = target_size
        self.padding = padding
    
    def _get_bbox(self, mask):
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None:
            return 0, 0, mask.shape[1], mask.shape[0]
        
        x, y, w, h = cv2.boundingRect(coords)
        
        x = max(0, x - self.padding)
        y = max(0, y - self.padding)
        w = min(mask.shape[1] - x, w + 2 * self.padding)
        h = min(mask.shape[0] - y, h + 2 * self.padding)
        
        return x, y, w, h
    
    def _adjust_bbox_to_target(self, x, y, w, h, img_h, img_w):
        if w < self.target_size and h < self.target_size:
            w_new = h_new = self.target_size
        elif w < self.target_size and h >= self.target_size:
            w_new = self.target_size
            h_new = h
        elif h < self.target_size and w >= self.target_size:
            w_new = w
            h_new = self.target_size
        else:
            w_new = w
            h_new = h
        
        x_center = x + w // 2
        y_center = y + h // 2
        
        x_new = max(0, min(img_w - w_new, x_center - w_new // 2))
        y_new = max(0, min(img_h - h_new, y_center - h_new // 2))
        
        return x_new, y_new, w_new, h_new
    
    def _crop_and_resize(self, img, x, y, w, h):
        cropped = img[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (self.target_size, self.target_size), 
                            interpolation=cv2.INTER_LINEAR)
        return resized
    
    def extract_inference(self, image, tooth_mask):
        x, y, w, h = self._get_bbox(tooth_mask)
        x, y, w, h = self._adjust_bbox_to_target(x, y, w, h, 
                                                  image.shape[0], image.shape[1])
        
        cropped_image = self._crop_and_resize(image, x, y, w, h)
        crop_params = {'x': x, 'y': y, 'w': w, 'h': h, 'orig_shape': image.shape[:2]}
        return cropped_image, crop_params
    
    def extract_training(self, image, tooth_mask, caries_mask):
        x, y, w, h = self._get_bbox(tooth_mask)
        x, y, w, h = self._adjust_bbox_to_target(x, y, w, h,
                                                  image.shape[0], image.shape[1])
        
        cropped_image = self._crop_and_resize(image, x, y, w, h)
        
        cropped_caries = caries_mask[y:y+h, x:x+w]
        cropped_caries = cv2.resize(cropped_caries, (self.target_size, self.target_size),
                                   interpolation=cv2.INTER_NEAREST)
        
        return cropped_image, cropped_caries
    
    def restore_full_mask(self, cropped_pred, crop_params):
        x, y, w, h = crop_params['x'], crop_params['y'], crop_params['w'], crop_params['h']
        orig_h, orig_w = crop_params['orig_shape']
        
        resized_pred = cv2.resize(cropped_pred, (w, h), 
                                 interpolation=cv2.INTER_LINEAR)
        
        full_mask = np.zeros((orig_h, orig_w), dtype=cropped_pred.dtype)
        full_mask[y:y+h, x:x+w] = resized_pred
        return full_mask