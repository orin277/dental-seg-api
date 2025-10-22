import os
import cv2
import torch.utils.data as data


class DentalDataset(data.Dataset):
    def __init__(self, path, transform=None, is_rgb=False, is_train=True):
        self.path = path
        self.transform = transform
        self.is_train = is_train
        self.image_files = []
        self.mask_files = []
        self.is_rgb = is_rgb

        image_path = os.path.join(self.path, "images")
        mask_path = os.path.join(self.path, "masks")

        image_file_names = os.listdir(image_path)
        mask_file_names = os.listdir(mask_path)
        mask_fn_dict = {}
        for fn in mask_file_names:
            curr_file_name = os.path.splitext(fn)
            mask_fn_dict[curr_file_name[0]] = curr_file_name[1]

        for fn in image_file_names:
            curr_file_name = os.path.splitext(fn)
            if curr_file_name[0] in mask_fn_dict:
                self.image_files.append(os.path.join(image_path, fn))
                self.mask_files.append(os.path.join(mask_path, curr_file_name[0] + mask_fn_dict[curr_file_name[0]]))

        self.length = len(self.image_files)

    def __getitem__(self, item):
        image = self.image_files[item]
        mask = self.mask_files[item]

        if self.is_rgb:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image, mask = self.transform(image, mask, self.is_train)
        return image, mask.float() / 255.0

    def __len__(self):
        return self.length