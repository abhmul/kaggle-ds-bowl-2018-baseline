from ww import f

from utils import Dataset
from glob import glob
import os
import numpy as np
import re
import cv2

class BowlDataset(Dataset):


    def load_bowl(self, base_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        self.add_class("bowl", 1, "nuclei")

        masks = dict()
        id_extractor = re.compile(f("{base_path}\{os.sep}(?P<image_id>.*)\{os.sep}masks\{os.sep}(?P<mask_id>.*)\.png"))

        for mask_path in glob(os.path.join(base_path, "**", "masks", "*.png")):
            matches = id_extractor.match(mask_path)

            image_id = matches.group("image_id")
            image_path = os.path.join(base_path, image_id, "images", image_id + ".png")

            if image_path in masks:
                masks[image_path].append(mask_path)
            else:
                masks[image_path] = [mask_path]

        for i, (image_path, mask_paths) in enumerate(masks.items()):
            self.add_image("bowl", image_id=i, path=image_path, mask_paths=mask_paths)

    def normalize(self, img, max_val=255):
        mean = np.average(img)
        if mean > (max_val / 2):
            import matplotlib.pyplot as plt
            plt.imshow(max_val - img)
            plt.show()
            raise ValueError
            return max_val - img
        return img

    def load_image(self, image_id):
        info = self.image_info[image_id]

        return self.normalize(cv2.imread(info["path"]), max_val=255)


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)


    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_paths = info["mask_paths"]
        count = len(mask_paths)
        masks = []

        for i, mask_path in enumerate(mask_paths):
            masks.append(cv2.imread(mask_path, 0))

        masks = np.stack(masks, axis=-1)
        masks = np.where(masks > 128, 1, 0)
        
        class_ids = np.ones(count)
        return masks, class_ids.astype(np.int32)
