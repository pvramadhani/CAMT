"""Camera-preserving version of MMT mutual-image preprocessor."""
from mmt.utils.data.preprocessor import Preprocessor
import os.path as osp
from PIL import Image


class CAPMutualPreprocessor(Preprocessor):
    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = osp.join(self.root, fname) if self.root is not None else fname
        image1 = Image.open(fpath).convert('RGB')
        image2 = image1.copy()
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, pid, camid
