import os
import pandas as pd
from skimage.transform import resize
import numpy as np
import imageio
import pydicom
from pydicom.pixel_data_handlers import util as dcm_util

import albumentations as A

# https://albumentations-demo.herokuapp.com
aug_pipeline_aggressive = A.Compose([
    A.ShiftScaleRotate(),
    A.GridDistortion(p=0.5, num_steps=5),
])
# below 2 were removed from above after epoch 6
# A.Cutout(p=0.5, num_holes=8, max_h_size=8, max_w_size=8),
# A.ChannelShuffle(p=0.5),

aug_pipeline = A.Compose([
    A.ShiftScaleRotate(p=0.5),
])
MIN_VAL,MAX_VAL = -1000,1000

class DataLoader():
    def __init__(self, dataset_name, augment=False, img_res=(256,256)):
        
        if dataset_name == 'c4kc-kits':
            self.mydf = pd.read_csv("../prepare/data.csv")
        else:
            raise NotImplementedError()

        self.dataset_name = dataset_name
        self.img_res = img_res
        self.augment = augment

    def load_data(self, domain, batch_size=1):
        
        path = list(self.mydf[self.mydf.series_description==domain].dcm_file)
        print(domain, len(path))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            img = resize(img, self.img_res)
            if self.augment:
                augmented = aug_pipeline(
                    image=img,
                )
                img = augmented['image']
            imgs.append(img)
        
        imgs = np.array(imgs)
        #print(imgs.shape)
        return imgs

    def load_batch(self, batch_size=1):

        path_A = list(self.mydf[self.mydf.series_description=="noncontrast"].dcm_file)
        path_B = list(self.mydf[self.mydf.series_description=="arterial"].dcm_file)

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):

                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = resize(img_A, self.img_res)
                img_B = resize(img_B, self.img_res)
                

                if self.augment:
                    augmented = aug_pipeline(
                        image=img_A,
                    )
                    img_A = augmented['image']

                    augmented = aug_pipeline(
                        image=img_B,
                    )
                    img_B = augmented['image']

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            #print(imgs_A.shape,imgs_B.shape)
            yield imgs_A, imgs_B

    def load_img(self, path):
        raise NotImplementedError()
        img = self.imread(path)
        img = resize(img, self.img_res)
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        ds = pydicom.dcmread(path)
        arr = dcm_util.apply_modality_lut(ds.pixel_array, ds)
        arr = ((arr-MIN_VAL)/(MAX_VAL-MIN_VAL)).clip(0,1)
        return arr

if __name__ == "__main__":
    
    os.makedirs('static',exist_ok=True)

    dl = DataLoader('c4kc-kits',augment=True)
    
    out = dl.load_data("noncontrast",batch_size=2)
    assert(out.shape==(2,256,256))
    
    out = dl.load_data("arterial",batch_size=2)
    assert(out.shape==(2,256,256))
    batch_size = 8
    for n,batch in zip(range(1),dl.load_batch(batch_size=batch_size)):
        A, B = batch
        A = (A*255).astype(np.uint8)
        B = (B*255).astype(np.uint8)
        assert(A.shape==(batch_size,256,256))
        assert(B.shape==(batch_size,256,256))
        for n in range(batch_size):
            imageio.imwrite(f"static/noncontrast-{n}.png",A[n,:,:].squeeze())
            imageio.imwrite(f"static/arterial-{n}.png",B[n,:,:].squeeze())

    print("done.")

'''
python data_loader.py
'''