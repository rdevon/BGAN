from PIL import Image
import h5py
import numpy as np
import os
from fuel.datasets.hdf5 import H5PYDataset


def normalize(x):
    return (x / 255.0)


def denormalize(x):
    return (x) * 255.0


class FileData:
    def __init__(self, loc, width, mb):

        self.lastIndex = 0

        if loc[-1] != "/":
            loc += "/"

        # images = glob.glob(loc + "**")

        images = []
        dirs = os.walk(loc)

        for dirp in dirs:
            for fi in dirp[2]:
                images.append(dirp[0] + "/" + fi)

        images_taken = []

        print("number of images total", len(images))

        for image in images:
            if "jpg" in image or "png" in image:
                images_taken.append(image)

        images = images_taken

        print("Number of images taken", len(images_taken))
        self.numExamples = len(images)
        self.images = images

        self.mb_size = mb
        self.image_width = width

    def convert2hdf5(self):


        imageLst = []

        index = self.lastIndex

        while True:

            if(index%10000==0):
                print("Current Index: ", index)

            image = self.images[index]
            try:
                imgObj = Image.open(image).convert('RGB')
            except:
                continue

            imgObj = imgObj.resize((self.image_width, self.image_width))
            img = np.asarray(imgObj)
            if img.shape == (self.image_width, self.image_width, 3):
                imageLst.append([img])

            index += 1
            if index >= self.numExamples:
                break

            imgObj.close()

        anime_npy = np.vstack(imageLst).astype('uint8')

        anime_npy = anime_npy.transpose(0, 3, 1, 2)

        f = h5py.File('anime_faces.hdf5', mode='w')

        anime_faces = f.create_dataset('features', anime_npy.shape, dtype = 'uint8')

        split_dict = {'train': {'features': (0, index)},
                      'test': {'features': (0, index)}}

        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        anime_faces[...] = anime_npy

        f.flush()
        f.close()



if __name__ == "__main__":

    loc = "/data/lisa/data/anime_faces/danbooru-faces/"

    hdf5_loc = "/data/lisa/data/anime_faces.hdf5"
    imageNetData = FileData(loc, 64, 64)
    print("loaded")

    imageNetData.convert2hdf5()
    print("Finished...")
