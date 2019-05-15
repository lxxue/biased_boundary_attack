import os
import os.path as osp
import numpy as np
from keras.preprocessing import image

def main():
    img_list = []
    data_dir = "/mnt/lixxue/coco17val/val2017"
    for img_fname in sorted(os.listdir(data_dir)):
        img_list.append(osp.join(data_dir, img_fname))

    imgs = np.zeros((len(img_list), 299, 299, 3), dtype=np.float32)
    for i, img_fname in enumerate(img_list):
        imgs[i] = image.load_img(img_fname, target_size=(299, 299))
    print(imgs.shape)

    imgs -= 127.5
    imgs /= 127.5

    np.save("/mnt/lixxue/coco17val/processed_coco_val2017", imgs)
    

if __name__ == "__main__":
    main()
