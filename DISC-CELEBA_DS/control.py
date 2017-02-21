import itertools
import celeba
import os
import time

genlrlist = [1e-3]
disclrlist = [1e-3]

epochs = 200
def main():
    for gen_lr, disc_lr in itertools.product(genlrlist, disclrlist):
        print("=========================================================")
        print("=========================================================")
        print("=========================================================")

        current_time = time.time()

        save_dir = "celeba_rw_ds"

        save_dir = os.path.join('/home/devon/Outs/', save_dir)
        binary_dir = save_dir +  '/binaries/'
        image_dir = save_dir + '/images/'
        gt_image_dir = save_dir + '/gt_images/'
        log_dir = save_dir + '/logs'

        for d in [save_dir, binary_dir, image_dir, log_dir, gt_image_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        filename = os.path.join(log_dir, 'log_train.txt')

        celeba.train(epochs, filename,
                     disc_lr=disc_lr,
                     gen_lr=gen_lr,
                     image_dir=image_dir,
                     binary_dir=binary_dir,
                     gt_image_dir=gt_image_dir)


if __name__ == '__main__':
    main()