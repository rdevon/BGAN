#!/usr/bin/env python
import itertools
import synthetic
import os
import time

l1list = [1.0]

disc_lr = 0.0001
gen_lr = 0.0001
epochs = 20
batch_size = 512
num_elements_per_gaussian=100000

def main():
    for l1 in itertools.product(l1list):
        print("=========================================================")
        print("=========================================================")
        print("=========================================================")
        current_time = time.time()

        save_dir = '{}-{}-{}'.format(
            "synthetic",
            "mix_gaussians",
            str(int(current_time)))
        save_dir = os.path.join('./save/', save_dir)
        image_dir = save_dir + '/images/'
        log_dir = save_dir + '/logs'

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            os.mkdir(image_dir)
            os.mkdir(log_dir)

        filename = os.path.join(log_dir, 'log_train.txt')

        synthetic.train(epochs,
                        filename,
                        batch_size=batch_size,
                        num_elements=num_elements_per_gaussian,
                        disc_lr=disc_lr,
                        gen_lr=gen_lr,
                        image_dir=image_dir)

if __name__ == '__main__':
    main()