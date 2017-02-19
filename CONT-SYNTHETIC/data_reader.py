import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt

class DataGenerator:
    def __init__(self, image_dir):
        self.X_train = None
        self.X_mask = None
        self.data_iterator = None
        self.data_loaded = False
        self.num_batches = 0
        self.image_dir = image_dir
        self.batch_iterator_index = 0

    def load_training_data(self, batch_size, num_elements=10000):
        mean1 = [0, 0]
        mean2 = [2, 0]
        mean3 = [0, 2]
        mean4 = [0, -2]
        mean5 = [-2, 0]
        cov = np.diag([0.002, 0.002])

        #x1, y1 = np.random.multivariate_normal(mean=mean1, cov=cov, size=num_elements).T
        x2, y2 = np.random.multivariate_normal(mean=mean2, cov=cov, size=num_elements).T
        x3, y3 = np.random.multivariate_normal(mean=mean3, cov=cov, size=num_elements).T
        x4, y4 = np.random.multivariate_normal(mean=mean4, cov=cov, size=num_elements).T
        x5, y5 = np.random.multivariate_normal(mean=mean5, cov=cov, size=num_elements).T

        X = np.append(x2, [x3, x4, x5])
        Y = np.append(y2, [y3, y4, y5])
        indexes = np.arange(num_elements*4)
        np.random.shuffle(indexes)
        self.data_loaded = True
        self.data_matrix = np.transpose([X, Y])
        self.batch_size = batch_size
        self.X_shuffled = X[indexes]
        self.Y_shuffled = Y[indexes]
        self.num_batches = num_elements/batch_size
        plt.title("Figure plot")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.axis([-10.0, 10.0, -10.0, 10.0])
        plt.scatter(self.X_shuffled, self.Y_shuffled)
        plt.grid()
        plt.legend()
        plt.draw()
        plt.savefig(self.image_dir + '/Original')
        plt.clf()
        plt.cla()
        plt.close()


    def get_batch(self):
        if not self.data_loaded:
            raise Exception("Data not loaded!")
        iter_num = self.batch_iterator_index
        batch_size = self.batch_size
        batch = self.data_matrix[batch_size*iter_num:(iter_num+1)*batch_size]
        self.batch_iterator_index+=1
        return batch

    def reset(self):
        self.batch_iterator_index = 0

