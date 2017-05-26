import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from gumbel_mnist import train as gumbel_train
from rw_mnist import train as rw_train
gumbel_hard_arr = ["True", "False"]
optimGD_arr = ['adam', 'rmsprop', 'sgd']
learning_rate_arr = [1e-2, 1e-3, 1e-4, 1e-5]
anneal_rate_arr = [0.01, 0.001, 0.0001]
anneal_interval_arr = [200, 300, 500]

def main():
    plt.title("Discriminator misclassification of generated samples")
    plt.xlabel("Training batches")
    plt.ylabel("Generator loss")
    gumbel_train(False, 'adam', 1e-4, 1e-3, 500, num_epochs=1, plot_colour="-b")
    gumbel_train(True, 'rmsprop', 1e-4, 1e-4, 200, num_epochs=1, plot_colour="-y")
    rw_train(num_epochs=1, n_samples=20, initial_eta=1e-4, plot_colour="-g")
    plt.grid()
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    art.append(lgd)
    plt.draw()
    plt.savefig('gen_plots/p_fake_comparison.png',
                additional_artists=art,
                bbox_inches="tight")

if __name__ == '__main__':
    main()
