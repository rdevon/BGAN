import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from disc_mnist import train
gumbel_hard_arr = ["True", "False"]
optimGD_arr = ['adam', 'rmsprop', 'sgd']
learning_rate_arr = [1e-2, 1e-3, 1e-4, 1e-5]
anneal_rate_arr = [0.01, 0.001, 0.0001]
anneal_interval_arr = [200, 300, 500]

def main():
    plt.title("GAN")
    plt.xlabel("Training batches")
    plt.ylabel("Generator loss")
    train(False, 'adam', 1e-2, 1e-3, 500, num_epochs=10, plot_colour="-b")
    train(False, 'adam', 1e-3, 1e-4, 200, num_epochs=10, plot_colour="-y")
    train(False, 'adam', 1e-4, 1e-3, 200, num_epochs=10, plot_colour="-r")
    train(False, 'adam', 1e-5, 1e-3, 300, num_epochs=10, plot_colour="-g")
    plt.grid()
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    art.append(lgd)
    plt.draw()
    plt.savefig('gen_plots/gan.png',
                additional_artists=art,
                bbox_inches="tight")

if __name__ == '__main__':
    main()
