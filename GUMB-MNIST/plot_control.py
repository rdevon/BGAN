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
    plt.title("Gumbel Softmax")
    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    train(False, 'adam', 1e-4, 1e-4, 300, num_epochs=1, plot_colour="-b")
    train(False, 'adam', 1e-3, 1e-3, 200, num_epochs=1, plot_colour="-r")
    train(False, 'rmsprop', 1e-4, 1e-4, 300, num_epochs=1, plot_colour="-y")
    train(False, 'rmsprop', 1e-4, 1e-2, 300, num_epochs=1, plot_colour="-g")
    plt.grid()
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    art.append(lgd)
    plt.draw()
    plt.savefig('gen_plots/gumbel_softmax.png',
                additional_artists=art,
                bbox_inches="tight")

if __name__ == '__main__':
    main()
