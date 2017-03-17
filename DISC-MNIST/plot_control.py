import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from disc_mnist import train

def main():
    plt.title("Boundary-seeking GAN")
    plt.xlabel("Training batches")
    plt.ylabel("Generator loss")
    train(num_epochs=1, n_samples=20, initial_eta=1e-5, plot_colour="-b")
    train(num_epochs=1, n_samples=20, initial_eta=1e-4, plot_colour="-r")
    train(num_epochs=1, n_samples=20, initial_eta=1e-3, plot_colour="-g")
    plt.grid()
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    art.append(lgd)
    plt.draw()
    plt.savefig('gen_plots/RWGAN.png',
                additional_artists=art,
                bbox_inches="tight")

if __name__ == '__main__':
    main()
