import matplotlib.pyplot as plt
import emetrics  # from Hakime
import lifelines.utils
import numpy as np

# IF you really want to change something here about the output, feel free to do so. It is not a priority.

def pltcolor(lst):
    colors = []
    for item in lst:
        if item <= 0.5:
            colors.append('cornflowerblue')
        else:
            colors.append('red')
    return colors


def print_loss(loss_vector):
    plt.plot(loss_vector)
    plt.title("Loss over Batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig("loss_over_batch.png")
    plt.show()


def print_output(predicted, labels, data_used, cols='cornflowerblue'):

    if data_used[0] == "pkd":
        plt.scatter(predicted, labels, alpha=0.2, color=cols, edgecolors="black")
        plt.plot([4, 11], [4, 11], ls="--", c=".3")
        plt.xlim(4, 11)
        plt.ylim(4, 11)

    if data_used[0] == "kiba":
        plt.scatter(predicted, labels, alpha=0.2, color=cols, edgecolors="black")
        plt.plot([7, 18], [7, 18], ls="--", c=".1")
        plt.xlim(7, 18)
        plt.ylim(7, 18)

    plt.title(data_used[1])
    plt.xlabel("Predicted")
    plt.ylabel("Measured")
    plt.savefig("scatterplot.png")
    plt.show()


def only_rm2(all_predicted, all_labels):
    return emetrics.get_rm2(all_labels, all_predicted)


def bootstrap_stats(all_predicted, all_labels, data_used):
    rm2 = emetrics.get_rm2(all_labels, all_predicted)
    aupr = emetrics.compute_aupr(all_labels, all_predicted, data_used[0])
    ci = lifelines.utils.concordance_index(all_labels, all_predicted)
    mse = np.square(np.subtract(all_labels, all_predicted)).mean()

    return [rm2, aupr, ci, mse]


def print_stats(all_predicted, all_labels, data_used):
    rm2 = emetrics.get_rm2(all_labels, all_predicted)
    aupr = emetrics.compute_aupr(all_labels, all_predicted, data_used[0])
    ci = lifelines.utils.concordance_index(all_labels, all_predicted)
    mse = np.square(np.subtract(all_labels, all_predicted)).mean()

    print("The rm2 value for this run is: ", round(rm2, 3))

    print("The AUPR for this run is: ", round(aupr, 3))

    print("The Concordance Index (CI) for this run is: ", round(ci, 3))

    print("The Mean Squared Error (MSE) for this run is: ", round(mse, 3))
