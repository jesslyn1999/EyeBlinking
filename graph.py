import matplotlib.pyplot as plt
import numpy as np


def save_ear_graph(data, output_filename):
    plt.xlabel('time(s)')
    plt.ylabel('EAR ratio')
    plt.plot(data[0], data[1])
    plt.savefig(output_filename)


def save_microsleep_perclos_graph(data, x_labels, output_filename):
    x = np.arange(len(x_labels))
    bar_width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x + bar_width / 2, data[0], bar_width, label='microsleep freq')
    rects2 = ax.bar(x + bar_width * 3 / 2, data[1], bar_width, label='perclos(in %)')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    ax.set_title('Perclos in % & Microsleep in frequency')
    ax.set_xlabel('time(s)')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.savefig(output_filename)
