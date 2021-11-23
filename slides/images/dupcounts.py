from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

if __name__ == '__main__':
    dupcounts = [234, 36, 51, 25, 11, 0, 0, 4, 0, 7, 0, 0, 0, 0, 0, 0, 1, 0]

    figure(figsize=(8, 4), dpi=80)

    plt.ylabel("# duplications")
    plt.xlabel("index of reorganisation step")
    plt.xticks(range(len(dupcounts)))
    plt.bar(range(len(dupcounts)), dupcounts)

    plt.title("Number of duplication events per reorganisation step")
    plt.tight_layout()
    plt.savefig("dupcounts")
    plt.show()