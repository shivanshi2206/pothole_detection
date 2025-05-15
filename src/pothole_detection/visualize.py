import os
import matplotlib.pyplot as plt
import numpy as np

def plot_image_label_distribution(dataset_path):
    splits = ['train', 'valid', 'test']
    image_counts = []
    label_counts = []

    for split in splits:
        image_path = os.path.join(dataset_path, split, 'images')
        label_path = os.path.join(dataset_path, split, 'labels')

        image_counts.append(len(os.listdir(image_path)))
        label_counts.append(len(os.listdir(label_path)))

    x = np.arange(len(splits))
    bar_width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - bar_width/2, image_counts, width=bar_width, label='Images', color='skyblue')
    plt.bar(x + bar_width/2, label_counts, width=bar_width, label='Labels', color='salmon')

    for i in range(len(splits)):
        plt.text(x[i] - bar_width/2, image_counts[i] + 10, str(image_counts[i]), ha='center')
        plt.text(x[i] + bar_width/2, label_counts[i] + 10, str(label_counts[i]), ha='center')

    plt.xticks(x, [s.capitalize() for s in splits])
    plt.ylabel("Count")
    plt.title("Images vs Labels per Dataset Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/split_distribution.png")
    plt.close()
