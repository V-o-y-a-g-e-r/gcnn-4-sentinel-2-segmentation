import os

import fire
import numpy as np
from eolearn.core import LoadTask
from matplotlib import pyplot as plt


def run_visualizations(sample_path: str, dest_path: str):
    load_task = LoadTask(path=os.path.dirname(sample_path))
    os.makedirs(dest_path, exist_ok=True)
    data = load_task.execute(
        eopatch_folder=os.path.basename(sample_path))
    x = data[('data', 'BANDS')].astype(np.float32)
    y = data[('mask_timeless', 'CULTIVATED')].squeeze()
    plt.imshow(y, cmap='gray_r')
    plt.axis('off')
    plt.savefig(os.path.join(dest_path, 'gt.png'),
                dpi=1000, bbox_inches='tight', pad_ionches=0)
    plt.clf()
    for i in range(x.shape[0]):
        x_t = x[i, :, :, [3, 2, 1]]
        x_t = np.rollaxis(x_t, 0, 3)
        x_t = (x_t - x_t.min()) / (x_t.max() - x_t.min())
        plt.imshow(x_t)
        plt.axis('off')
        plt.savefig(os.path.join(
            dest_path, 'frame_' + str(i) + '.png'),
            dpi=1000, bbox_inches='tight', pad_ionches=0)
        plt.clf()


if __name__ == '__main__':
    fire.Fire(run_visualizations)
