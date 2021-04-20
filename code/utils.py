import torch
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def plot_toy_regions(X,
                     T,
                     predict,
                     ax=None,
                     transform=None,
                     xlims=None,
                     ylims=None,
                     M=300):

    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()

    if torch.is_tensor(T):
        T = T.detach().cpu().numpy()

    T = T.squeeze()

    n_classes = int(np.max(T)) + 1

    if xlims is None:
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
    else:
        x_min, x_max = xlims
    x_range = x_max - x_min

    if ylims is None:
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()
    else:
        y_min, y_max = ylims
    y_range = y_max - y_min


    # Define the grid where we plot
    vx = np.linspace(x_min-0.1*x_range, x_max+0.1*x_range, M)
    vy = np.linspace(y_min-0.1*y_range, y_max+0.1*y_range, M)
    data_feat = np.zeros((M**2, 2), np.float32)

    XX, YY = np.meshgrid(vx, vy)

    # this can be done much more efficient for sure
    coords = np.array([XX, YY])
    data_feat = coords.transpose((1,2,0)).reshape([-1, 2])  

    if transform is not None:
        data_feat = transform(data_feat)

    # forward through the model
    preds = predict(data_feat)
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()

    # check if binary clasification
    if preds.shape[1] == 1:
        preds = np.hstack((1-preds, preds))

    max_conf, max_target = np.max(preds, axis=1), np.argmax(preds, axis=1)


    conf = max_conf.reshape([M, M])
    labl = max_target.reshape([M, M])

    cmap = [
            plt.cm.get_cmap("Reds"),
            plt.cm.get_cmap("Greens"),
            plt.cm.get_cmap("Blues"),
            plt.cm.get_cmap("Greys")
        ]

    color_list_tr = ['*r', '*g', '*b', '*k']
    color_list_te = ['orange', 'lightgreen', 'cyan', 'gray']
    markers = ['d', '*', 'P', 'v']
    # Build plot
    if ax is None:
        ax = plt.gca()

    for ctr, cte, i, c, marker in zip(color_list_tr[:n_classes],
                                      color_list_te[:n_classes],
                                      range(n_classes),
                                      cmap[:n_classes],
                                      markers[:n_classes]):

        idx = T == i
        x = X[idx, :]

        confs = np.zeros(conf.shape) * np.nan
        confs[labl==i] = conf[labl==i]

        x1, x2 = x[:, 0], x[:, 1]
        ax.contourf(vx, vy, confs,
        cmap=c,
        alpha=0.5,
        vmin=0,
        vmax=1, 
        levels=[
                0,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.92,
                0.94,
                0.96,
                0.98,
                1.0
            ])

        ax.plot(x1, x2, marker, color=cte, alpha=0.5)

    if transform is not None:
        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
    else:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    

    return ax