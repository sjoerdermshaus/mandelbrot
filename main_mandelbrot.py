import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import itertools
import multiprocessing as mp
import timeit


def f_mandelbrot_pixel(px, py, max_iter):
    x = 0
    y = 0
    iter_counter = 0
    niter = max_iter
    b = 1
    while (iter_counter < max_iter) and (b == 1):
        x_temp = x * x - y * y + px
        y = 2 * x * y + py
        x = x_temp
        if x * x + y * y > 2 * 2:
            niter = iter_counter + 1
            b = 0
        iter_counter = iter_counter + 1

    return b, niter


def f_mandelbrot_set(coordinates, max_iter, scale=1, nprocesses=0):
    x_min, x_max, y_min, y_max = coordinates

    x_step = (x_max - x_min) / 3200.0
    y_step = (y_max - y_min) / 1800.0
    x_axis = np.arange(x_min, x_max, scale * x_step)
    y_axis = np.arange(y_min, y_max, scale * y_step)

    pixels_temp = np.matrix(list(itertools.product(x_axis, y_axis)))
    npixels = len(pixels_temp)
    print(npixels)
    index = range(0, npixels)
    columns = ['x', 'y', 'niter']
    data = np.hstack((pixels_temp, np.zeros((npixels, 1))))

    pixels = pd.DataFrame(data=data, columns=columns, index=index)

    if nprocesses == 0:
        pool = mp.Pool()
    else:
        pool = mp.Pool(nprocesses)
    res = pool.starmap_async(f_mandelbrot_pixel,
                             iterable=((pixels.x[i], pixels.y[i], max_iter) for i in range(0, npixels))).get()
    pool.close()
    pixels.niter = [i[1] for i in res]

    return pixels


def main_performance():

    ncpus = 8
    performance = pd.DataFrame(columns=['ncpus', 'time'], data=np.zeros((ncpus, 2)))
    for k in range(1, ncpus + 1):
        performance['ncpus'].iloc[k - 1] = k

        start_time = timeit.default_timer()

        max_iter = 1000
        scale = 10
        columns = ['x_min', 'x_max', 'y_min', 'y_max']
        data = np.array([[-2.00, 1.00, -1.00, 1.00],
                         [-0.7470, -0.7445, 0.1110, 0.1135],
                         [-0.7463 - 0.005, -0.7463 + 0.005, 0.1102 - 0.005, 0.1102 + 0.005],
                         [-0.7459, -0.7456, 0.1098, 0.1102]])
        coordinates_set = pd.DataFrame(columns=columns, data=data)

        f, ax = plot.subplots(2, 2)
        for i in range(0, len(coordinates_set)):
            coordinates = coordinates_set.iloc[i]
            mb = f_mandelbrot_set(coordinates, max_iter, scale, k)
            ax[i // 2, i % 2].scatter(x=mb.x, y=mb.y, c=mb.niter / max_iter, cmap='RdYlBu', marker='.')
            ax[i // 2, i % 2].set_xlim([coordinates_set.x_min.iloc[i], coordinates_set.x_max.iloc[i]])
            ax[i // 2, i % 2].set_ylim([coordinates_set.y_min.iloc[i], coordinates_set.y_max.iloc[i]])
            if i + 1 < len(coordinates_set):
                x_min = coordinates_set.x_min.iloc[i + 1]
                y_min = coordinates_set.y_min.iloc[i + 1]
                dx = coordinates_set.x_max.iloc[i + 1] - coordinates_set.x_min.iloc[i + 1]
                dy = coordinates_set.y_max.iloc[i + 1] - coordinates_set.y_min.iloc[i + 1]

                ax[i // 2, i % 2].add_patch(patches.Rectangle((x_min, y_min), dx, dy, fill=False))

        performance['time'].iloc[k - 1] = timeit.default_timer() - start_time
    print(performance)


def main():

    start_time = timeit.default_timer()

    max_iter = 1000
    scale = 2
    columns = ['x_min', 'x_max', 'y_min', 'y_max']
    data = np.array([[-2.00, 1.00, -1.00, 1.00],
                     [-0.58, -0.43, 0.52, 0.62],
                     [-0.48, -0.45, 0.59, 0.61],
                     [-0.4657, -0.46465, 0.5912, 0.5919]])
    coordinates_set = pd.DataFrame(columns=columns, data=data)

    f, ax = plot.subplots(2, 2)
    for i in range(0, len(coordinates_set)):
        coordinates = coordinates_set.iloc[i]
        mb = f_mandelbrot_set(coordinates, max_iter, scale, 5)
        ax[i // 2, i % 2].scatter(x=mb.x, y=mb.y, c=mb.niter / max_iter, cmap='Spectral', marker='.')
        ax[i // 2, i % 2].set_xlim([coordinates_set.x_min.iloc[i], coordinates_set.x_max.iloc[i]])
        ax[i // 2, i % 2].set_ylim([coordinates_set.y_min.iloc[i], coordinates_set.y_max.iloc[i]])
        if i + 1 < len(coordinates_set):
            x_min = coordinates_set.x_min.iloc[i + 1]
            y_min = coordinates_set.y_min.iloc[i + 1]
            dx = coordinates_set.x_max.iloc[i + 1] - coordinates_set.x_min.iloc[i + 1]
            dy = coordinates_set.y_max.iloc[i + 1] - coordinates_set.y_min.iloc[i + 1]

            ax[i // 2, i % 2].add_patch(patches.Rectangle((x_min, y_min), dx, dy, fill=False))

    print(timeit.default_timer() - start_time)
    plot.show()


if __name__ == '__main__':
    main()
