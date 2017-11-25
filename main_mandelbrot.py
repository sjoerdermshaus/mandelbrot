import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import itertools
import multiprocessing as mp
import timeit
import pickle


class CMandelbrot:

    def __init__(self, coordinates, max_iter, scale=1, nprocesses=0):
        self.coordinates = coordinates
        self.max_iter = max_iter
        self.scale = scale
        self.nprocesses = nprocesses
        self.pixels = self._calculate_mandelbrot_set()

    def _calculate_mandelbrot_pixel(self, px, py):
        x = 0
        y = 0
        iter_counter = 0
        niter = self.max_iter
        b = 1
        while (iter_counter < self.max_iter) and (b == 1):
            x_temp = x * x - y * y + px
            y = 2 * x * y + py
            x = x_temp
            if x * x + y * y > 2 * 2:
                niter = iter_counter + 1
                b = 0
            iter_counter = iter_counter + 1

        return b, niter

    def _calculate_mandelbrot_set(self):
        x_min, x_max, y_min, y_max = self.coordinates

        x_step = (x_max - x_min) / 3200.0
        y_step = (y_max - y_min) / 1800.0
        x_axis = np.arange(x_min, x_max, self.scale * x_step)
        y_axis = np.arange(y_min, y_max, self.scale * y_step)

        pixels_temp = np.matrix(list(itertools.product(x_axis, y_axis)))
        npixels = len(pixels_temp)
        print(npixels)
        index = range(0, npixels)
        columns = ['x', 'y', 'niter']
        data = np.hstack((pixels_temp, np.zeros((npixels, 1))))

        pixels = pd.DataFrame(data=data, columns=columns, index=index)

        if self.nprocesses == 0:
            pool = mp.Pool()
        else:
            pool = mp.Pool(self.nprocesses)
        iterable = ((pixels.x[pixel_i], pixels.y[pixel_i]) for pixel_i in range(0, npixels))
        results = pool.starmap_async(self._calculate_mandelbrot_pixel, iterable=iterable).get()
        pool.close()
        pixels.niter = [result[1] for result in results]

        return pixels


# def main_performance():
#
#     ncpus = 8
#     performance = pd.DataFrame(columns=['ncpus', 'time'], data=np.zeros((ncpus, 2)))
#     for k in range(1, ncpus + 1):
#         performance['ncpus'].iloc[k - 1] = k
#
#         start_time = timeit.default_timer()
#
#         max_iter = 1000
#         scale = 10
#         columns = ['x_min', 'x_max', 'y_min', 'y_max']
#         data = np.array([[-2.00, 1.00, -1.00, 1.00],
#                          [-0.7470, -0.7445, 0.1110, 0.1135],
#                          [-0.7463 - 0.005, -0.7463 + 0.005, 0.1102 - 0.005, 0.1102 + 0.005],
#                          [-0.7459, -0.7456, 0.1098, 0.1102]])
#         coordinates_set = pd.DataFrame(columns=columns, data=data)
#
#         f, ax = plot.subplots(2, 2)
#         for i in range(0, len(coordinates_set)):
#             coordinates = coordinates_set.iloc[i]
#             mb = f_mandelbrot_set(coordinates, max_iter, scale, k)
#             ax[i // 2, i % 2].scatter(x=mb.x, y=mb.y, c=mb.niter / max_iter, cmap='RdYlBu', marker='.')
#             ax[i // 2, i % 2].set_xlim([coordinates_set.x_min.iloc[i], coordinates_set.x_max.iloc[i]])
#             ax[i // 2, i % 2].set_ylim([coordinates_set.y_min.iloc[i], coordinates_set.y_max.iloc[i]])
#             if i + 1 < len(coordinates_set):
#                 x_min = coordinates_set.x_min.iloc[i + 1]
#                 y_min = coordinates_set.y_min.iloc[i + 1]
#                 dx = coordinates_set.x_max.iloc[i + 1] - coordinates_set.x_min.iloc[i + 1]
#                 dy = coordinates_set.y_max.iloc[i + 1] - coordinates_set.y_min.iloc[i + 1]
#
#                 ax[i // 2, i % 2].add_patch(patches.Rectangle((x_min, y_min), dx, dy, fill=False))
#
#         performance['time'].iloc[k - 1] = timeit.default_timer() - start_time
#     print(performance)


class CRuns:

    def __init__(self, data, file_name):
        self.data = data
        self.file_name = file_name
        columns = ['x_min', 'x_max', 'y_min', 'y_max']
        self.coordinates_set = pd.DataFrame(columns=columns, data=data)
        self.nruns = len(self.coordinates_set)

    def plot_data(self, colormap='Blues', invert=False):
        f, ax = plot.subplots(2, 2)
        for i in range(0, self.nruns):
            file_name_mb = 'run_{}.pickle'.format(i)
            mb = pickle.load(open(file_name_mb, 'rb'))
            if invert is False:
                c = mb.pixels.niter / mb.max_iter
            else:
                c = (mb.max_iter - mb.pixels.niter) / mb.max_iter
            ax[i // 2, i % 2].scatter(x=mb.pixels.x, y=mb.pixels.y, c=c, cmap=colormap, marker='.')
            ax[i // 2, i % 2].set_xlim([self.coordinates_set.x_min.iloc[i], self.coordinates_set.x_max.iloc[i]])
            ax[i // 2, i % 2].set_ylim([self.coordinates_set.y_min.iloc[i], self.coordinates_set.y_max.iloc[i]])
            if i + 1 < self.nruns:
                x_min = self.coordinates_set.x_min.iloc[i + 1]
                y_min = self.coordinates_set.y_min.iloc[i + 1]
                dx = self.coordinates_set.x_max.iloc[i + 1] - self.coordinates_set.x_min.iloc[i + 1]
                dy = self.coordinates_set.y_max.iloc[i + 1] - self.coordinates_set.y_min.iloc[i + 1]
                ax[i // 2, i % 2].add_patch(patches.Rectangle((x_min, y_min), dx, dy, fill=False))

        fig_manager = plot.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plot.show()


def main():

    start_time = timeit.default_timer()

    data = np.array([[-2.00, 1.00, -1.00, 1.00],
                     [-0.58, -0.43, 0.52, 0.62],
                     [-0.48, -0.45, 0.59, 0.61],
                     [-0.4657, -0.46465, 0.5912, 0.5919]])
    file_name = 'runs.pickle'
    main_runs = CRuns(data, file_name)
    pickle.dump(main_runs, open(file_name, 'wb'))

    max_iter = 1000
    scale = 4

    for i in range(0, main_runs.nruns):
        coordinates = main_runs.coordinates_set.iloc[i]
        mb = CMandelbrot(coordinates, max_iter, scale, 5)
        file_name = 'run_{}.pickle'.format(i)
        pickle.dump(mb, open(file_name, 'wb'))

    print(timeit.default_timer() - start_time)
    main_runs.plot_data()


if __name__ == '__main__':
    do_run = 0
    if do_run == 1:
        main()
    else:
        file_name_runs = 'runs.pickle'
        runs = pickle.load(open(file_name_runs, 'rb'))
        runs.plot_data(colormap='coolwarm', invert=False)
