import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import multiprocessing as mp
import timeit
import pickle
import datetime as dt


class CMandelbrot:

    def __init__(self, coordinates, max_iter, file_name=None, scale=1.0, nprocesses=0):
        self.coordinates = coordinates
        self.max_iter = max_iter
        self.scale = scale
        self.nprocesses = nprocesses
        self.pixels = []
        self.file_name = file_name

    def run(self):
        self.pixels = self._calculate_mandelbrot_set()
        if self.file_name is not None:
            self.save(self.file_name)

    def save(self, file_name):
        pickle.dump(self, open(file_name, 'wb'))

    @staticmethod
    def load(file_name):
        return pickle.load(open(file_name, 'rb'))

    def _calculate_mandelbrot_pixel(self, px, py):
        """
        This method determines for a pixel if it's a member of the Mandelbrot set. If it's not a member, it
        returns the number of iterations to escape. For members of the Mandelbrot set, the number of iterations equals
        max_iter.
        :param px: x-coordinate of pixel
        :param py: y-coordinate of pixel
        :return: b: membership boolean, niter: number of iterations to "escape"
        """
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

    def _calculate_mandelbrot_pixel2(self, px, py):
        """
        This method determines for a pixel if it's a member of the Mandelbrot set. If it's not a member, it
        returns the number of iterations to escape. For members of the Mandelbrot set, the number of iterations equals
        max_iter.
        :param px: x-coordinate of pixel
        :param py: y-coordinate of pixel
        :return: b: membership boolean, niter: number of iterations to "escape"
        """
        iter_counter = 0
        niter = self.max_iter
        b = 1
        c = px + 1j * py
        z = 0
        while (iter_counter < self.max_iter) and (b == 1):
            z = z ** z + c ** 3
            if abs(z) > 2:
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
        print('{:d} x {:d} = {:d} pixels'.format(len(x_axis), len(y_axis), npixels))
        index = range(0, npixels)
        columns = ['x', 'y', 'niter']
        data = np.hstack((pixels_temp, np.zeros((npixels, 1))))

        pixels = pd.DataFrame(data=data, columns=columns, index=index).sample(frac=1)

        if self.nprocesses == 0:
            pool = mp.Pool()
        else:
            pool = mp.Pool(self.nprocesses)
        iterable = ((pixels.x[pixel_i], pixels.y[pixel_i]) for pixel_i in range(0, npixels))
        results = pool.starmap_async(self._calculate_mandelbrot_pixel, chunksize=1000, iterable=iterable).get()
        pool.close()
        pixels.sort_index(inplace=True)
        pixels.niter = [result[1] for result in results]

        return pixels


class CRuns:

    def __init__(self, data, file_name=None):
        self.data = data
        self.file_name = file_name
        columns = ['x_min', 'x_max', 'y_min', 'y_max']
        self.coordinates_set = pd.DataFrame(columns=columns, data=data)
        self.nruns = len(self.coordinates_set)

    def plot_data(self, colormap='RdYlGn', invert=False, add_rectangle=True, dpi=100):

        print('Plotting')
        start_time = timeit.default_timer()

        # for now, 4 subplots are sufficient
        fig, ax = plt.subplots(2, 2)

        inch_width = 32
        inch_height = 18
        fig.set_size_inches(inch_width, inch_height)

        # loop over the (4) runs
        for i in range(0, self.nruns):

            # load mandelbrot set
            file_name_mb = 'run_{}.pickle'.format(i)
            mb = CMandelbrot.load(file_name_mb)
            mb.pixels.to_csv(open('run_{}.csv'.format(i), 'w'), index=False)

            # invert colormap if necessary
            if invert is False:
                c = mb.pixels.niter / mb.max_iter
            else:
                c = 1 - mb.pixels.niter / mb.max_iter

            # make scatter plots and set axes
            ax[i // 2, i % 2].scatter(x=mb.pixels.x, y=mb.pixels.y, c=c, cmap=colormap, marker='.')
            ax[i // 2, i % 2].set_xlim([self.coordinates_set.x_min.iloc[i], self.coordinates_set.x_max.iloc[i]])
            ax[i // 2, i % 2].set_ylim([self.coordinates_set.y_min.iloc[i], self.coordinates_set.y_max.iloc[i]])
            ax[i // 2, i % 2].axis('off')

            # add rectangles: these are zoomed in and visualized in the next subplot
            if (add_rectangle is True) and (i + 1 < self.nruns):
                x_min = self.coordinates_set.x_min.iloc[i + 1]
                y_min = self.coordinates_set.y_min.iloc[i + 1]
                dx = self.coordinates_set.x_max.iloc[i + 1] - self.coordinates_set.x_min.iloc[i + 1]
                dy = self.coordinates_set.y_max.iloc[i + 1] - self.coordinates_set.y_min.iloc[i + 1]
                ax[i // 2, i % 2].add_patch(patches.Rectangle((x_min, y_min), dx, dy,
                                                              fill=False, color='white', linewidth=4.0))
                columns = ['x', 'y', 'niter']
                rx = np.arange(x_min, x_min + dx,  mb.scale * dx / 3200.0)
                rx = rx.reshape((len(rx), 1))
                data1 = np.hstack((rx, y_min * np.ones((len(rx), 1)), mb.max_iter * np.ones((len(rx), 1))))
                data2 = np.hstack((rx, (y_min + dy) * np.ones((len(rx), 1)), mb.max_iter * np.ones((len(rx), 1))))
                ry = np.arange(y_min, y_min + dy, mb.scale * dy / 1800.0)
                ry = ry.reshape((len(ry), 1))
                data3 = np.hstack((x_min * np.ones((len(ry), 1)), ry, mb.max_iter * np.ones((len(ry), 1))))
                data4 = np.hstack(((x_min + dx) * np.ones((len(ry), 1)), ry, mb.max_iter * np.ones((len(ry), 1))))
                data = np.vstack((data1, data2, data3, data4))
                df = pd.DataFrame(columns=columns, data=data)
                file_name_df = 'run_rectangle{}.csv'.format(i)
                df.to_csv(open(file_name_df, 'w'), index=False)

        print(elapsed_time(timeit.default_timer() - start_time))
        print('Plotting finished')

        print('Saving the plt')
        start_time = timeit.default_timer()

        now = dt.datetime.now()
        time_string = '{:4d}{:02d}{:02d}_{:02d}{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute)

        plt.savefig('{:s}_MandelbrotSet.png'.format(time_string), dpi=dpi)
        plt.show()
        # plt.close(fig)
        print(elapsed_time(timeit.default_timer() - start_time))
        print('Saving the plt finished')

        # x = mb.pixels.x.unique()
        # y = mb.pixels.y.unique()
        # X, Y = np.meshgrid(x, y)
        # Z = mb.pixels.niter.values.reshape(len(y), len(x))
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_trisurf(mb.pixels.x, mb.pixels.y, mb.pixels.niter, cmap='coolwarm_r', shade=False)
        #
        # # surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
        # # fig.colorbar(surf, shrink=0.5, aspect=5)

        # plt.show()

    @staticmethod
    def test_performance(scale=20, max_iter=1000):

        # test settings for mandelbrot set
        data = np.array([[-2.00, 1.00, -1.00, 1.00],
                         [-0.7470, -0.7445, 0.1110, 0.1135],
                         [-0.7463 - 0.005, -0.7463 + 0.005, 0.1102 - 0.005, 0.1102 + 0.005],
                         [-0.7459, -0.7456, 0.1098, 0.1102]])
        main_runs = CRuns(data)

        # record performance
        ncpus = 8
        performance = pd.DataFrame(columns=['ncpus', 'time'], data=np.zeros((ncpus, 2)))
        for k in range(1, ncpus + 1):
            print('ncpus: {:d}/{:d}'.format(k, ncpus))
            performance['ncpus'].iloc[k - 1] = int(k)
            start_time = timeit.default_timer()
            for i in range(0, main_runs.nruns):
                coordinates = main_runs.coordinates_set.iloc[i]
                mb = CMandelbrot(coordinates, max_iter, scale=scale, nprocesses=k)
                mb.run()

            performance['time'].iloc[k - 1] = timeit.default_timer() - start_time
        print(performance)

        plt.plot(performance.ncpus, performance.time, 'b-', performance.ncpus, performance.time, 'bo')
        plt.xlabel('Number of CPUs')
        plt.ylabel('time')
        plt.grid()

        now = dt.datetime.now()
        time_string = '{:4d}{:02d}{:02d}_{:02d}{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute)

        inch_width = 32
        inch_height = 18
        plt.gcf().set_size_inches(inch_width, inch_height)

        plt.savefig('{:s}_Performance.png'.format(time_string), dpi=100)
        plt.show()

    @staticmethod
    def main():

        start_time = timeit.default_timer()

        data = np.array([[-2.00000,  1.00000, -1.00000, 1.00000],
                         [-0.70000, -0.42000,  0.52000, 0.70670],
                         [-0.48000, -0.45000,  0.59000, 0.61000],
                         [-0.46570, -0.46465,  0.59120, 0.59190]])

        file_name = 'runs.pickle'
        main_runs = CRuns(data, file_name)
        pickle.dump(main_runs, open(file_name, 'wb'))

        max_iter = [50, 100, 250, 1000]
        # max_iter = [250, 500, 750, 1000]
        scales = [5 * 5.0/3.0] * 4
        # scales = [10] * 4

        for i in range(0, main_runs.nruns):
            start_time_loop = timeit.default_timer()
            coordinates = main_runs.coordinates_set.iloc[i]
            file_name = 'run_{}.pickle'.format(i)
            mb = CMandelbrot(coordinates, max_iter[i], file_name=file_name, scale=scales[i], nprocesses=5)
            mb.run()
            print(elapsed_time(timeit.default_timer() - start_time_loop))

        print('--------')
        print(elapsed_time(timeit.default_timer() - start_time))
        print('--------')
        main_runs.plot_data()


def elapsed_time(e):
    m, s = divmod(e, 60)
    h, m = divmod(m, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(h), int(m), int(s))


if __name__ == '__main__':
    run_type = 'pltklklk'
    if run_type == 'calc':
        CRuns.main()
    elif run_type == 'plt':
        file_name_runs = 'runs.pickle'
        runs = pickle.load(open(file_name_runs, 'rb'))
        for cm in plt.colormaps():
            runs.plot_data(colormap=cm, add_rectangle=True)
    elif run_type == 'test':
        CRuns.test_performance(20)
