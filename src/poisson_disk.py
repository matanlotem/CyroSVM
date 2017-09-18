import scipy
import random

#reference: http://connor-johnson.com/2015/04/08/poisson-disk-sampling/
#article: http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
class pds:
    """
    generates random points that are separated from one another in a given distance using a poisson disk process
    """

    def __init__(self, w, h, d, r, n, k=30):
        """
        for 2D use d=1
        :param w: x dim size
        :param h: y dim size
        :param d: z dim size
        :param r: minimal separation
        :param n: amount of points
        :param k: amount of points to try around each initial point (default 30)
        """
        # w and h are the width and height of the field
        self.w = w
        self.h = h
        self.d = d

        # n is the number of test points
        self.n = n
        #k is the number of attempts is a single sphere
        self.k = k
        self.r2 = r ** 2.0
        self.A = 3.0 * self.r2
        # cs is the cell size
        if d == 1:
            self.cs = r / scipy.sqrt(2)
        else:
            self.cs = r / scipy.sqrt(3)

        # gw and gh are the number of grid cells
        self.gw = int(scipy.ceil(self.w / self.cs))
        self.gh = int(scipy.ceil(self.h / self.cs))
        self.gd = int(scipy.ceil(self.h / self.cs))

        self.shape = [w, h, d]
        self.bin_shape = [1, self.gw, self.gw * self.gh]

        # create a grid and a queue
        self.grid = [None] * self.gd * self.gw * self.gh
        self.queue = list()
        # set the queue size and sample size to zero
        self.qs, self.ss = 0, 0

    def distance(self, s):
        # find where (x,y,z) sits in the grid
        [x_idx, y_idx, z_idx] = self.find_bin(s)
        # determine a neighborhood of cells around (x,y,z)
        x0 = max(x_idx - 2, 0)
        y0 = max(y_idx - 2, 0)
        z0 = max(z_idx - 2, 0)
        x1 = max(x_idx - 3, self.gw)
        y1 = max(y_idx - 3, self.gh)
        z1 = max(z_idx - 3, self.gd)
        # search around (x,y,z)
        for z_idx in range(z0, z1):
            for y_idx in range(y0, y1):
                for x_idx in range(x0, x1):
                    step = self.to_3d([x_idx,y_idx,z_idx])
                    # if the sample point exists on the grid
                    if self.grid[step]:
                        p = self.grid[step]
                        dx = (p[0] - s[0]) ** 2.0
                        dy = (p[1] - s[1]) ** 2.0
                        dz = (p[2] - s[2]) ** 2.0
                        # and it is too close
                        if dx + dy + dz < self.r2:
                            # then barf
                            return False
        return True

    def find_bin(self, s):
        return [int(x / self.cs) for x in s]

    def to_3d(self, s):
        return sum([x[0]*x[1] for x in zip(s,self.bin_shape)])


    def set_point(self, s):
        self.queue.append(s)
        self.qs += 1

        # find where (x,y) sits in the grid

        step = self.to_3d(self.find_bin(s))

        assert(self.grid[step] == None)
        self.grid[step] = s
        self.ss += 1

        return s

    def create_point_grid(self):
        while self.ss < self.n:
            #print("iter")
            if (self.qs == 0):
                print("randomization failed- trying to randomize ", self.n, " points in a ", self.shape, " space with min separation ", int(self.r2**0.5))
                exit(1)
            idx_in_q = int(random.uniform(0,1) * self.qs)
            #print("inx:",idx_in_q)
            s = self.queue[idx_in_q]
            added_points = 0
            for i in range(self.k):

                r = scipy.sqrt(self.A * random.uniform(0, 1) + self.r2)
                phi = 2 * scipy.pi * random.uniform(0, 1)
                if (self.d == 1):
                    theta = scipy.pi / 2
                else:
                    theta = scipy.pi * random.uniform(0, 1)



                x = int(s[0] + r * scipy.sin(theta) * scipy.cos(phi))
                y = int(s[1] + r * scipy.sin(theta) * scipy.sin(phi))
                z = int(s[2] + r * scipy.cos(theta))

                rand_point = [x, y, z]
                if (x >= 0) and (x < self.w):
                    if (y >= 0) and (y < self.h):
                        if (z >= 0) and (z < self.d):
                            if (self.distance(rand_point)):
                                self.set_point(rand_point)
                                #print("iter3")
                                added_points += 1
                                if (self.ss >= self.n):
                                    break
            if (added_points == 0):
                #print("deleting")
                self.grid[self.to_3d(self.find_bin(s))] = None
                self.ss -= 1
                del self.queue[idx_in_q]
                self.qs -= 1

    def randomize_spaced_points(self):
        if self.ss == 0:
            s = [int(random.uniform(0,1) * i) for i in [self.w, self.h, self.d]]
            #print("seed:", s)
            self.set_point(s)
        self.create_point_grid()
        sample = list(filter(None, self.grid))
        return sample

if __name__=='__main__':
    from matplotlib import pyplot as plt
    r = 6
    n = 60
    for it in range(15):
        obj = pds(100, 80, 1, r, n)
        sample1 = obj.randomize_spaced_points()
        x = [x[0] for x in sample1]
        y = [x[1] for x in sample1]
        for i in enumerate(sample1):
            for j in range(i[0]+1, len(sample1)):
                if sum([(x[1]-x[0])**2 for x in zip(i[1],sample1[j])]) < r:
                    assert(False)
        assert(len(sample1) == n)

    plt.scatter(x, y)
    plt.axvline(x=0, ymin=10 / 90, ymax=80 / 90, color='red')
    plt.axvline(x=100, ymin=10 / 90, ymax=80 / 90, color='red')
    plt.axhline(y=80, xmin=20 / 140, xmax=120 / 140, color='red')
    plt.axhline(y=0, xmin=20 / 140, xmax=120 / 140, color='red')
    plt.show()

    r = 15
    n = 100
    for it in range(15):
        obj = pds(100, 80, 80, r, n)
        sample1 = obj.randomize_spaced_points()
        x = [x[0] for x in sample1]
        y = [x[1] for x in sample1]
        z = [x[2] for x in sample1]
        for i in enumerate(sample1):
            for j in range(i[0]+1, len(sample1)):
                if sum([(x[1]-x[0])**2 for x in zip(i[1],sample1[j])]) < r:
                    assert(False)
        assert(len(sample1) == n)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.margins(0, 0, 0)
    plt.show()

