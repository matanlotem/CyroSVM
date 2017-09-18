import random

class naive_spaced_randomizer:
    def __init__(self, w, h, d, r, n, max_tries=10000, max_iterations=10):
        """
        naive randomization- randomize and check if it is separated from all other points
        :param w: x dim size
        :param h: y dim size
        :param d: z dim size, use 1 for 2D
        :param r: minimal separation
        :param n: num of points to randomize
        :param max_tries: give up after
        """
        self.w=w
        self.h=h
        self.d=d
        self.r2=r**2
        self.n=n
        self.max_tries = max_tries
        self.max_iterations = max_iterations


    def randomize_spaced_points(self):
        iterations = 0
        while iterations < self.max_iterations:
            points = self.distribute_points()
            if len(points) != 0:
                break
            iterations += 1
        if iterations == self.max_iterations:
            raise Exception("Randomization hit max tries and failed")
        return points


    def distribute_points(self):
        points = []
        for i in range(self.n):
            for j in range(self.max_tries):
                x = random.randint(0,self.w-1)
                y = random.randint(0, self.h-1)
                z = random.randint(0,self.d-1)
                point = (x,y,z)
                valid = True
                for p in points:
                    if (sum([(x[0]-x[1])**2 for x in zip(point,p)]) < self.r2):
                        valid = False
                        break
                if valid:
                    points.append(point)
                    break
                if j==self.max_tries-1:
                    points = []
                    return points
        return points


