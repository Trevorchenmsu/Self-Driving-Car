from math import *
import random


landmarks = [[20.0, 20.0],
             [80.0, 80.0],
             [20.0, 80.0],
             [80.0, 20.0]]

world_size = 100.0

class robot:
    def __init__(self):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= world_size:
            raise ValueError('X coordinate out of band')

        if new_y < 0 or new_y >= world_size:
            raise  ValueError('Y coordinate out of bound')

        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0, 2pi]')

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)


    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
            # make it possible to change the noise parameters
            # this is often useful in particle filters
            self.forward_noise = float(new_f_noise)
            self.turn_noise    = float(new_t_noise)
            self.sense_noise   = float(new_s_noise)

    def sense(self):
        Z = []
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            Z.append(dist)
        return Z

    def move(self, turn, forward):
        if forward < 0:
            raise ValueError('Robot cannot move backwards')

        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi

        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)

        x %= world_size
        y %= world_size

        # set particle
        res = robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res

    def Gaussian(self, mu, sigma, x):
        # Calculates the probability of x for 1-dim Gaussian with mean mu
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

    # the measurement is from different particles
    def measurement_prob(self, measurement):
        # Calculates how likely a measurement should be, which is an essential step
        prob = 1.0
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))

    def eval(r, p):
        sum = 0.0;
        for i in range(len(p)): # calculate mean error
            dx = (p[i].x - r.x + (world_size/2.0)) % world_size - (world_size/2.0)
            dy = (p[i].y - r.y + (world_size/2.0)) % world_size - (world_size/2.0)
            err = sqrt(dx * dx + dy * dy)
            sum += err
        return sum / float(len(p))

# myrobot = robot()
# myrobot.set_noise(5.0, 0.1, 5.0)
# myrobot.set(30.0, 50.0, pi/2)
# # print(myrobot)
# myrobot = myrobot.move(-pi/2, 15.0)
# # print(myrobot)
# print (myrobot.sense())
# myrobot = myrobot.move(-pi/2, 10.0)
# # print(myrobot)
# print (myrobot.sense())


myrobot = robot()
# print(myrobot)


# Create 1000 random particles in the world
N = 1000
T = 10

p = []
for i in range(N):
    x = robot()
    x.set_noise(0.05, 0.05, 5.0)
    p.append(x)


for t in range(T):
    myrobot = myrobot.move(0.1, 5.0)
    Z = myrobot.sense()

p2 = []
for i in range(N):
    p2.append(p[i].move(0.1, 5.0))
p = p2

w = []
for i in range(N):
    w.append(p[i].measurement_prob(Z))

p3 = []
index = int(random.random() * N)
beta = 0.0
mw = max(w)
for i in range(N):
    beta += random.random() * 2.0 * mw
    while beta > w[index]:
        beta -= w[index]
        index = (index + 1) % N
    p3.append(p[index])
p = p3
# print(p)

e = eval(myrobot, p)
print(e)


    # def move (myrobot, p,t, d):
    #     p2 = []
    #     for i in range(N):
    #         p2.append(p[i].move(t,d))
    #     p = p2
    #     myrobot = myrobot.move(t,d)
    #
    # def resample(myrobot, p):
    #     Z = myrobot.sense()
    #     w = []
    #     for i in range(N):
    #         w.append(p[i].measurement_prob(Z))
    #
    #     p3 = []
    #     index = int(random.random() * N)
    #     beta = 0.0
    #     mw = max(w)
    #     for i in range(N):
    #         beta += random.random() * 2.0 * mw
    #         while beta > w[index]:
    #             beta -= w[index]
    #             index = (index + 1) % N
    #         p3.append(p[index])
    #     p = p3

#
# print p #Leave this print statement for grading purposes!
