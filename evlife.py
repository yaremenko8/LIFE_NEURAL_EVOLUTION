import numpy as np
import itertools as its
import fprop as fp
import random as rnd

local = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

table2 = [()] + list(its.combinations(local, 1)) + list(its.combinations(local, 2))

decision2 = lambda x: table2[np.argmax(x)]

decision = decision2 #default

rad2 = (lambda x: x[:12] + x[13:])([(i - 2, j - 2) for i in range(5) for j in range(5)])
def ngen_nn2(weights, self):
    temp = [[0] * self.dimy for i in range(self.dimx)]
    for i in range(self.dimx):
        for j in range(self.dimy):
            if(self[i][j]):
                temp[i][j] = 1
                inputs = [self[(i + k) % self.dimx][(j + l) % self.dimy] for k, l in rad2]
                for m, n in decision2(fp.fprop(weights, inputs, fp.sigmoid3, None)):
                    temp[(i + m) % self.dimx][(j + n) % self.dimy] = 1
    self[:] = temp

def randpopp(r, p, seed, self): #expected value of neigbor cells is 8*p
    rnd.seed(a = seed)
    for i in range(2 * r):
        for j in range(2 * r):
            if(rnd.random() < p):
                self[i - r][j - r] = 1

class Field(list): #toroid space
    def __init__(self, dimx, dimy, ngen, populate = None, survive = lambda x: 2 <= x <= 3):
        temp = [[0 for j in range(dimy)] for i in range(dimx)]
        list.__init__(self, temp)
        self.dimx = dimx
        self.dimy = dimy
        self.__survive = survive
        self.__ngen = ngen
        if populate != None:
            self.alive = populate(self)        
        
    def __clear(self):
        alive = 0
        temp = [[0] * self.dimy for i in range(self.dimx)]
        for i in range(self.dimx): 
            for j in range(self.dimy):
                if(self[i][j]):
                    temp[i][j] = int(self.__survive(sum([self[(i + k) % self.dimx][(j + l) % self.dimy] for k, l in local])))
                    if(temp[i][j]):
                        alive += 1
        self.alive = alive
        self[:] = temp
    
    def nextgen(self):
        self.__ngen(self)
        self.__clear()
        return self.alive
    
    
        