import numpy as np
import matplotlib as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import multiprocessing as mp
import pickle as p

import fprop as fp
import myalgs as ma
from evlife import *
import plotlife

params = {
    "nind"      : 300,            #population size
    "macrogens" : 100,            #evolutionary generations
    "microgens" : 10,             #cellular generations
    "initrad"   : 5,              #species radius at g0
    "expnbr"    : 1.5,            #expected value of neighbors at g0
    "structure" : [24, 8, 8, 37], #NN layers
    "cxpb"      : 0.7,            #crossover probability
    "mutpb"     : 0.3,            #mutation probability
    "xoi"       : 0.5,            #crossover intensity
    "mi"        : 0.7,            #mutation intensity
    "mmu"       : 0,              #center of gaussian mutation
    "msigma"    : 0.6,            #standard deviation of mutation
    "tsize"     : 12,              #tournament size
    "algo"      : "mplstoch",     #EA name
    "res1"      : 300,            #reserved param 1
    "res2"      : None,           #reserved param 2
    "res3"      : None,           #reserved param 3
    "res4"      : None }          #reserved param 4

def wsim(a, b):
    for i, j in zip(a, b):
        if((i != j).any()):
            return False
    return True

def save(name, spop, slog, hof = None):
    f = open("species/" + name + ".sp", "wb")
    p.dump((spop, slog, params, hof), f)
    f.close()
    
def load(name):
    f = open("species/" + name + ".sp", "rb")
    temp = p.load(f)
    f.close()
    return temp

g = 0
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness, gen = 0)

toolbox = base.Toolbox()
hof = None
print("Input species' name or skip for starting from g0.")
spname = input()
if(spname == ''):
    pop = None
else:
    temp = load(spname)
    pop = temp[0]
    hof = temp[3]
    print("Perhaps you just wanted to investigate the species?[y/n]")
    if((lambda x: x[0] == 'y' or x[0] == 'Y')(input())):
        params = temp[2]
        params["macrogens"] = 0
    else:
        print("Load the parameters aswell?[y/n]")
        if((lambda x: x[0] == 'y' or x[0] == 'Y')(input())):
            params = temp[2] 
        print("Adjust the parameters. Input an empty line when you are done.")
        while True:
            com = input()
            if com == '':
                break
            key, val = com.split()
            params[key] = eval(val)

toolbox.register("preind", fp.randinit, params["structure"])
toolbox.register("individual", lambda f: creator.Individual(f()), toolbox.preind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("genesis", randpopp, params["initrad"], params["expnbr"] / 8)

if(pop == None):
    pop = toolbox.population(n=params["nind"])

def evalNN(ind):
    fld = Field(params["microgens"] + params["initrad"] * 2 + 5, 
                params["microgens"] + params["initrad"] * 2 + 5,
                lambda x: ngen_nn2(ind, x),
                populate = lambda x: toolbox.genesis(ind.gen, x))
    cnt = params["microgens"]
    while(fld.nextgen() > 0  and  cnt > 0):
        cnt -= 1
    return params["microgens"] - cnt + fld.alive,

def genotype(ind):
    temp = creator.Individual([i for sublist in ind for i in sublist])
    ind[:] = temp
    return temp
    
def phenotype(ind):
    temp = []
    ind0 = ind
    for size in params["structure"][1:]:
        temp.append(np.array(ind0[:size]))
        ind0 = ind0[size:]
    temp = creator.Individual(temp)
    ind[:] = temp
    return temp
    

def mutateNN(ind):
    gen = ind.gen
    genotype(ind)
    tools.mutGaussian(ind, params["mmu"], params["msigma"], params["mi"])
    phenotype(ind)
    ind.gen = gen
    return ind,

def mateNN(ind1, ind2):
    gen = ind1.gen
    genotype(ind1)
    genotype(ind2)
    tools.cxUniform(ind1, ind2, params["xoi"])
    phenotype(ind1)
    phenotype(ind2)
    ind1.gen = gen
    ind2.gen = gen
    return (ind1, ind2)

if(hof == None):   
    hof = tools.HallOfFame(5, similar = wsim)

def selectNN(pop, k):
    return tools.selTournament(pop, k, params["tsize"])

def posneighbourhood(inputs):
    grid = [[0] * 7, [0] + inputs[:5] + [0], [0] + inputs[5:10] + [0], [0] + inputs[10:12] + [1] + inputs[12:14] + [0], [0] + inputs[14:19] + [0], [0] + inputs[19:] + [0], [0] * 7]
    for i in range(5):
        for j in range(5):
            if sum([grid[1 + i + k][1 + j + l] for k, l in local]) > 3:
                return False
    return True

def complexity(ind, sampling = 500):
    reg = {}
    for i in range(sampling):
        inputs = list(np.random.randint(2, size = 24))
        while not posneighbourhood(inputs):
            inputs = list(np.random.randint(2, size = 24))
        temp = str(decision2(fp.fprop(ind, inputs, fp.sigmoid3, None)))
        if temp in reg:
            reg[temp] += 1
        else:
            reg[temp] = 1
    res = [(k, '%.1f%%' % (100 * reg[k] / sampling)) for k in sorted(reg, key=reg.get, reverse=True)]
    for i, j in res:
        print(i, j)


pool = mp.Pool()
toolbox.register("map", pool.map)
toolbox.register("evaluate", evalNN)
toolbox.register("mate", mateNN)
toolbox.register("mutate", mutateNN)
toolbox.register("select", selectNN)


stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

if params["algo"] == "simple":
    fpop, log = ma.eaSimpleStochastic(pop, toolbox, cxpb=params["cxpb"], mutpb=params["mutpb"], ngen=params["macrogens"], stats=stats, halloffame=hof, verbose=True)
elif params["algo"] == "mplstoch":
    fpop, log = ma.eaMuPlusLambdaStochastic(pop, toolbox, len(pop), params["res1"], params["cxpb"], params["mutpb"], params["macrogens"], stats=stats, halloffame=hof, verbose=True)
else:
    print("\"" + params["algo"] + "\" is not an available evolution strategy.")
