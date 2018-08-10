import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backend_bases import NavigationToolbar2

from evlife import *

def centralise(a):
    return [[a[i - (a.dimx // 2)][j - (a.dimy // 2)] for j in range(a.dimy)] for i in range(a.dimx)]

class MRenderer:
    def __init__(self, f, params, toolbox, ind, seed = None):
        self.fld = Field(params["microgens"] + params["initrad"] * 2 + 5, 
                         params["microgens"] + params["initrad"] * 2 + 5,
                         lambda x: ngen_nn2(ind, x),
                         populate = lambda x: toolbox.genesis(seed, x))
        self.rendered = [centralise(self.fld)]
        self.index    = 0  
        self.fig = f
        self.cid = f.canvas.mpl_connect('button_press_event', self)
        self.render()
    
    def render(self):
        plt.imshow(self.rendered[self.index], interpolation='nearest')
        self.fig.canvas.set_window_title(str(self.index))

def mback(self, *args, **kwargs):
    if renderer.index > 0:
        renderer.index -= 1
    renderer.render()

def mforward(self, *args, **kwargs):
    renderer.index += 1
    if renderer.index == len(renderer.rendered):
        renderer.fld.nextgen()
        renderer.rendered.append(centralise(renderer.fld))
    renderer.render()

def mhome(self, *args, **kwargs):
    renderer.index = 0
    renderer.render()
        
NavigationToolbar2.back    = mback
NavigationToolbar2.forward = mforward
NavigationToolbar2.home    = mhome

renderer = None

def plotind(ind, params, toolbox, seed = None):
    global renderer
    fig = plt.figure()
    ax = fig.add_subplot(111)
    renderer = MRenderer(fig, params, toolbox, ind, seed=seed)
    plt.show()
