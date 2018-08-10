# Cellular automaton "species" decision-making evolution
This project is a purely experimental crossbreed between such concepts as artificial neural networks, evolutionary algorithms and cellular automatons.<br><br>
The aim of the project is to create a cellular automaton that resembles Conway's game of life, except it has complex orderly behavioral patterns. A single set of weights for a multilayer perceptron defines a species. The perceptron's inputs are states of spaces (dead or live cells) in a 5x5 square around the specimen. Each of the nodes in the output layer represents a decision concerning the specimen's reproduction. A decision consists of two hyperparameters: number of daughter cells to be spawned due next micro-generation* and the location of those cells in a 3x3 ring around the specimen. Each macro-generation* the cells of each species are randomly distributed in starting areas of their cellular planes (the distribution varies between macro-generations, but is kept the same between species within a population of each macro-generation). Fitness of a species is determined by the number of cells it managed to have in total by micro-generation N, where N is a predefined parameter.

*to avoid confusion between evolutionary generations and cellular automata generations they are referred to as macro-generations and micro-generations accordingly.
