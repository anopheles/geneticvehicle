# Genetic Vehicle Design

![Genetiv Vehicle Sample Analysis](http://i.imgur.com/MLuqShQ.png)

### Specification
Implementing a genetic algorithm for optimizing a two-dimensional vehicle design on a given terrain. Loosely based on [Boxcar2D](http://www.boxcar2d.com/about.html).

### Introduction
The idea is to randomly generate a pool of vehicle designs which are evaluated against a given terrain using simple physics simulation approach.
A score is assigned to each vehicle depending on how far the vehicle travels without stopping.
The genetic algorithm is divided into three distinct parts, namely selection, crossover and mutation phase.
In the selection step, as the name implies, vehicle designs are selected in a specific manner.
Selecting the n-best designs (those with the highest score) is not always the best solution since diversity is lost.
[Boxcar2D](http://www.boxcar2d.com/about.html) proposes a probabilistic and a deterministic approach.
In the crossover phase, two vehicle designs are fused together to produce offspring.
In order to increase diversity, the mutation phase randomly changes properties of a given offspring.

### Implementation
The above problem doesn't belong to the “classic” parallelization problems.
The challenge in this assignment is to utilize the parallel nature of the GPU wherever possible and also evaluate the boundaries of GPU computing.
Parallelization is achieved on vehicle design level, meaning that multiple vehicle designs are evaluated in parallel.
The physics simulation can be easily parallelized, since individual vehicle designs don’t interact with each other.
Different aspects of the genetic algorithm can also be parallelized using parallel programming primitives. Following features are currently implemented:

- Random generation of vehicle designs using the approach described in [Boxcar2D](http://www.boxcar2d.com/about.html) (The Chromosome). This can be easily parallelized using [RanluxCL](https://bitbucket.org/ivarun/ranluxcl/). 

- Calculating the correct masses for every vehicle design.

- Physics simulation: Only consider active/passive rigid body collisions, meaning only collisions between moving and static objects are considered. Collision detection can be sped up by caching the terrain in local memory. The terrain is represented as a set of points pairs.

- Selection phase:  [Roulette-Wheel Selection](http://www.boxcar2d.com/about.html) works by summing up the score of every vehicle design and calculating the probabilities based on this number. This can be calculated by using the parallel reduction primitive.

- Crossover phase: Given a set of 2n vehicles designs, calculate the crossover of vehicle designs in a tree like fashion, resulting in n vehicle designs in the next generation.
 
- As with Roulette-Wheel Selection, this problem can also be solved by using parallel reduction.

- Mutation phase: Given a mutation rate, simply alter properties of individual vehicle designs.

- Visualization


### Dependancies
 - pyopencl
 - PyQt 4.8 (only for visualization)

Tested on Windows 7 & Windows 8.1 using python 2.7.9, pyopencl-2015.1 and PyQt4-4.11.3

### Roadmap
- full PEP8 compliance
- more sophisticated selection strategies
- improved physics simulation

### Acknowledgments
This project was part of the course "Praktikum General-Purpose Computation on GPUs" at the [ Computer Graphics Group](https://cg.ivd.kit.edu/english/index.php), Karlsruher Institut für Technologie (KIT).

Feedback is greatly appreciated!
