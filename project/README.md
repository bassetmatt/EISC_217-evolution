# Project - Evolution of agents

The 2022 project will focus on the evolution of agents using the <a
href="https://evolutiongym.github.io/">Evolution Gym</a> suite. To get started
with evogym, see the [neuroevolution notebook](https://github.com/d9w/evolution/blob/master/neuroevolution/evogym.ipynb)
([Colab
version](https://colab.research.google.com/github/d9w/evolution/blob/master/neuroevolution/evogym.ipynb)).

You will need to evolve movement policies for three tasks independently:

+ Walker-v0 (easy)
+ Thrower-v0 (medium)
+ Climb-v2 (hard) 

You have a budget of 10000 **evaluations** for evolution (for example, a
population of 10 for 1000 generations).  Each evaluation can have 500 maximum
steps, but you are encouraged to reduce this and the total number of
evaluations while making algorithm decisions to have faster results.  You
choose the evolutionary algorithm, gene representation, and evolutionary
hyperparameters, but you must demonstrate that you only used the allocated
evaluation budget. The goal is to obtain the best score independently on each
task. Scores should be shown in your final presentation as the average best
score over **at least 2 independent evolutions**.

For each task, points will be allocated to teams in the following
manner:

+ 1st place: 5 points
+ 2nd place: 4 points
+ 3rd place: 3 points
+ 4th place: 2 points
+ 5th place: 1 point

Project presentations will take place on Thursday, May 5th.

Alternatively, students interested in pursuing co-evolution of robot morphology
and movement policy are encouraged to do so. The tasks remain the same but the
morphology is not fixed. The total score will not be compared with the fixed
morphologies, rather the total evaluation will be based on how the coevolution.

You can use the code provided during class for your evolutionary algorithms, and you can also use any code online. Some popular libraries are:

+ [cmaes](https://github.com/CyberAgentAILab/cmaes)
+ [pycma](https://github.com/CMA-ES/pycma)
+ [pymoo](https://pymoo.org/)
+ [DEAP](https://github.com/DEAP/deap)
