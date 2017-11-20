# DESPOT Solver

## Basic Implementation
This is actually an offline DESPOT solver for autonomous driving simulator. It loads a json file of the future positions of the agent cars, take the observation of every step, and do the planning in a "fake online" manner.

## How to use the Solver
To use the solver, please first change the hyperparameters specified in *despot_config.json*, then 

```sh
$ cd despot
$ sh run_despot.sh
```

Then a *cmds.json* file will be generated and moved automatically to *../ros-lanechanging/autocar/scripts/* folder.