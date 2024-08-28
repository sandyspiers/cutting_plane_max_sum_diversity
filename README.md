# An Exact Cutting Plane Algorithm for the Maximum Sum Diversity Problem

This code is loosely associated with our paper [1].

It features a cutting plane method used to solve the Euclidean max-sum diversity problem (EMSDP).
The cutting plane method is implemented using CPLEX 22.1 using the _lazy constraint callback_ functionality.

## Usage

Four solvers have been made:

- `ct`: The cut plane solver
- `oa`: Uses convexified outer approximation
- `ct_lm`: A low-memory implementation of `ct`
- `glover`: A glover linearisation of the EMSDP
- `quad`: A quadratic programming formulation of the EMSDP
- `obma`: [A heuristic procedure]()

Each is used as follows:

```bash
ct instance_set instance_file output_file p_ratio timelimit
```

## Building and Results Recreation

To build:

```bash
mkdir build
cd build
cmake ..
make
cd -
```

To then repeat the experiments, we use the handy [bq](https://github.com/sitaramc/notes/blob/master/bq.mkd) tool, so start by making as many workers as required.
Then run

```bash
./sh/bq -w # repeat as many times as you need
./sh/run_GKD
./sh/run_TSP
```

Note that this will look in `data/` for the instance data.  
This is available [here](https://curtin-my.sharepoint.com/:u:/g/personal/283710c_curtin_edu_au/EaW4T0ErZJBNhWEo0jO2FuABi61Cic2hcDxvShj6O_nBmQ).

## References

 1. [Spiers, S., Bui, H. T., & Loxton, R. (2023). An exact cutting plane method for the Euclidean max-sum diversity problem. European Journal of Operational Research, 311(2), 444-454.](https://www.sciencedirect.com/science/article/pii/S037722172300379X)
