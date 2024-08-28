# CPP Code of solution methods for maximum diversity problem

## Structure:

Each .cpp file is its own solution method 
Each solution method is compiled into its own executable.
The ```max_diversity``` header is to help read maximum diversity problem instances.
Each executable is run using 3 commands: (input file, output file, log code).

## Executable Arugments

```solver.exe instance_set instance output_file p_ratio time_limit(s)```

## OBMA_MDP

To run the OBMA algorithm, four parameters are: Instance Name, Dataset Name, Time Limit, Number of Runs.

For example, to solve the instance MDG-a_21_n2000_m200.txt of dataset MDG-a 30 times with the time limit of t = 3600 seconds per run, do the following

./OBMA.exe MDG-a_21_n2000_m200.txt MDG-a 3600 30

Todo:
 1. Add p to arguments
 2. Adjust the read-write procedures
 3. homogoenoize the arguments required