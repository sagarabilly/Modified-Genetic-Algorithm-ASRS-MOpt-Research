# Research Publication Progress Record

This GitHub repository is intended to show the progress record for modified genetic algorithm in correspond with the research publication.  

The goal of the research to optimze the movement operation for minimizing time and penalty points that the SBS/RS [Shuttle Based Storage and Retrieval System] machine took to finish the tasks listed in the dataset.  
The dataset could vary according to the assumptions and consideration to limit the scope of the research.  
The research proposed a new mathematical model formulation along with the exact solution method using Gurobi Optimizer as well as metaheuristic solution method using developed modified genetic algorithm.  

![alt text](sbsrs_mechanism.png)

Note: This is still just a code progress, it hasn't been fully implemented in real life as it is open for other reserachers to create a better metaheuristics algorithm.
all written codes belongs to sagara and it is developed in completion for master's graduate programs in Science and Technology Department of Sophia University, Tokyo, Japan.

Feel free to see this as a reference. 

Consist of 3 parts:
1. Base Main mathematical model for exact solution (math_model.py)
2. Modified Genetic Algorithm (mga_sbsrs_checkpoint2.py)
3. Mathematical Model Result Testing and Obj.value Calculation, math model with penalty (testing_with_penalty.py)

For more information regarding the mathematical model and test results, please refer to the published research paper on my Linkedin. IEOM Conference Series: Industrial Engineering and Operation Management
