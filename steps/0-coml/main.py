# this project is about developing/researching ml for combinatorial optimization and combinatorial optimization for ml
# the idea is to have a program in which it is really easy to declare new combinatorial optimization problems, ml models (including training and validating) and hybrid pipelines and to solve them 
# eg, what if i develop an ml model that learns how to choose best branches to follow in the solution of a branch and bound algorithm? it should be really easy to implement a solution pipeline that uses this ml model to solve the problem and validate algorithm against benchmark data.
# it should also be able to generate synthetic data for the problem and to train the ml model on it.
# it should also have a DBMS to hold problem data, ml model data, hybrid pipeline data, results, logs, reports, etc.
# it should also have an agentic harness framework to allow for the development of agentic interventions on solvers
# what else is there beyond these:
# - a way to declare/use hybrid pipelines to solve the problem
# RL framework
# combinatorial games ?
# competition/rank ?
# we will use the following datasets for testing:
# - the knapsack problem
# - the traveling salesman problem
# - the job scheduling problem
# - the vehicle routing problem
# - the knapsack problem
# - the traveling salesman problem
# - the job scheduling problem
# - the vehicle routing problem

# project folder structure and architecture:
# - main.py: the main file that contains the main function and the project structure
# - problems.py: the file that contains the problem declarations
# - ml.py: the file that contains the ml model declarations
# - hybrid.py: the file that contains the hybrid pipeline declarations
# - data.py: the file that contains the data generation and loading functions
# - utils.py: the file that contains the utility functions
# - config.py: the file that contains the project configuration
# - requirements.txt: the file that contains the project dependencies
# - tests: the folder that contains the test files
# - docs: the folder that contains the documentation
# - data: the folder that contains the data
# - models: the folder that contains the models
# - results: the folder that contains the results
# - logs: the folder that contains the logs
# - reports: the folder that contains the reports
# - notebooks: the folder that contains the notebooks


def main():
    print(
        "2D Euclidean TSP commands:\n"
        "  python -m experiments.generate_euclidean_tsp --help\n"
        "  python -m experiments.train_euclidean_tsp --help\n"
        "  python -m experiments.benchmark_euclidean_tsp --help"
    )

if __name__ == "__main__":
    main()  