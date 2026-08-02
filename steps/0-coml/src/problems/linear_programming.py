# this file contains a wrapper for linear programming problem declarations
# besides the normal pyomo and other solver's models, we will also suppport our own problem/model framework, which we should call: coml_model


import pyomo.environ as pyomo

class LinearProgrammingProblem:
    def __init__(self, name, variables, constraints, objective):
        self.name = name
        self.variables = variables
        self.constraints = constraints
        self.objective = objective
        self.solution = None

    def create_pyomo_model(self):
        model = pyomo.ConcreteModel()
        model.x = pyomo.Var(self.variables, bounds=(0, None))
        model.obj = pyomo.Objective(expr=self.objective)
        model.con = pyomo.Constraint(expr=self.constraints)
        return model

    def create_coml_model(self):
        pass


    # def pyomo_solve(self, solver_name='glpk'):
    #     model = pyomo.ConcreteModel()
    #     model.x = pyomo.Var(bounds=(0, None))

    #     model.obj = pyomo.Objective(expr=self.objective)
    #     model.con = pyomo.Constraint(expr=self.constraints)

    #     solver = pyomo.SolverFactory(solver_name)
    #     solver.solve(model)

    #     self.solution = model.solution
    #     return self.solution


    # def get_solution(self):
    #     return self.solution.x.value

    # def get_objective_value(self):
    #     return self.solution.obj.value

    