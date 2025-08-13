# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:09:12 2025

@author: u226422
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import funcs_multi
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator.mutation import PolynomialMutation
from jmetal.core.problem import FloatProblem
from jmetal.util.evaluator import Evaluator
from jmetal.util.archive import CrowdingDistanceArchive
from concurrent.futures import ThreadPoolExecutor
from jmetal.core.solution import FloatSolution
from jmetal.core.quality_indicator import HyperVolume
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.core.observer import Observer
from jmetal.util.observer import ProgressBarObserver


class HVObserver(Observer):
    def __init__(self, reference_point, interval=1):
        self.reference_point = reference_point
        self.interval = interval
        self.hv = HyperVolume(reference_point=reference_point)
        self.history = []
        self.counter = 0

    def update(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.interval != 0:
            return
        front = get_non_dominated_solutions(kwargs['SOLUTIONS'])
        value = self.hv.compute([s.objectives for s in front])
        self.history.append((self.counter, value))

class MyProblem(FloatProblem):
    def __init__(self, **kwargs):
        super(MyProblem, self).__init__()
        self.lower_bound = [0.16, 0.16, 0.16, 0.16, 0.0006, 0.0006, 0.0006, 0.0006,
                            500, 500, 500, 500, 0.0144, 0.0144, 0.0144, 0.0144,
                            9, 9, 9, 9, 0.50, 0.065, 0.94,
                            0.008, 0.008, 0.008, 0.008, 0.72, 0,
                            1.26, 1.26, 1.26, 1.26, 0.01,
                            20, 20, 20, 20,
                            0.0, 0.0, 0.0, 0.0, 0.3,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.4, 0.005, 1, 1]
        self.upper_bound = [0.5, 0.5, 0.5, 0.5, 0.002, 0.002, 0.002, 0.002,
                            3500, 3500, 3500, 3500, 0.0216, 0.0216, 0.0216, 0.0216,
                            18, 18, 18, 18, 0.57, 0.075, 0.99,
                            0.02, 0.02, 0.02, 0.02, 0.95, 5,
                            1.54, 1.54, 1.54, 1.54, 0.05,
                            50, 50, 50, 50,
                            1, 1, 1, 1, 0.5,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            0.7, 0.05, 2.5, 1.4]
        self.n_var = 55
        self.n_obj = 3  #
        self.names = ['alfa_1', 'alfa_2', 'alfa_3', 'alfa_4', 'Xn_1', 'Xn_2', 'Xn_3', 'Xn_4',
                      'Do_1', 'Do_2', 'Do_3', 'Do_4', 'SpLAI_1', 'SpLAI_2', 'SpLAI_3', 'SpLAI_4',
                      'm_1', 'm_2', 'm_3', 'm_4', 'A_brunt', 'B_brunt', 'epsilon_leaf',
                      'b_1', 'b_2', 'b_3', 'b_4', 'sigma_NIR', 'PenPar2',
                      'NNI_crit_1', 'NNI_crit_2', 'NNI_crit_3', 'NNI_crit_4', 'K_sat_1',
                      'ShldResC_1', 'ShldResC_2', 'ShldResC_3', 'ShldResC_4',
                      'delta_1', 'delta_2', 'delta_3', 'delta_4', 'k_net',
                      'SOrgPhotEff_1', 'SOrgPhotEff_2', 'SOrgPhotEff_3', 'SOrgPhotEff_4',
                      'StemPhotEff_1', 'StemPhotEff_2', 'StemPhotEff_3', 'StemPhotEff_4',
                      'EpFactor', 'K_aquitard', 'Z_aquitard', 'EpFac']
        # Getting measurements
        self.observations = funcs_multi.measured_data()

    def number_of_variables(self):
        return self.n_var

    def number_of_objectives(self):
        return self.n_obj

    def number_of_constraints(self):
        return 0
    
    def name(self):
        return "Daisy"

    def evaluate(self, sol: FloatSolution, idx: int):
        # Getting model ouputs when there are measurements and rRMSE computation
        return funcs_multi.running_daisy(param_names=self.names, crops=self.observations,
                                         x=sol.variables, thread_id=idx)

class ParallelEvaluator(Evaluator):
    def __init__(self, n_workers=23):
        self.n_workers = n_workers

    def evaluate(self, solutions: list[FloatSolution], problem):
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Assign an index (folder ID) to each solution before evaluating
            tasks = [(idx, problem, sol) for idx, sol in enumerate(solutions)]
            results = list(executor.map(self.evaluate_solution, tasks))
        return results

    def evaluate_solution(self, task):
        idx, problem, sol = task
        # Assign each solution to its corresponding `setup_idx` folder
        sol.objectives = problem.evaluate(sol, idx)
        return sol


if __name__ == "__main__":
    problem = MyProblem()

    algorithm = SMPSO(problem=problem,
                      swarm_size=200,
                      leaders=CrowdingDistanceArchive(200),
                      mutation=PolynomialMutation(probability=0.025, distribution_index=20),
                      termination_criterion=StoppingByEvaluations(max_evaluations=50000),
                      swarm_evaluator=ParallelEvaluator())
    hv_observer = HVObserver(reference_point=[1, 1, 1], interval=10)
    pb_observer = ProgressBarObserver(max=50000)
    algorithm.observable.register(observer=hv_observer)
    algorithm.observable.register(observer=pb_observer)

    algorithm.run()
    solution = algorithm.result()
    hv = HyperVolume(reference_point=[1, 1, 1])
    front = get_non_dominated_solutions(solution)
    outX = np.array([sol.variables for sol in front])
    outF = np.array([sol.objectives for sol in front])
    pd.DataFrame(outX).to_csv("outX_SMPSO_alloctest.csv")  #
    pd.DataFrame(outF).to_csv("outF_SMPSO_alloctest.csv")  #
    value = hv.compute([solution[i].objectives for i in range(len(solution))])

    evals, hv_values = zip(*hv_observer.history)
    fig, ax = plt.subplots()
    ax.plot(evals, hv_values)
    convergence = pd.DataFrame(data=[evals, hv_values])
    convergence.to_csv('convergencetest.csv')
