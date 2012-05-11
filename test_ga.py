from ga import FitnessFunction, GA
import unittest
import random
import numpy as np


class TestCase(unittest.TestCase):
    def test_ga(self):
        """
        Test against a known result. Significant changes to the way GA works will break this test.
        """
        random.seed(0)
        func = FitnessFunction(obj_fun=lambda x: np.sum(np.power(x, 2)),
                               num_vars=4,
                               lb=[-10] * 4,
                               ub=[10] * 4)
        ga = GA(fitness_function=func,
                stall_generations=10)
        fitness, genotype, history = ga.run()
        self.assertAlmostEqual(fitness, 12.09820438854528, places=7)

    def test_save_load(self):
        """
        Test that saving and loading a GA produces the same result we would have gotten by running it normally.
        """
        random.seed(0)
        func = FitnessFunction(obj_fun=lambda x: np.sum(np.power(x, 2)),
                               num_vars=4,
                               lb=[-10] * 4,
                               ub=[10] * 4)
        ga = GA(fitness_function=func,
                stall_generations=10,
                save_state=True)
        for i in range(5):
            ga.step()
        random.seed(0)
        f1, g1, h1 = ga.run()
        random.seed(0)
        loaded_ga = GA.load(func)
        f2, g2, h2 = loaded_ga.run()
        self.assertAlmostEqual(f1, f2)
        for i, x in enumerate(g1):
            self.assertAlmostEqual(x, g2[i])
