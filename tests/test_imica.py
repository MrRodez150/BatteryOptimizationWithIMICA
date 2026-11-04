import unittest

import numpy as np

from src.optimization.imica import (
    IMICA,
    Individual,
    _dominates,
    _nondominated_reference,
    _survivors,
)
from src.optimization.irbo import IRBO
from src.optimization.indicators import DeltaP, EpsilonPlus, IGDPlus, R2, SMS
from src.optimization.comparison_metrics import (
    evaluate_front,
    hypervolume,
    pooled_feasible_reference,
)
from src.optimization.nsga3 import NSGA3, _reference_directions
from src.optimization.operators import make_offspring, sample_design
from src.optimization.problem import EV_LCO_VARIABLES, EVLCOProblem, Variable


class QuadraticProblem:
    variables = (Variable("x", -2.0, 2.0), Variable("y", -2.0, 2.0))
    objective_names = ("f1", "f2")

    def evaluate(self, design):
        x = design["x"]
        y = design["y"]
        return {
            "objectives": (x * x + y * y, (x - 1.0) ** 2 + (y + 1.0) ** 2),
            "constraints": (abs(x) + abs(y) - 3.0,),
        }


class TestOperators(unittest.TestCase):
    def test_sample_respects_bounds_and_integer_types(self):
        rng = np.random.default_rng(42)
        design = sample_design(EV_LCO_VARIABLES, rng)

        for variable in EV_LCO_VARIABLES:
            self.assertGreaterEqual(design[variable.name], variable.lower)
            self.assertLessEqual(design[variable.name], variable.upper)
        self.assertIsInstance(design["Ns"], int)
        self.assertIsInstance(design["Np"], int)
        self.assertLessEqual(design["efp"] + design["efo"] + design["efn"], 1.0)

    def test_offspring_preserves_domain_and_variable_types(self):
        rng = np.random.default_rng(11)
        parent_a = sample_design(EV_LCO_VARIABLES, rng)
        parent_b = sample_design(EV_LCO_VARIABLES, rng)
        children = make_offspring(parent_a, parent_b, EV_LCO_VARIABLES, rng)

        for child in children:
            for variable in EV_LCO_VARIABLES:
                self.assertGreaterEqual(child[variable.name], variable.lower)
                self.assertLessEqual(child[variable.name], variable.upper)
            self.assertIsInstance(child["Ns"], int)
            self.assertIsInstance(child["Np"], int)


class TestConstraintDominance(unittest.TestCase):
    def _individual(self, objectives, constraints):
        return Individual({}, np.array(objectives), np.array(constraints), {})

    def test_feasible_solution_dominates_infeasible_solution(self):
        feasible = self._individual((100.0, 100.0), (-0.1,))
        infeasible = self._individual((-100.0, -100.0), (0.01,))

        self.assertTrue(_dominates(feasible, infeasible))
        self.assertFalse(_dominates(infeasible, feasible))

    def test_survivors_keep_spread_of_nondominated_front(self):
        population = [
            self._individual((0.0, 1.0), (-1.0,)),
            self._individual((0.5, 0.5), (-1.0,)),
            self._individual((1.0, 0.0), (-1.0,)),
        ]

        survivors = _survivors(population, 2)

        self.assertEqual(len(survivors), 2)
        self.assertTrue(any(item.objectives[0] == 0.0 for item in survivors))
        self.assertTrue(any(item.objectives[1] == 0.0 for item in survivors))


class TestNondominatedReference(unittest.TestCase):
    def _individual(self, design, objectives, constraints):
        return Individual(
            design,
            np.array(objectives, dtype=float),
            np.array(constraints, dtype=float),
            {},
        )

    def test_reference_merges_history_and_removes_dominated_points(self):
        old_reference = [
            self._individual({"id": "a"}, (0.0, 2.0), (-1.0,)),
            self._individual({"id": "dominated"}, (2.0, 2.0), (-1.0,)),
        ]
        new_candidates = [
            self._individual({"id": "b"}, (2.0, 0.0), (-1.0,)),
            self._individual({"id": "worse"}, (3.0, 3.0), (-1.0,)),
        ]

        reference = _nondominated_reference(old_reference, new_candidates, 10)

        self.assertEqual(
            {tuple(item.objectives) for item in reference},
            {(0.0, 2.0), (2.0, 0.0)},
        )

    def test_reference_prefers_feasible_points_when_available(self):
        infeasible = self._individual({"id": "infeasible"}, (0.0, 0.0), (0.1,))
        feasible = self._individual({"id": "feasible"}, (10.0, 10.0), (-0.1,))

        reference = _nondominated_reference([infeasible], [feasible], 10)

        self.assertEqual([item.design["id"] for item in reference], ["feasible"])

    def test_reference_pruning_respects_capacity_and_preserves_extremes(self):
        candidates = [
            self._individual({"id": index}, (float(index), float(10 - index)), (-1.0,))
            for index in range(11)
        ]

        reference = _nondominated_reference([], candidates, 5)

        self.assertEqual(len(reference), 5)
        self.assertTrue(any(item.objectives[0] == 0.0 for item in reference))
        self.assertTrue(any(item.objectives[1] == 0.0 for item in reference))


class TestNSGA3(unittest.TestCase):
    def test_reference_directions_are_simplex_points(self):
        directions = _reference_directions(objective_count=4, minimum_count=40)

        self.assertGreaterEqual(len(directions), 40)
        np.testing.assert_allclose(np.sum(directions, axis=1), 1.0)
        self.assertTrue(np.all(directions >= 0.0))

    def test_run_respects_budget_and_returns_population(self):
        optimizer = NSGA3(QuadraticProblem(), population_size=8, seed=13)

        result = optimizer.run(max_evaluations=19)

        self.assertEqual(result["evaluations"], 19)
        self.assertEqual(optimizer.evaluations, 19)
        self.assertEqual(len(result["population"]), 8)
        self.assertGreater(len(result["archive"]), 0)
        self.assertTrue(all(item.feasible for item in result["archive"]))

    def test_budget_smaller_than_population_is_rejected(self):
        optimizer = NSGA3(QuadraticProblem(), population_size=8)

        with self.assertRaises(ValueError):
            optimizer.run(max_evaluations=7)


class TestComparisonMetrics(unittest.TestCase):
    def test_hypervolume_for_single_point(self):
        result = hypervolume(np.array([[0.5, 0.5]]), np.array([1.0, 1.0]))

        self.assertAlmostEqual(result, 0.25)

    def test_hypervolume_for_overlapping_boxes(self):
        points = np.array([[0.5, 0.8], [0.8, 0.5]])

        result = hypervolume(points, np.array([1.0, 1.0]))

        self.assertAlmostEqual(result, 0.16)

    def test_metrics_use_pooled_feasible_approximation(self):
        first = Individual({}, np.array([0.0, 1.0]), np.array([-1.0]), {})
        second = Individual({}, np.array([1.0, 0.0]), np.array([-1.0]), {})
        reference = pooled_feasible_reference([[first], [second]])

        metrics = evaluate_front([first], [first], reference)

        self.assertEqual(metrics["archive_size"], 1)
        self.assertEqual(metrics["feasible_fraction"], 1.0)
        self.assertIsNotNone(metrics["hypervolume"])
        self.assertIsNotNone(metrics["igd_plus"])
        self.assertIsNotNone(metrics["direction_coverage"])
        self.assertIsNotNone(metrics["spacing_cv"])


class TestEVLCOProblemAdapter(unittest.TestCase):
    def test_energy_is_converted_to_minimization_and_constraints_scaled(self):
        class StubBattery:
            v_pack = 48.0

            def evaluate(self, design):
                if len(design) != len(EV_LCO_VARIABLES):
                    raise AssertionError("Adapter did not pass the full design.")
                return {
                    "objectives": {
                        "specific_energy": 123.0,
                        "sei_growth": 2.0,
                        "temp_increase": 3.0,
                        "price": 4.0,
                    },
                    "constraints": {
                        "upper_violation": 4.8,
                        "lower_violation": -2.4,
                        "volume_fraction_violation": 0.1,
                    },
                }

        problem = EVLCOProblem(StubBattery())
        result = problem.evaluate({variable.name: 1 for variable in EV_LCO_VARIABLES})

        self.assertEqual(result["objectives"], (-123.0, 2.0, 3.0, 4.0))
        np.testing.assert_allclose(
            result["constraints"], (0.1, -0.05, 0.1)
        )


class TestIRBO(unittest.TestCase):
    def test_run_respects_budget_and_returns_nondominated_archive(self):
        optimizer = IRBO(
            QuadraticProblem(),
            population_size=8,
            islands=2,
            migration_interval=2,
            migrants=1,
            seed=7,
        )

        result = optimizer.run(max_evaluations=39)

        self.assertEqual(result["evaluations"], 39)
        self.assertEqual(optimizer.evaluations, 39)
        self.assertEqual([len(population) for population in result["populations"]], [8, 8])
        self.assertGreater(len(result["archive"]), 0)
        self.assertTrue(all(item.feasible for item in result["archive"]))
        self.assertEqual(len(result["references"]), 2)
        self.assertTrue(all(result["references"]))
        for reference in result["references"]:
            self.assertLessEqual(len(reference), optimizer.population_size)
            self.assertTrue(all(item.feasible for item in reference))

    def test_invalid_budget_is_rejected(self):
        optimizer = IRBO(QuadraticProblem(), population_size=8, islands=2)

        with self.assertRaises(ValueError):
            optimizer.run(max_evaluations=15)


class TestIndicators(unittest.TestCase):
    def test_reference_indicators_are_zero_on_their_reference_front(self):
        points = np.array([[0.0, 1.0], [1.0, 0.0]])

        self.assertAlmostEqual(IGDPlus()(points, points), 0.0)
        self.assertAlmostEqual(EpsilonPlus()(points, points), 0.0)
        self.assertAlmostEqual(DeltaP()(points, points), 0.0)

    def test_sms_and_r2_return_finite_scores(self):
        points = np.array([[0.2, 0.8], [0.8, 0.2]])

        self.assertAlmostEqual(SMS()(points), 0.45)
        self.assertTrue(np.isfinite(R2()(points)))


class TestIMICA(unittest.TestCase):
    def test_each_island_evaluates_its_assigned_indicator(self):
        indicators = [SMS(), IGDPlus()]
        optimizer = IMICA(
            QuadraticProblem(),
            population_size=4,
            islands=2,
            indicators=indicators,
            migration_interval=2,
            migrants=1,
            seed=17,
        )

        result = optimizer.run(max_evaluations=19)

        self.assertEqual(result["evaluations"], 19)
        self.assertEqual(result["indicator_names"], ["HV", "IGD+"])
        self.assertEqual(len(result["indicator_values"]), 2)
        self.assertTrue(all(np.isfinite(result["indicator_values"])))
        self.assertEqual(
            {(entry["island"], entry["indicator"]) for entry in result["indicator_history"]},
            {(0, "HV"), (1, "IGD+")},
        )
        self.assertTrue(
            all(
                sum(entry["island"] == island for entry in result["indicator_history"])
                >= 2
                for island in range(2)
            )
        )
        self.assertEqual([len(population) for population in result["populations"]], [4, 4])
        self.assertEqual(len(result["references"]), 2)

    def test_indicator_count_must_match_island_count(self):
        with self.assertRaises(ValueError):
            IMICA(
                QuadraticProblem(),
                islands=2,
                indicators=[SMS()],
            )

    def test_default_indicator_assignment_supports_requested_island_count(self):
        optimizer = IMICA(QuadraticProblem(), islands=3)

        self.assertEqual(
            [indicator.name for indicator in optimizer.indicators],
            ["HV", "R2", "IGD+"],
        )

    def test_roulette_parent_selection_respects_evaluation_budget(self):
        optimizer = IMICA(
            QuadraticProblem(),
            population_size=4,
            islands=2,
            migrants=0,
            select_mode="Roulette",
            seed=23,
        )

        result = optimizer.run(max_evaluations=11)

        self.assertEqual(result["evaluations"], 11)


if __name__ == "__main__":
    unittest.main()
