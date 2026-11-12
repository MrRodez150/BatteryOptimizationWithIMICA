import unittest

import numpy as np

from src.core.p2d_evaluator import P2DEvaluator
from src.p2d import (
    ButlerVolmerKinetics,
    ElectrochemicalGrid1D,
    GalvanostaticElectrochemistry,
    ElectrolyteTransport,
    SphericalParticleDiffusion,
    TabulatedOpenCircuitPotential,
    AI2020_SOURCE_COMMIT,
    AI2020_SEI_PARAMETERS,
    build_ai2020_reference_model,
    load_ai2020_reference_ocps,
    electrolyte_conductivity_s_m,
    electrolyte_diffusivity_m2_s,
    reaction_overpotential_v,
    solve_solid_potential,
)


class TestP2DEvaluator(unittest.TestCase):
    def setUp(self):
        self.design = {
            "C": 0.2,
            "lp": 100e-6,
            "lo": 25e-6,
            "ln": 100e-6,
            "Rp": 3e-6,
            "Rn": 5e-6,
            "efp": 0.55,
            "efo": 0.3,
            "efn": 0.55,
            "la": 20e-6,
            "lz": 20e-6,
            "Lh": 80e-3,
            "Rcell": 10e-3,
            "Ns": 14,
            "Np": 2,
        }
        self.evaluator = P2DEvaluator(
            positive_cells=2,
            separator_cells=1,
            negative_cells=2,
            radial_cells=4,
            time_step_s=0.5,
            maximum_steps=2,
            aging_cycles=None,
        )

    def test_design_returns_reproducible_energy_constraints_and_discharge(self):
        first = self.evaluator.evaluate(self.design)
        second = self.evaluator.evaluate(self.design)

        self.assertGreater(first["objectives"]["specific_energy"], 0.0)
        self.assertGreater(first["objectives"]["ohmic_energy_loss_j_m2"], 0.0)
        self.assertGreater(first["objectives"]["price"], 0.0)
        self.assertEqual(first["objectives"], second["objectives"])
        self.assertEqual(first["constraints"], second["constraints"])
        self.assertEqual(first["metrics"], second["metrics"])
        np.testing.assert_array_equal(
            first["trajectory"]["time_s"], second["trajectory"]["time_s"]
        )
        np.testing.assert_array_equal(
            first["trajectory"]["voltage_v"], second["trajectory"]["voltage_v"]
        )
        np.testing.assert_array_equal(
            first["trajectory"]["ohmic_power_loss_w_m2"],
            second["trajectory"]["ohmic_power_loss_w_m2"],
        )
        self.assertEqual(
            first["trajectory"]["time_s"].shape,
            first["trajectory"]["ohmic_power_loss_w_m2"].shape,
        )
        self.assertGreaterEqual(first["metrics"]["ohmic_energy_loss_fraction"], 0.0)
        self.assertLessEqual(first["metrics"]["ohmic_energy_loss_fraction"], 1.0)
        self.assertAlmostEqual(
            first["metrics"]["ohmic_energy_loss_j_m2"],
            first["objectives"]["ohmic_energy_loss_j_m2"],
        )
        for region, expected in (
            ("positive", 0.45),
            ("separator", 0.7),
            ("negative", 0.45),
        ):
            self.assertAlmostEqual(
                first["mapped_parameters"]["electrolyte_porosity"][region],
                expected,
            )
        self.assertEqual(
            first["trajectory"]["time_s"].shape,
            first["trajectory"]["voltage_v"].shape,
        )
        self.assertTrue(
            all(np.isfinite(value) for value in first["constraints"].values())
        )
        self.assertEqual(first["constraints"]["discharge_completion"], 1.0)
        self.assertLessEqual(first["constraints"]["maximum_voltage"], 0.0)
        self.assertLessEqual(
            first["constraints"]["electrolyte_concentration_nonnegative"], 0.0
        )
        self.assertAlmostEqual(
            first["metrics"]["applied_current_a_m2"],
            self.design["C"] * first["metrics"]["theoretical_capacity_ah_m2"],
        )
        self.assertEqual(first["metrics"]["pack_cell_count"], 28)
        self.assertAlmostEqual(
            first["objectives"]["price"],
            sum(first["metrics"]["material_costs_usd"].values()),
        )
        self.assertEqual(
            set(first["metrics"]["material_costs_usd"]),
            {"LCO", "graphite", "separator", "electrolyte", "aluminum", "copper"},
        )
        self.assertEqual(
            set(first["metrics"]["material_masses_kg"]),
            set(first["metrics"]["material_costs_usd"]),
        )

    def test_pack_material_cost_scales_with_series_and_parallel_cells(self):
        baseline = self.evaluator.evaluate(self.design)
        doubled_pack = dict(self.design)
        doubled_pack["Ns"] *= 2
        doubled_pack["Np"] *= 2
        scaled = self.evaluator.evaluate(doubled_pack)

        self.assertAlmostEqual(
            scaled["objectives"]["price"],
            4.0 * baseline["objectives"]["price"],
        )
        for material, baseline_cost in baseline["metrics"][
            "material_costs_usd"
        ].items():
            self.assertAlmostEqual(
                scaled["metrics"]["material_costs_usd"][material],
                4.0 * baseline_cost,
            )

    def test_design_must_include_all_p2d_inputs(self):
        incomplete = dict(self.design)
        del incomplete["C"]

        with self.assertRaisesRegex(ValueError, "missing variables"):
            self.evaluator.evaluate(incomplete)

    def test_default_aging_protocol_uses_selected_voltage_cutoffs(self):
        evaluator = P2DEvaluator()

        self.assertEqual(evaluator.positive_cells, 8)
        self.assertEqual(evaluator.separator_cells, 3)
        self.assertEqual(evaluator.negative_cells, 8)
        self.assertEqual(evaluator.radial_cells, 12)
        self.assertEqual(evaluator.aging_cycles, 20)
        self.assertEqual(evaluator.aging_discharge_cutoff_voltage_v, 3.4)
        self.assertEqual(evaluator.charge_cutoff_voltage_v, 4.0)


class TestElectrochemicalGrid1D(unittest.TestCase):
    def test_regions_and_interfaces_have_expected_geometry(self):
        grid = ElectrochemicalGrid1D(
            positive_thickness_m=90e-6,
            separator_thickness_m=20e-6,
            negative_thickness_m=100e-6,
            positive_cells=3,
            separator_cells=2,
            negative_cells=4,
        )

        self.assertEqual(len(grid.cell_centers_m), 9)
        self.assertAlmostEqual(grid.face_positions_m[-1], 210e-6)
        self.assertEqual(grid.region_slices["positive_electrode"], slice(0, 3))
        self.assertEqual(grid.region_slices["separator"], slice(3, 5))
        self.assertEqual(grid.region_slices["negative_electrode"], slice(5, 9))
        np.testing.assert_allclose(grid.interface_positions_m, (90e-6, 110e-6))

    def test_invalid_grid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            ElectrochemicalGrid1D(0.0, 20e-6, 100e-6)


class TestSphericalParticleDiffusion(unittest.TestCase):
    def setUp(self):
        self.model = SphericalParticleDiffusion(
            radius_m=5e-6,
            radial_cells=12,
            diffusivity_m2_s=1.8e-14,
            activation_energy_j_mol=5000.0,
            maximum_concentration_mol_m3=51554.0,
        )
        self.concentration = np.full(12, 0.5 * 51554.0)

    def test_zero_flux_preserves_uniform_concentration(self):
        updated = self.model.step(self.concentration, time_step_s=0.1)

        np.testing.assert_allclose(updated, self.concentration, rtol=0.0, atol=1e-10)

    def test_no_flux_step_conserves_spherical_inventory(self):
        initial = np.linspace(0.4, 0.6, self.model.radial_cells) * 51554.0

        updated = self.model.step(initial, time_step_s=0.2)

        self.assertAlmostEqual(
            self.model.inventory_mol(updated),
            self.model.inventory_mol(initial),
            delta=1e-14 * self.model.inventory_mol(initial),
        )

    def test_outward_surface_flux_has_exact_inventory_balance(self):
        flux = 1e-7
        time_step = 0.05
        initial_inventory = self.model.inventory_mol(self.concentration)

        updated = self.model.step(
            self.concentration,
            time_step_s=time_step,
            surface_molar_flux_out_mol_m2_s=flux,
        )
        expected_loss = (
            4.0 * np.pi * self.model.radius_m**2 * flux * time_step
        )

        self.assertAlmostEqual(
            initial_inventory - self.model.inventory_mol(updated),
            expected_loss,
            delta=1e-9 * expected_loss,
        )
        self.assertLess(self.model.surface_concentration_mol_m3(updated), 0.5 * 51554.0)

    def test_diffusivity_increases_with_temperature_for_positive_activation_energy(self):
        cold = self.model.diffusivity_at(280.0)
        reference = self.model.diffusivity_at(298.15)
        hot = self.model.diffusivity_at(320.0)

        self.assertLess(cold, reference)
        self.assertGreater(hot, reference)

    def test_current_density_uses_faraday_conversion(self):
        flux = self.model.current_density_to_surface_molar_flux(96485.33212)

        self.assertAlmostEqual(flux, 1.0)


class TestElectrolyteTransport(unittest.TestCase):
    def setUp(self):
        self.grid = ElectrochemicalGrid1D(
            positive_thickness_m=3e-6,
            separator_thickness_m=3e-6,
            negative_thickness_m=3e-6,
            positive_cells=3,
            separator_cells=3,
            negative_cells=3,
        )
        self.model = ElectrolyteTransport(
            grid=self.grid,
            porosity=np.array([0.3] * 3 + [0.4] * 3 + [0.3] * 3),
            bruggeman_exponent=1.5,
        )

    def test_bulk_property_correlations_are_positive_for_reference_state(self):
        diffusivity = electrolyte_diffusivity_m2_s(1000.0, 298.15)
        conductivity = electrolyte_conductivity_s_m(1000.0, 298.15)

        self.assertGreater(float(diffusivity), 0.0)
        self.assertLess(float(diffusivity), 1e-8)
        self.assertGreater(float(conductivity), 0.0)

    def test_conductivity_matches_corrected_valoen_reimers_polynomial(self):
        concentration = 1000.0
        temperature = 298.15
        polynomial = (
            -10.5
            + 0.668e-3 * concentration
            + 0.494e-6 * concentration**2
            + (
                0.074
                - 1.78e-5 * concentration
                - 8.86e-10 * concentration**2
            )
            * temperature
            + (-6.96e-5 + 2.8e-8 * concentration) * temperature**2
        )
        expected = 1e-4 * concentration * polynomial**2

        self.assertAlmostEqual(
            float(electrolyte_conductivity_s_m(concentration, temperature)),
            expected,
            places=14,
        )

    def test_concentration_step_preserves_uniform_state_without_reaction(self):
        initial = np.full(9, 1000.0)

        updated = self.model.advance_concentration(initial, 10.0)

        np.testing.assert_allclose(updated, initial, rtol=0.0, atol=1e-10)

    def test_concentration_step_conserves_porosity_weighted_inventory(self):
        initial = np.linspace(900.0, 1100.0, 9)

        updated = self.model.advance_concentration(initial, 0.1)
        widths = self.grid.cell_widths_m

        self.assertAlmostEqual(
            float(np.dot(self.model.porosity * widths, updated)),
            float(np.dot(self.model.porosity * widths, initial)),
            delta=1e-12 * float(np.dot(self.model.porosity * widths, initial)),
        )

    def test_reaction_source_has_expected_lithium_balance(self):
        initial = np.full(9, 1000.0)
        reaction_current = np.zeros(9)
        reaction_current[0] = 1e5
        time_step = 0.01

        updated = self.model.advance_concentration(
            initial,
            time_step,
            volumetric_reaction_current_a_m3=reaction_current,
        )
        widths = self.grid.cell_widths_m
        initial_inventory = float(
            np.dot(self.model.porosity * widths, initial)
        )
        final_inventory = float(np.dot(self.model.porosity * widths, updated))
        expected_change = (
            (1.0 - self.model.cation_transference_number)
            * time_step
            * float(np.dot(widths, reaction_current))
            / 96485.33212
        )

        self.assertAlmostEqual(
            final_inventory - initial_inventory,
            expected_change,
            delta=1e-10 * abs(expected_change),
        )

    def test_potential_is_constant_at_zero_current_and_uniform_concentration(self):
        state = self.model.solve_potential(
            concentration_mol_m3=np.full(9, 1000.0),
            temperature_k=298.15,
            volumetric_reaction_current_a_m3=0.0,
        )

        np.testing.assert_allclose(state.potential_v, 0.0, atol=1e-12)
        np.testing.assert_allclose(state.ionic_current_faces_a_m2, 0.0, atol=1e-12)

    def test_potential_current_conserves_charge_across_all_interfaces(self):
        reaction_current = np.zeros(9)
        reaction_current[0] = 100.0
        reaction_current[-1] = -100.0

        state = self.model.solve_potential(
            concentration_mol_m3=np.full(9, 1000.0),
            temperature_k=298.15,
            volumetric_reaction_current_a_m3=reaction_current,
        )

        expected_faces = np.concatenate(
            ([0.0], np.cumsum(reaction_current * self.grid.cell_widths_m))
        )
        np.testing.assert_allclose(
            state.ionic_current_faces_a_m2, expected_faces, atol=1e-12
        )
        self.assertTrue(np.all(np.isfinite(state.potential_v)))

    def test_potential_rejects_unbalanced_reaction_current(self):
        with self.assertRaises(ValueError):
            self.model.solve_potential(
                concentration_mol_m3=np.full(9, 1000.0),
                temperature_k=298.15,
                volumetric_reaction_current_a_m3=np.ones(9),
            )


class TestButlerVolmerKinetics(unittest.TestCase):
    def setUp(self):
        self.model = ButlerVolmerKinetics(
            rate_constant_reference=7.898e-12,
            activation_energy_j_mol=5000.0,
            maximum_concentration_mol_m3=51554.0,
        )

    def test_zero_overpotential_gives_zero_reaction_current(self):
        current = self.model.current_density_a_m2(
            0.0, 1000.0, 0.5 * 51554.0, 298.15
        )

        self.assertAlmostEqual(float(current), 0.0, delta=1e-14)

    def test_phase_potentials_use_legacy_overpotential_sign_convention(self):
        overpotential = reaction_overpotential_v(4.0, 1.0, 3.2)

        self.assertAlmostEqual(float(overpotential), -0.2)

    def test_current_is_odd_and_increases_with_positive_overpotential(self):
        positive = self.model.current_density_a_m2(
            0.02, 1000.0, 0.5 * 51554.0, 298.15
        )
        negative = self.model.current_density_a_m2(
            -0.02, 1000.0, 0.5 * 51554.0, 298.15
        )

        self.assertGreater(float(positive), 0.0)
        self.assertAlmostEqual(float(positive), -float(negative), places=12)

    def test_current_density_is_faraday_times_molar_flux(self):
        state = (0.02, 1000.0, 0.45 * 51554.0, 298.15)
        molar_flux = self.model.molar_flux_mol_m2_s(*state)
        current_density = self.model.current_density_a_m2(*state)

        self.assertAlmostEqual(
            float(current_density),
            96485.33212 * float(molar_flux),
            places=12,
        )

    def test_specific_area_converts_interfacial_to_volumetric_current(self):
        volumetric = self.model.volumetric_reaction_current_a_m3(2e5, 0.3)

        self.assertAlmostEqual(float(volumetric), 6e4)

    def test_symmetric_butler_volmer_current_and_overpotential_are_inverses(self):
        concentrations = (1000.0, 0.45 * 51554.0, 298.15)
        requested_current = 0.2
        overpotential = self.model.overpotential_v(
            requested_current, *concentrations
        )
        current = self.model.current_density_a_m2(
            overpotential, *concentrations
        )

        self.assertAlmostEqual(float(current), requested_current, places=12)

    def test_rate_constant_increases_with_temperature(self):
        cold = self.model.rate_constant_at(280.0)
        hot = self.model.rate_constant_at(320.0)

        self.assertLess(float(cold), self.model.rate_constant_reference)
        self.assertGreater(float(hot), self.model.rate_constant_reference)


class TestSolidPotential(unittest.TestCase):
    def test_zero_reaction_and_zero_boundary_current_give_constant_potential(self):
        state = solve_solid_potential(
            cell_widths_m=np.full(4, 10e-6),
            effective_conductivity_s_m=np.full(4, 10.0),
            volumetric_reaction_current_a_m3=np.zeros(4),
            left_boundary_current_a_m2=0.0,
            right_boundary_current_a_m2=0.0,
            left_boundary_potential_v=3.7,
        )

        np.testing.assert_allclose(state.potential_v, 3.7, atol=1e-14)
        np.testing.assert_allclose(
            state.electronic_current_faces_a_m2, 0.0, atol=1e-14
        )

    def test_reaction_current_is_conserved_and_changes_solid_current(self):
        widths = np.full(3, 10e-6)
        reaction = np.full(3, 2e5)
        integrated_reaction = float(np.dot(widths, reaction))

        state = solve_solid_potential(
            cell_widths_m=widths,
            effective_conductivity_s_m=np.full(3, 10.0),
            volumetric_reaction_current_a_m3=reaction,
            left_boundary_current_a_m2=integrated_reaction,
            right_boundary_current_a_m2=0.0,
        )

        np.testing.assert_allclose(
            state.electronic_current_faces_a_m2,
            integrated_reaction - np.concatenate(([0.0], np.cumsum(widths * reaction))),
            atol=1e-12,
        )
        self.assertTrue(np.all(np.isfinite(state.potential_v)))
        self.assertGreater(state.potential_v[0], state.potential_v[-1])

    def test_rejects_incompatible_boundary_currents(self):
        with self.assertRaises(ValueError):
            solve_solid_potential(
                cell_widths_m=np.full(2, 10e-6),
                effective_conductivity_s_m=np.full(2, 10.0),
                volumetric_reaction_current_a_m3=np.zeros(2),
                left_boundary_current_a_m2=1.0,
                right_boundary_current_a_m2=0.0,
            )

    def test_rejects_nonpositive_effective_conductivity(self):
        with self.assertRaises(ValueError):
            solve_solid_potential(
                cell_widths_m=np.full(2, 10e-6),
                effective_conductivity_s_m=np.array([10.0, 0.0]),
                volumetric_reaction_current_a_m3=np.zeros(2),
                left_boundary_current_a_m2=0.0,
                right_boundary_current_a_m2=0.0,
            )


class TestTabulatedOpenCircuitPotential(unittest.TestCase):
    def setUp(self):
        self.ocp = TabulatedOpenCircuitPotential(
            stoichiometry=np.array([0.1, 0.5, 0.9]),
            voltage_v=np.array([4.3, 4.0, 3.6]),
            source="Test fixture only; replace with a cited LCO parameter source",
            entropy_coefficient_v_k=np.array([1e-4, 2e-4, 3e-4]),
        )

    def test_interpolates_voltage_and_entropy_temperature_correction(self):
        voltage = self.ocp.voltage_at(0.3, temperature_k=308.15)

        self.assertAlmostEqual(float(voltage), 4.1515)

    def test_rejects_stoichiometry_outside_table_without_extrapolation(self):
        with self.assertRaises(ValueError):
            self.ocp.voltage_at(0.05)

    def test_rejects_missing_source(self):
        with self.assertRaises(ValueError):
            TabulatedOpenCircuitPotential(
                stoichiometry=np.array([0.1, 0.9]),
                voltage_v=np.array([4.3, 3.6]),
                source="",
            )

    def test_rejects_temperature_request_without_entropy_data(self):
        ocp = TabulatedOpenCircuitPotential(
            stoichiometry=np.array([0.1, 0.9]),
            voltage_v=np.array([4.3, 3.6]),
            source="Cited parameter source",
        )

        with self.assertRaises(ValueError):
            ocp.voltage_at(0.5, temperature_k=300.0)


class TestAi2020ReferenceParameters(unittest.TestCase):
    def setUp(self):
        self.grid = ElectrochemicalGrid1D(
            positive_thickness_m=68e-6,
            separator_thickness_m=25e-6,
            negative_thickness_m=76.5e-6,
            positive_cells=2,
            separator_cells=1,
            negative_cells=2,
        )
        self.reference = build_ai2020_reference_model(self.grid, radial_cells=8)

    def test_reference_ocps_have_source_and_documented_domains(self):
        lco, graphite = load_ai2020_reference_ocps()

        self.assertIn(AI2020_SOURCE_COMMIT, lco.source)
        self.assertIn(AI2020_SOURCE_COMMIT, graphite.source)
        self.assertAlmostEqual(float(lco.stoichiometry[0]), 0.43)
        self.assertLessEqual(float(lco.stoichiometry[-1]), 0.9936)
        self.assertGreater(float(graphite.stoichiometry[0]), 0.0)
        self.assertLess(float(graphite.stoichiometry[-1]), 1.0)
        self.assertGreater(float(lco.voltage_at(0.6)), 3.5)
        self.assertLess(float(graphite.voltage_at(0.5)), 0.3)

    def test_reference_parameters_build_a_finite_ocv_state(self):
        positive, negative = self.reference.initial_particle_states(0.6, 0.5)
        state = self.reference.electrochemistry.solve(
            applied_current_a_m2=0.0,
            electrolyte_concentration_mol_m3=np.full(5, 1000.0),
            positive_surface_concentration_mol_m3=positive[:, -1],
            negative_surface_concentration_mol_m3=negative[:, -1],
        )

        self.assertGreater(state.cell_voltage_v, 3.0)
        self.assertLess(state.cell_voltage_v, 4.5)
        self.assertLess(state.nonlinear_residual_norm, 1e-7)

    def test_reference_model_can_advance_a_low_current_step(self):
        positive, negative = self.reference.initial_particle_states(0.6, 0.5)
        trajectory = self.reference.electrochemistry.simulate_constant_current(
            applied_current_a_m2=5.0,
            electrolyte_concentration_initial_mol_m3=np.full(
                len(self.grid.cell_widths_m),
                self.reference.initial_electrolyte_concentration_mol_m3,
            ),
            positive_particle_concentration_initial_mol_m3=positive,
            negative_particle_concentration_initial_mol_m3=negative,
            positive_particles=self.reference.positive_particles,
            negative_particles=self.reference.negative_particles,
            time_step_s=0.1,
            maximum_steps=1,
            cutoff_voltage_v=2.5,
        )

        self.assertEqual(len(trajectory.time_s), 2)
        self.assertTrue(np.all(np.isfinite(trajectory.voltage_v)))
        self.assertTrue(np.all(trajectory.voltage_v > 2.5))

    def test_reference_solver_converges_at_300_a_per_m2(self):
        positive, negative = self.reference.initial_particle_states(0.6, 0.5)
        state = self.reference.electrochemistry.solve(
            applied_current_a_m2=300.0,
            electrolyte_concentration_mol_m3=np.full(5, 1000.0),
            positive_surface_concentration_mol_m3=positive[:, -1],
            negative_surface_concentration_mol_m3=negative[:, -1],
        )

        self.assertGreater(state.cell_voltage_v, 0.0)
        self.assertLess(state.nonlinear_residual_norm, 1e-7)
        self.assertLess(state.function_evaluations, 100)

    def test_reference_discharge_voltage_converges_with_time_step_refinement(self):
        positive, negative = self.reference.initial_particle_states(0.6, 0.5)
        voltages = []
        for time_step in (1.0, 0.5, 0.25):
            trajectory = self.reference.electrochemistry.simulate_constant_current(
                applied_current_a_m2=100.0,
                electrolyte_concentration_initial_mol_m3=np.full(
                    len(self.grid.cell_widths_m),
                    self.reference.initial_electrolyte_concentration_mol_m3,
                ),
                positive_particle_concentration_initial_mol_m3=positive,
                negative_particle_concentration_initial_mol_m3=negative,
                positive_particles=self.reference.positive_particles,
                negative_particles=self.reference.negative_particles,
                time_step_s=time_step,
                maximum_steps=round(10.0 / time_step),
                cutoff_voltage_v=2.5,
            )
            self.assertFalse(trajectory.reached_cutoff)
            self.assertAlmostEqual(float(trajectory.time_s[-1]), 10.0)
            voltages.append(float(trajectory.voltage_v[-1]))

        coarse_error = abs(voltages[0] - voltages[1])
        refined_error = abs(voltages[1] - voltages[2])
        self.assertGreater(coarse_error, refined_error)


class TestGalvanostaticElectrochemistry(unittest.TestCase):
    def setUp(self):
        self.grid = ElectrochemicalGrid1D(
            positive_thickness_m=60e-6,
            separator_thickness_m=20e-6,
            negative_thickness_m=60e-6,
            positive_cells=2,
            separator_cells=1,
            negative_cells=2,
        )
        self.transport = ElectrolyteTransport(
            grid=self.grid,
            porosity=np.array([0.3, 0.3, 0.4, 0.3, 0.3]),
            bruggeman_exponent=1.5,
        )
        self.model = GalvanostaticElectrochemistry(
            electrolyte=self.transport,
            positive_effective_conductivity_s_m=np.full(2, 10.0),
            negative_effective_conductivity_s_m=np.full(2, 10.0),
            positive_interfacial_area_m_inv=np.full(2, 1e5),
            negative_interfacial_area_m_inv=np.full(2, 1e5),
            positive_kinetics=ButlerVolmerKinetics(
                rate_constant_reference=7.898e-12,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=51554.0,
            ),
            negative_kinetics=ButlerVolmerKinetics(
                rate_constant_reference=5.031e-11,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=30555.0,
            ),
            positive_ocp=TabulatedOpenCircuitPotential(
                stoichiometry=np.array([0.1, 0.5, 0.9]),
                voltage_v=np.array([4.3, 4.0, 3.6]),
                source="synthetic test fixture only",
            ),
            negative_ocp=TabulatedOpenCircuitPotential(
                stoichiometry=np.array([0.1, 0.5, 0.9]),
                voltage_v=np.array([0.3, 0.15, 0.08]),
                source="synthetic test fixture only",
            ),
        )

    def solve(self, applied_current):
        return self.model.solve(
            applied_current_a_m2=applied_current,
            electrolyte_concentration_mol_m3=np.full(5, 1000.0),
            positive_surface_concentration_mol_m3=np.full(2, 0.5 * 51554.0),
            negative_surface_concentration_mol_m3=np.full(2, 0.5 * 30555.0),
            temperature_k=298.15,
        )

    def test_zero_applied_current_recovers_open_circuit_voltage(self):
        state = self.solve(0.0)

        self.assertAlmostEqual(state.cell_voltage_v, 4.0 - 0.15, places=8)
        np.testing.assert_allclose(
            state.positive_reaction_current_a_m3, 0.0, atol=1e-8
        )
        np.testing.assert_allclose(
            state.negative_reaction_current_a_m3, 0.0, atol=1e-8
        )

    def test_discharge_closes_electronic_and_ionic_current_balances(self):
        applied_current = 1.0
        state = self.solve(applied_current)
        widths = self.grid.cell_widths_m

        self.assertAlmostEqual(
            float(np.dot(widths[:2], state.positive_reaction_current_a_m3)),
            -applied_current,
            places=8,
        )
        self.assertAlmostEqual(
            float(np.dot(widths[-2:], state.negative_reaction_current_a_m3)),
            applied_current,
            places=8,
        )
        self.assertAlmostEqual(
            state.electrolyte_current_faces_a_m2[0], 0.0, places=8
        )
        self.assertAlmostEqual(
            state.electrolyte_current_faces_a_m2[-1], 0.0, places=8
        )
        self.assertAlmostEqual(
            state.positive_solid_current_faces_a_m2[-1], 0.0, places=8
        )
        self.assertAlmostEqual(
            state.negative_solid_current_faces_a_m2[0], 0.0, places=8
        )
        self.assertGreater(state.cell_voltage_v, 0.0)
        self.assertLess(state.nonlinear_residual_norm, 1e-7)

    def test_charge_reverses_reaction_current_and_increases_voltage(self):
        discharge = self.solve(1.0)
        charge = self.solve(-1.0)
        widths = self.grid.cell_widths_m

        self.assertAlmostEqual(
            float(np.dot(widths[:2], charge.positive_reaction_current_a_m3)),
            1.0,
            places=8,
        )
        self.assertAlmostEqual(
            float(np.dot(widths[-2:], charge.negative_reaction_current_a_m3)),
            -1.0,
            places=8,
        )
        self.assertGreater(charge.cell_voltage_v, discharge.cell_voltage_v)
        self.assertLess(charge.nonlinear_residual_norm, 1e-7)

    def test_sei_side_reaction_closes_total_negative_current_balance(self):
        state = self.model.solve(
            applied_current_a_m2=1.0,
            electrolyte_concentration_mol_m3=np.full(5, 1000.0),
            positive_surface_concentration_mol_m3=np.full(2, 0.5 * 51554.0),
            negative_surface_concentration_mol_m3=np.full(2, 0.5 * 30555.0),
            temperature_k=298.15,
            sei_thickness_m=np.full(2, AI2020_SEI_PARAMETERS.initial_thickness_m),
            sei_parameters=AI2020_SEI_PARAMETERS,
        )
        negative_total_reaction = (
            state.negative_reaction_current_a_m3
            + state.negative_sei_reaction_current_a_m3
        )

        self.assertTrue(np.all(state.negative_sei_reaction_current_a_m3 < 0.0))
        self.assertAlmostEqual(
            float(np.dot(self.grid.cell_widths_m[-2:], negative_total_reaction)),
            1.0,
            places=8,
        )
        self.assertLess(state.nonlinear_residual_norm, 1e-7)

    def test_sei_thickness_and_lithium_loss_increase_during_simulation(self):
        particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=1.8e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=51554.0,
            )
            for _ in range(2)
        )
        negative_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=3.9e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=30555.0,
            )
            for _ in range(2)
        )
        trajectory = self.model.simulate_constant_current(
            applied_current_a_m2=1.0,
            electrolyte_concentration_initial_mol_m3=np.full(5, 1000.0),
            positive_particle_concentration_initial_mol_m3=np.full(
                (2, 8), 0.5 * 51554.0
            ),
            negative_particle_concentration_initial_mol_m3=np.full(
                (2, 8), 0.5 * 30555.0
            ),
            positive_particles=particles,
            negative_particles=negative_particles,
            time_step_s=0.1,
            maximum_steps=1,
            sei_parameters=AI2020_SEI_PARAMETERS,
        )

        self.assertTrue(
            np.all(
                trajectory.final_sei_thickness_m
                > AI2020_SEI_PARAMETERS.initial_thickness_m
            )
        )
        self.assertGreater(trajectory.lithium_loss_to_sei_mol_m2, 0.0)

    def test_voltage_cutoff_time_is_interpolated_within_the_last_step(self):
        positive_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=1.8e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=51554.0,
            )
            for _ in range(2)
        )
        negative_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=3.9e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=30555.0,
            )
            for _ in range(2)
        )
        electrolyte_initial = np.full(5, 1000.0)
        positive_initial = np.full((2, 8), 0.5 * 51554.0)
        negative_initial = np.full((2, 8), 0.5 * 30555.0)
        initial_state = self.solve(1.0)
        advanced = self.model.advance_species(
            electrochemical_state=initial_state,
            electrolyte_concentration_old_mol_m3=electrolyte_initial,
            positive_particle_concentration_old_mol_m3=positive_initial,
            negative_particle_concentration_old_mol_m3=negative_initial,
            positive_particles=positive_particles,
            negative_particles=negative_particles,
            time_step_s=0.1,
        )
        positive_surface = np.asarray(
            [
                particle.surface_concentration_mol_m3(profile)
                for particle, profile in zip(
                    positive_particles,
                    advanced.positive_particle_concentration_mol_m3,
                )
            ]
        )
        negative_surface = np.asarray(
            [
                particle.surface_concentration_mol_m3(profile)
                for particle, profile in zip(
                    negative_particles,
                    advanced.negative_particle_concentration_mol_m3,
                )
            ]
        )
        next_state = self.model.solve(
            applied_current_a_m2=1.0,
            electrolyte_concentration_mol_m3=(
                advanced.electrolyte_concentration_mol_m3
            ),
            positive_surface_concentration_mol_m3=positive_surface,
            negative_surface_concentration_mol_m3=negative_surface,
        )
        cutoff = 0.5 * (initial_state.cell_voltage_v + next_state.cell_voltage_v)
        trajectory = self.model.simulate_constant_current(
            applied_current_a_m2=1.0,
            electrolyte_concentration_initial_mol_m3=electrolyte_initial,
            positive_particle_concentration_initial_mol_m3=positive_initial,
            negative_particle_concentration_initial_mol_m3=negative_initial,
            positive_particles=positive_particles,
            negative_particles=negative_particles,
            time_step_s=0.1,
            maximum_steps=1,
            cutoff_voltage_v=cutoff,
        )

        self.assertTrue(trajectory.reached_cutoff)
        self.assertAlmostEqual(trajectory.voltage_v[-1], cutoff)
        self.assertGreater(trajectory.time_s[-1], 0.0)
        self.assertLess(trajectory.time_s[-1], 0.1)

    def test_split_species_advance_updates_particles_and_electrolyte(self):
        state = self.solve(1.0)
        positive_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=1.8e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=51554.0,
            )
            for _ in range(2)
        )
        negative_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=3.9e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=30555.0,
            )
            for _ in range(2)
        )
        positive_old = np.full((2, 8), 0.5 * 51554.0)
        negative_old = np.full((2, 8), 0.5 * 30555.0)

        updated = self.model.advance_species(
            electrochemical_state=state,
            electrolyte_concentration_old_mol_m3=np.full(5, 1000.0),
            positive_particle_concentration_old_mol_m3=positive_old,
            negative_particle_concentration_old_mol_m3=negative_old,
            positive_particles=positive_particles,
            negative_particles=negative_particles,
            time_step_s=0.1,
        )

        self.assertTrue(np.all(np.isfinite(updated.electrolyte_concentration_mol_m3)))
        self.assertGreater(
            float(np.mean(updated.positive_particle_concentration_mol_m3)),
            float(np.mean(positive_old)),
        )
        self.assertLess(
            float(np.mean(updated.negative_particle_concentration_mol_m3)),
            float(np.mean(negative_old)),
        )

    def test_constant_current_simulation_honors_step_budget_and_cutoff(self):
        positive_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=1.8e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=51554.0,
            )
            for _ in range(2)
        )
        negative_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=3.9e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=30555.0,
            )
            for _ in range(2)
        )
        trajectory = self.model.simulate_constant_current(
            applied_current_a_m2=1.0,
            electrolyte_concentration_initial_mol_m3=np.full(5, 1000.0),
            positive_particle_concentration_initial_mol_m3=np.full(
                (2, 8), 0.5 * 51554.0
            ),
            negative_particle_concentration_initial_mol_m3=np.full(
                (2, 8), 0.5 * 30555.0
            ),
            positive_particles=positive_particles,
            negative_particles=negative_particles,
            time_step_s=0.1,
            maximum_steps=2,
            cutoff_voltage_v=2.5,
        )

        self.assertEqual(len(trajectory.time_s), 3)
        self.assertEqual(len(trajectory.voltage_v), 3)
        np.testing.assert_allclose(trajectory.time_s, [0.0, 0.1, 0.2])
        self.assertFalse(trajectory.reached_cutoff)
        self.assertTrue(np.all(trajectory.voltage_v > 2.5))

    def test_constant_current_simulation_stops_at_initial_cutoff(self):
        positive_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=1.8e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=51554.0,
            )
            for _ in range(2)
        )
        negative_particles = tuple(
            SphericalParticleDiffusion(
                radius_m=5e-6,
                radial_cells=8,
                diffusivity_m2_s=3.9e-14,
                activation_energy_j_mol=5000.0,
                maximum_concentration_mol_m3=30555.0,
            )
            for _ in range(2)
        )
        trajectory = self.model.simulate_constant_current(
            applied_current_a_m2=1.0,
            electrolyte_concentration_initial_mol_m3=np.full(5, 1000.0),
            positive_particle_concentration_initial_mol_m3=np.full(
                (2, 8), 0.5 * 51554.0
            ),
            negative_particle_concentration_initial_mol_m3=np.full(
                (2, 8), 0.5 * 30555.0
            ),
            positive_particles=positive_particles,
            negative_particles=negative_particles,
            time_step_s=0.1,
            maximum_steps=2,
            cutoff_voltage_v=4.0,
        )

        self.assertEqual(len(trajectory.time_s), 1)
        self.assertEqual(len(trajectory.voltage_v), 1)
        self.assertTrue(trajectory.reached_cutoff)


if __name__ == "__main__":
    unittest.main()
