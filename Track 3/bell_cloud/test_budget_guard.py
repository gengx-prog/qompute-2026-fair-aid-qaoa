"""Budget guards and an offline IBM client smoke test; never use cloud access."""
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bell_core import analyze_counts, build_circuits
from run_bell import (
    collect_job_usage, estimate_hardware_usage, parse_args,
    run_hardware, submit_bounded_job, validate_hardware_budget,
)


class BudgetGuardTests(unittest.TestCase):
    def test_cli_defaults_and_limits(self):
        args = parse_args(["hardware"])
        self.assertEqual((args.shots, args.max_execution_time), (2048, 120))
        for option, values in (
            ("--shots", ("0", "-1", "4097")),
            ("--max-execution-time", ("0", "-1", "121", "600", "1.5")),
        ):
            for value in values:
                with self.subTest(option=option, value=value):
                    with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                        parse_args(["hardware", option, value])
        self.assertEqual(parse_args(["hardware", "--shots", "4096"]).shots, 4096)
        self.assertEqual(parse_args(["local", "--shots", "8192"]).shots, 8192)

    def test_invalid_budget_is_rejected_before_account_lookup(self):
        args = SimpleNamespace(shots=4097, max_execution_time=120)
        with patch("run_bell.connect") as connect, self.assertRaises(ValueError):
            run_hardware(args)
        connect.assert_not_called()

    def test_exactly_six_circuits_and_integer_parameters_required(self):
        for shots, cap, count in ((2048, 120, 5), (2048, 120, 7),
                                  (2048.0, 120, 6), (2048, True, 6)):
            with self.subTest(shots=shots, cap=cap, count=count), self.assertRaises(ValueError):
                validate_hardware_budget(shots, cap, count)

    def test_cap_is_set_before_exactly_one_run(self):
        sampler = Mock(options=SimpleNamespace(max_execution_time=None))
        backend, circuits = object(), build_circuits()
        factory = Mock(return_value=sampler)
        job = object()

        def check_run(actual_circuits, shots):
            self.assertEqual(sampler.options.max_execution_time, 120)
            self.assertIs(actual_circuits, circuits)
            self.assertEqual(shots, 2048)
            return job

        sampler.run.side_effect = check_run
        self.assertIs(submit_bounded_job(backend, circuits, 2048, 120, factory), job)
        factory.assert_called_once_with(mode=backend)
        sampler.run.assert_called_once()

    def test_submission_failure_does_not_retry(self):
        sampler = Mock(options=SimpleNamespace(max_execution_time=None))
        sampler.run.side_effect = TimeoutError("Simulated lost response; job may exist.")
        with self.assertRaises(TimeoutError):
            submit_bounded_job(object(), build_circuits(), 2048, 120, Mock(return_value=sampler))
        sampler.run.assert_called_once()

    def test_invalid_submission_never_creates_sampler(self):
        factory = Mock()
        with self.assertRaises(ValueError):
            submit_bounded_job(object(), build_circuits(), 2048, 600, factory)
        factory.assert_not_called()

    def test_duration_planning_estimate_and_fallback_are_labeled(self):
        circuit = Mock(name="test_circuit")
        circuit.name = "test_circuit"
        circuit.estimate_duration.return_value = 2e-6
        backend = Mock()
        backend.configuration.return_value = SimpleNamespace(rep_delay_range=[0, 0.0005])
        estimate = estimate_hardware_usage(backend, [circuit] * 6, 2048)
        self.assertAlmostEqual(estimate["estimated_system_seconds"], 30 + 2 * 2048 * 6 * 0.000502)
        self.assertIn("not_guaranteed", estimate["kind"])
        self.assertEqual(estimate["duration_fallback_circuits"], [])
        circuit.estimate_duration.side_effect = ValueError("missing duration")
        fallback = estimate_hardware_usage(backend, [circuit] * 6, 2048)
        self.assertEqual(fallback["duration_fallback_circuits"], ["test_circuit"] * 6)
        self.assertEqual(fallback["circuit_durations_seconds"], [0.001] * 6)

    def test_pending_usage_is_not_reported_as_final_zero(self):
        job = Mock()
        job.metrics.return_value = {"usage": {"status": "pending", "qpu_charge_time_seconds": 2.5}}
        job.usage.return_value = 0
        record = collect_job_usage(job)
        self.assertEqual(record["usage_status"], "pending")
        self.assertEqual(record["reported_qpu_charge_time_seconds"], 2.5)
        self.assertEqual(record["usage_method_seconds"], 0)
        self.assertIn("not proof of zero charge", record["note"])

    def test_metrics_error_does_not_destroy_results(self):
        job = Mock()
        job.metrics.side_effect = RuntimeError("Metadata temporarily unavailable")
        job.usage.side_effect = RuntimeError("Metadata temporarily unavailable")
        record = collect_job_usage(job)
        self.assertEqual(record["usage_status"], "unknown")
        self.assertIn("RuntimeError", record["metrics_error"])
        self.assertIn("RuntimeError", record["usage_error"])

    def test_fake_backend_executes_six_circuits_with_network_blocked(self):
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime.executor_sampler import Sampler
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden in offline test")):
            backend = FakeTorino()
            circuits = build_circuits()
            compiled = generate_preset_pass_manager(
                backend=backend, initial_layout=[0, 1, 2],
                optimization_level=1, seed_transpiler=11,
            ).run(circuits)
            estimate = estimate_hardware_usage(backend, compiled, 2048)
            self.assertLess(estimate["estimated_system_seconds"], 120)
            self.assertEqual(estimate["duration_fallback_circuits"], [])
            created_samplers = []

            def factory(mode):
                sampler = Sampler(mode=mode)
                sampler.options.simulator.seed_simulator = 11
                created_samplers.append(sampler)
                return sampler

            result = submit_bounded_job(backend, compiled, 64, 120, factory).result()
            self.assertEqual(len(created_samplers), 1)
            self.assertEqual(created_samplers[0].options.max_execution_time, 120)
            self.assertEqual(len(result), 6)
            counts = [pub.data.meas.get_counts() for pub in result]
            self.assertTrue(all(sum(row.values()) == 64 for row in counts))
            summary = analyze_counts(circuits, counts)
            self.assertIsNotNone(summary["baseline"]["fidelity"])
            self.assertIsNotNone(summary["protected"]["fidelity"])


if __name__ == "__main__":
    unittest.main()
