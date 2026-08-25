from pathlib import Path
import unittest

from scripts.audit_v2_protocol import audit_paired_schedules


def audit_result(filename, signature):
    return {
        "path": Path(filename),
        "issues": [],
        "pairing_key": (202608121, 0, 8, 10, "balanced", ("Agent_1", "Agent_2")),
        "pairing_signature": signature,
    }


class PairedScheduleAuditTests(unittest.TestCase):
    def test_matching_realized_schedules_pass(self):
        signature = ((1, (("dyad_1", "Agent_1", "Agent_2"),)),)
        results = [
            audit_result("control.json", signature),
            audit_result("treatment.json", signature),
        ]

        audit_paired_schedules(results)

        self.assertEqual(results[0]["issues"], [])
        self.assertEqual(results[1]["issues"], [])

    def test_mismatched_realized_schedules_fail_both_runs(self):
        results = [
            audit_result(
                "control.json",
                ((1, (("dyad_1", "Agent_1", "Agent_2"),)),),
            ),
            audit_result(
                "treatment.json",
                ((1, (("dyad_1", "Agent_2", "Agent_1"),)),),
            ),
        ]

        audit_paired_schedules(results)

        for result in results:
            self.assertEqual(len(result["issues"]), 1)
            self.assertIn("realized pairing schedule differs", result["issues"][0])


if __name__ == "__main__":
    unittest.main()
