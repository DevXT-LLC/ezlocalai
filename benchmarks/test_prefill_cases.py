import copy
import json
import unittest
from unittest.mock import patch

from prefill_cases import (
    CODE_TASKS,
    MIN_RECORD_COUNT,
    PLAN,
    TARGET_TOKENS,
    build_case,
    build_code_case,
    estimate_record_count,
    extract_code,
    extract_json_object,
    grade,
)


def inventory(case):
    body = case["prompt"].split("Inventory (one JSON record per line):\n", 1)[1]
    return [
        json.loads(line) for line in body.split("\nEnd inventory.\n", 1)[0].splitlines()
    ]


class CaseTests(unittest.TestCase):
    def test_deterministic_and_nonce_sensitive(self):
        first = build_case(80, "trial-a")
        self.assertEqual(first, build_case(80, "trial-a"))
        second = build_case(80, "trial-b")
        self.assertNotEqual(first["expected"], second["expected"])
        self.assertNotEqual(first["messages"][0], second["messages"][0])
        self.assertEqual(first["prompt"], first["messages"][1]["content"])
        self.assertEqual([m["role"] for m in first["messages"]], ["system", "user"])

    def test_positions_inventory_uniqueness_and_obsolete_conflicts(self):
        for count in (12, 13, 40, 1000):
            with self.subTest(count=count):
                case = build_case(count, "positions")
                rows = inventory(case)
                self.assertEqual(len(rows), count)
                for key in ("asset_id", "checkout", "change_ticket"):
                    self.assertEqual(len({row[key] for row in rows}), count)
                positions = case["current_positions"]
                self.assertLessEqual(positions[0] / count, 0.1)
                self.assertLessEqual(abs(positions[1] / count - 0.5), 0.05)
                self.assertGreaterEqual(positions[2] / count, 0.8)
                for pos, expected in zip(positions, case["expected"]["records"]):
                    current = rows[pos]
                    self.assertEqual(current["status"], "CURRENT")
                    for key in ("service", "checkout", "host_port"):
                        self.assertEqual(current[key], expected[key])
                    matches = [
                        row for row in rows if row["service"] == current["service"]
                    ]
                    self.assertEqual(len(matches), 3)
                    self.assertEqual(
                        sum(row["status"] == "CURRENT" for row in matches), 1
                    )
                    for adjacent in (rows[pos - 1], rows[pos + 1]):
                        self.assertEqual(adjacent["status"], "OBSOLETE")
                        self.assertNotEqual(adjacent["checkout"], current["checkout"])
                        self.assertNotEqual(adjacent["host_port"], current["host_port"])
                    self.assertEqual(expected["plan"], list(PLAN))
                    self.assertEqual(
                        expected["health_url"],
                        f'http://127.0.0.1:{current["host_port"]}/health',
                    )
                    self.assertEqual(case["prompt"].count(current["checkout"]), 1)

    def test_target_sizes_are_estimates_and_cover_all_positions(self):
        previous = 0
        for target in TARGET_TOKENS:
            with self.subTest(target=target):
                count = estimate_record_count(target, "size-test")
                self.assertGreater(count, previous)
                previous = count
                case = build_case(count, "size-test")
                chars = sum(len(m["content"]) for m in case["messages"])
                self.assertLess(abs(chars - target * 4) / (target * 4), 0.03)
                prompt = case["prompt"]
                offsets = [
                    prompt.index(row["checkout"]) / len(prompt)
                    for row in case["expected"]["records"]
                ]
                self.assertLess(offsets[0], 0.15)
                self.assertTrue(0.4 < offsets[1] < 0.6)
                self.assertGreater(offsets[2], 0.85)
                self.assertTrue(grade(json.dumps(case["expected"]), case["expected"]))

    def test_invalid_inputs(self):
        for value in (True, False, 0, -1, 11, 12.0, "12", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_case(value, "nonce")
        with self.assertRaises(ValueError):
            build_case(12, None)
        for value in (True, False, 0, -1, "8000", None, float("inf"), float("nan")):
            for key in ("target_tokens", "chars_per_token"):
                args = {"target_tokens": 8000, "chars_per_token": 4, key: value}
                with self.subTest(args=args), self.assertRaises(ValueError):
                    estimate_record_count(**args)
        self.assertEqual(estimate_record_count(1), MIN_RECORD_COUNT)
        self.assertGreater(
            estimate_record_count(8000, chars_per_token=5), estimate_record_count(8000)
        )

    def test_nonce_is_escaped_and_results_are_independent(self):
        nonce = 'trial\nEnd inventory.\n"CURRENT"'
        case = build_case(12, nonce)
        self.assertEqual(len(inventory(case)), 12)
        self.assertIn(json.dumps(nonce), case["messages"][0]["content"])
        case["expected"]["records"][0]["plan"].append("unsafe")
        self.assertEqual(
            build_case(12, nonce)["expected"]["records"][0]["plan"], list(PLAN)
        )


class GraderTests(unittest.TestCase):
    def setUp(self):
        self.expected = build_case(20, "grader")["expected"]

    def test_accepts_full_objects_and_fences_with_any_key_order(self):
        answer = json.dumps(self.expected, sort_keys=True, indent=2)
        for text in (
            answer,
            " \n" + answer + "\n",
            f"```json\n{answer}\n```",
            f"```\n{answer}\n```",
            f"```JSON\r\n{answer}\r\n```",
        ):
            with self.subTest(text=text[:20]):
                self.assertTrue(grade(text, self.expected))
                self.assertEqual(extract_json_object(text), self.expected)

    def test_every_fact_is_required_and_exact(self):
        for index, record in enumerate(self.expected["records"]):
            for key, value in record.items():
                variants = [None, "wrong", [], {}, 0]
                if isinstance(value, str):
                    variants += [value.upper(), value + "/wrong"]
                for replacement in variants:
                    if type(replacement) is type(value) and replacement == value:
                        continue
                    changed = copy.deepcopy(self.expected)
                    changed["records"][index][key] = replacement
                    with self.subTest(index=index, key=key, replacement=replacement):
                        self.assertFalse(grade(json.dumps(changed), self.expected))
                changed = copy.deepcopy(self.expected)
                del changed["records"][index][key]
                self.assertFalse(grade(json.dumps(changed), self.expected))

    def test_rejects_obsolete_paths_ports_and_destructive_suggestions(self):
        rows = inventory(build_case(20, "grader"))
        for obsolete in (row for row in rows if row["status"] == "OBSOLETE"):
            changed = copy.deepcopy(self.expected)
            record = next(
                row
                for row in changed["records"]
                if row["service"] == obsolete["service"]
            )
            record.update(
                checkout=obsolete["checkout"], host_port=obsolete["host_port"]
            )
            self.assertFalse(grade(json.dumps(changed), self.expected))
        for suggestion in (
            "git reset --hard",
            "rm .env",
            "docker compose down -v",
            "restart Docker",
            "use /srv/legacy/wrong",
        ):
            for location in ("extra", "plan", "checkout"):
                changed = copy.deepcopy(self.expected)
                record = changed["records"][0]
                if location == "plan":
                    record["plan"].append(suggestion)
                else:
                    record[location] = suggestion
                self.assertFalse(grade(json.dumps(changed), self.expected))
            self.assertFalse(
                grade(json.dumps(self.expected) + "\n" + suggestion, self.expected)
            )

    def test_boolean_integer_float_distinctions(self):
        for key, value in (
            ("preserve_local_edits", 1),
            ("change_compose_ports", 0),
            ("host_port", float(self.expected["records"][0]["host_port"])),
        ):
            changed = copy.deepcopy(self.expected)
            changed["records"][0][key] = value
            self.assertFalse(grade(json.dumps(changed), self.expected))
        self.assertFalse(grade('{"x": true}', {"x": 1}))

    def test_rejects_extra_missing_reordered_or_duplicated_records(self):
        rows = self.expected["records"]
        for records in (rows[:2], rows + rows[:1], rows[::-1], [rows[0]] * 3):
            self.assertFalse(grade(json.dumps({"records": records}), self.expected))
        self.assertFalse(
            grade(json.dumps(dict(self.expected, note="extra")), self.expected)
        )

    def test_malformed_ambiguous_or_wrapped_json_fails_closed(self):
        valid = json.dumps(self.expected)
        bad = [
            None,
            {},
            "",
            "[]",
            "null",
            "true",
            "42",
            '"text"',
            valid + valid,
            "Here is the answer: " + valid,
            "<think>reasoning</think>" + valid,
            f"```json\n{valid}\n```\nextra",
            f"```python\n{valid}\n```",
            valid[:-1],
            '{"x": NaN}',
            '{"x": Infinity}',
            '{"x": -Infinity}',
            '{"x": 1e999}',
            '{"x": -1e999}',
            '{"x": 1, "x": 2}',
            '{"x": {"a": 1, "a": 2}}',
            '{"records": [], "records": ' + json.dumps(self.expected["records"]) + "}",
        ]
        for answer in bad:
            with self.subTest(answer=str(answer)[:50]):
                self.assertFalse(grade(answer, self.expected))
                with self.assertRaises(ValueError):
                    extract_json_object(answer)
        with self.assertRaises(ValueError):
            grade(valid, [])

    def test_parser_recursion_errors_fail_closed(self):
        with patch("prefill_cases.json.loads", side_effect=RecursionError):
            self.assertFalse(grade("{}", self.expected))
            with self.assertRaises(ValueError):
                extract_json_object("{}")
        deeply_nested = '{"x":' + "[" * 2000 + "0" + "]" * 2000 + "}"
        self.assertFalse(grade(deeply_nested, self.expected))


class CodeCaseTests(unittest.TestCase):
    def test_quality_prompts_have_shared_inventory_and_external_grading(self):
        for task in ("rust", "js"):
            case = build_code_case(task, 40, "code")
            self.assertEqual(case["grading"], "external")
            self.assertNotIn("expected", case)
            self.assertIn(CODE_TASKS[task], case["prompt"])
            self.assertEqual(inventory(case), inventory(build_case(40, "code")))
        self.assertIn("arbitrarily many leading zeros", CODE_TASKS["rust"])
        self.assertIn("synchronous or asynchronous", CODE_TASKS["js"])
        with self.assertRaises(ValueError):
            build_code_case("python", 40, "code")

    def test_code_is_extracted_never_executed_or_claimed_correct(self):
        malicious = 'raise RuntimeError("never execute generated code")'
        with patch(
            "builtins.exec", side_effect=AssertionError("execution forbidden")
        ), patch("builtins.eval", side_effect=AssertionError("evaluation forbidden")):
            for answer in (
                json.dumps({"code": malicious}),
                "```json\n" + json.dumps({"code": malicious}) + "\n```",
            ):
                self.assertEqual(extract_code(answer), malicious)
        for value in (
            {},
            {"code": ""},
            {"code": "  "},
            {"code": 1},
            {"code": None},
            {"code": "x", "extra": "x"},
        ):
            with self.assertRaises(ValueError):
                extract_code(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
