# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The budget checker flags what it is supposed to flag.

The cases below are the ones that decide whether the check is worth having: a
target whose explicit `timeout` overrides what its `size` implies, a sharded
target whose budget applies per shard rather than to the sum, and a cached
result — which is most of what a merge run reports and therefore most of what
the check reads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from absl.testing import absltest

from tools import check_test_budgets as budgets

# The workspace root `_rule` writes its locations under. The two have to agree,
# so `_budgets` pairs them rather than leaving each call site to restate it.
WORKSPACE = Path("/w")


def _query_xml(*rules: str) -> str:
    header = '<?xml version="1.1" encoding="UTF-8" standalone="no"?>'
    return f'{header}\n<query>{"".join(rules)}</query>'


def _rule(label: str, size: str, timeout: str, line: int = 7) -> str:
    return (
        f'<rule class="py_test" location="/w/pkg/BUILD.bazel:{line}:8" name="{label}">'
        f'<string name="size" value="{size}"/>'
        f'<string name="timeout" value="{timeout}"/>'
        f"</rule>"
    )


def _budgets(*rules: str) -> dict[str, budgets.Budget]:
    """Parsed budgets for `rules`, rooted where `_rule` puts them."""
    return budgets.parse_budgets(_query_xml(*rules), WORKSPACE)


def _bep(*events: Mapping[str, object]) -> str:
    return "\n".join(json.dumps(event) for event in events)


def _result(
    label: str, millis: int, *, shard: int = 1, attempt: int = 1, cached: bool = False
) -> dict[str, object]:
    """One `testResult` event.

    `parse_durations` keys on `label` alone — `run`, `shard` and `attempt` are
    carried so the fixture matches what bazel actually emits, not because the
    parser reads them.
    """
    payload: dict[str, object] = {"testAttemptDurationMillis": str(millis)}
    if cached:
        payload["cachedLocally"] = True
    return {
        "id": {
            "testResult": {"label": label, "run": 1, "shard": shard, "attempt": attempt}
        },
        "testResult": payload,
    }


def _options(*flags: str) -> dict[str, object]:
    """The `optionsParsed` event, carrying the rc-expanded command line."""
    return {"id": {"optionsParsed": {}}, "optionsParsed": {"cmdLine": list(flags)}}


def _row(millis: int, *, line: int = 7) -> budgets.Row:
    """One joined row for `//pkg:a` — the shape `annotate` formats."""
    parsed = _budgets(_rule("//pkg:a", "small", "short", line=line))
    durations = budgets.parse_durations(_bep(_result("//pkg:a", millis)))
    return budgets.collect(parsed, durations)[0]


class ParseBudgetsTest(absltest.TestCase):
    def test_reads_the_bucket_the_timeout_names(self) -> None:
        self.assertEqual(
            _budgets(_rule("//pkg:a", "small", "short"))["//pkg:a"].seconds, 60
        )

    def test_an_explicit_timeout_beats_the_size(self) -> None:
        """`size = "large", timeout = "eternal"` is 3600 s, not 900 s.

        Reading the budget off `size` would report such a target at four times
        the fraction of budget it actually uses, and flag one that is fine.
        """
        parsed = _budgets(_rule("//pkg:traced", "large", "eternal"))
        self.assertEqual(parsed["//pkg:traced"].seconds, 3600)

    def test_location_becomes_a_repo_relative_annotation_target(self) -> None:
        parsed = _budgets(_rule("//pkg:a", "small", "short", line=42))
        self.assertEqual(parsed["//pkg:a"].path, "pkg/BUILD.bazel")
        self.assertEqual(parsed["//pkg:a"].line, "42")

    def test_a_non_test_rule_is_skipped(self) -> None:
        """A `py_library` carries no timeout, so it has no budget to check."""
        rule = (
            '<rule class="py_library" location="/w/pkg/BUILD.bazel:1:8"'
            ' name="//pkg:lib"/>'
        )
        self.assertEqual(_budgets(rule), {})


class ParseDurationsTest(absltest.TestCase):
    def test_takes_the_worst_shard_not_the_sum(self) -> None:
        """Each shard gets the whole timeout, so the sum is not the figure.

        Summing would flag a target that shards precisely so that no shard comes
        near its budget — the opposite of what the check is for.
        """
        parsed = budgets.parse_durations(
            _bep(
                _result("//pkg:sweep", 100_000, shard=1),
                _result("//pkg:sweep", 130_000, shard=2),
                _result("//pkg:sweep", 90_000, shard=3),
            )
        )
        self.assertEqual(parsed["//pkg:sweep"].seconds, 130.0)

    def test_takes_the_worst_retry_attempt(self) -> None:
        parsed = budgets.parse_durations(
            _bep(
                _result("//pkg:flaky", 10_000, attempt=1),
                _result("//pkg:flaky", 55_000, attempt=2),
            )
        )
        self.assertEqual(parsed["//pkg:flaky"].seconds, 55.0)

    def test_a_cached_result_counts_and_says_so(self) -> None:
        """A merge run is almost entirely cache hits.

        The duration a cache hit reports is the last real execution of that
        configuration, which is the number that applies the next time the target
        runs — so it is measured, not skipped, and marked for the reader.
        """
        parsed = budgets.parse_durations(_bep(_result("//pkg:a", 45_000, cached=True)))
        self.assertEqual(parsed["//pkg:a"].seconds, 45.0)
        self.assertTrue(parsed["//pkg:a"].cached)

    def test_ignores_events_that_are_not_test_results(self) -> None:
        progress = {"id": {"progress": {"opaqueCount": 1}}, "progress": {}}
        self.assertEqual(budgets.parse_durations(_bep(progress)), {})


class CollectTest(absltest.TestCase):
    def test_orders_by_fraction_of_budget_not_by_duration(self) -> None:
        """A 40 s `small` test is in more danger than a 300 s `large` one."""
        parsed = _budgets(
            _rule("//pkg:small_one", "small", "short"),
            _rule("//pkg:large_one", "large", "long"),
        )
        rows = budgets.collect(
            parsed,
            budgets.parse_durations(
                _bep(
                    _result("//pkg:small_one", 40_000),
                    _result("//pkg:large_one", 300_000),
                )
            ),
        )
        self.assertEqual(
            [row.label for row in rows], ["//pkg:small_one", "//pkg:large_one"]
        )

    def test_a_target_filtered_off_this_leg_is_skipped(self) -> None:
        """`--test_tag_filters` drops targets, and a budget alone is not a run."""
        self.assertEqual(
            budgets.collect(_budgets(_rule("//pkg:sweep", "large", "long")), {}), []
        )


class MainTest(absltest.TestCase):
    def _xml(self, size: str = "small") -> str:
        return self.create_tempfile(
            content=_query_xml(_rule("//pkg:a", size, "short"))
        ).full_path

    def _run(self, size: str, millis: int, mode: str) -> int:
        bep = self.create_tempfile(content=_bep(_result("//pkg:a", millis)))
        return budgets.main(
            ["--bep", bep.full_path, "--query-xml", self._xml(size), "--mode", mode]
        )

    def test_under_the_threshold_passes(self) -> None:
        self.assertEqual(self._run("small", 20_000, "fail"), 0)

    def test_the_threshold_itself_is_a_violation(self) -> None:
        """The rule reads "at or above 50%", so exactly half fails."""
        self.assertEqual(self._run("small", 30_000, "fail"), 1)

    def test_warn_mode_reports_without_failing(self) -> None:
        """A pull request should not go red over a duration it did not change."""
        self.assertEqual(self._run("small", 55_000, "warn"), 0)

    def test_a_missing_build_event_file_stands_down(self) -> None:
        """The check runs after a red test step, which may not have written one.

        Failing here would bury the failure the leg actually has under a
        traceback from the check.
        """
        missing = Path(self.create_tempdir().full_path) / "absent.json"
        self.assertEqual(
            budgets.main(["--bep", str(missing), "--query-xml", self._xml()]), 0
        )

    def test_a_build_event_file_with_no_tests_stands_down(self) -> None:
        """An analysis failure writes a build event file carrying no results."""
        bep = self.create_tempfile(content="")
        self.assertEqual(
            budgets.main(["--bep", bep.full_path, "--query-xml", self._xml()]), 0
        )

    def test_a_test_timeout_override_refuses_rather_than_mis_measures(self) -> None:
        """The declared `timeout` is not the budget bazel enforced under it.

        Reporting a percentage against the wrong denominator is worse than
        reporting nothing, because it looks like a measurement.
        """
        bep = self.create_tempfile(
            content=_bep(
                _options("--test_output=errors", "--test_timeout=1200"),
                _result("//pkg:a", 20_000),
            )
        )
        self.assertEqual(
            budgets.main(["--bep", bep.full_path, "--query-xml", self._xml()]), 1
        )


class OverridingTimeoutFlagTest(absltest.TestCase):
    def test_finds_the_flag_wherever_the_rc_files_put_it(self) -> None:
        bep = _bep(_options("--test_output=errors", "--test_timeout=1200"))
        self.assertEqual(budgets.overriding_timeout_flag(bep), "--test_timeout=1200")

    def test_an_ordinary_run_carries_none(self) -> None:
        bep = _bep(_options("--test_output=errors", "--test_env=FRX_PLATFORMS=cpu"))
        self.assertIsNone(budgets.overriding_timeout_flag(bep))


class AnnotateTest(absltest.TestCase):
    def test_points_at_the_build_line_that_declares_the_budget(self) -> None:
        annotation = budgets.annotate(_row(57_300, line=42), "cpu", 0.5, "error")
        self.assertIn("file=pkg/BUILD.bazel,line=42", annotation)
        self.assertIn("96%", annotation)
        self.assertIn("cpu", annotation)

    def test_carries_no_newline_that_would_truncate_it(self) -> None:
        """GitHub reads an annotation to the end of the line and no further."""
        self.assertNotIn("\n", budgets.annotate(_row(57_300), "cpu", 0.5, "error"))

    def test_the_property_block_carries_no_separator_character(self) -> None:
        """`:` and `,` end a property value, so neither may appear in one.

        A target name is `pkg:name`, which is exactly the shape that would
        truncate the title back to `test budget — pkg` if it leaked in.
        """
        annotation = budgets.annotate(_row(57_300), "cpu", 0.5, "error")
        properties = annotation.split("::")[1]
        for value in (pair.split("=", 1)[1] for pair in properties.split(",")):
            self.assertNotIn(":", value)


if __name__ == "__main__":
    absltest.main()
