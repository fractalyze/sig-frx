# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Fail when a target's declared budget stops covering what CI measures.

A test that times out reports a red build for a suite that works, and the fix is
always the same one-line edit — so the useful moment to catch it is while the
margin is shrinking, not after the flake. This reads the durations CI just
produced against the budgets the BUILD files declare, and refuses a target whose
worst run crosses the headroom threshold.

The threshold is a *fraction of budget* rather than a fixed slack because the
CPU leg's run-to-run spread for one unchanged target is wide enough that any
single measurement may be the fast one. `docs/reference/conventions.md` carries
the rule, the measured spread it comes from, and why cached results count.

Durations come from the build event protocol rather than the console summary,
because a sharded target's budget applies *per shard* and the summary reports
only the aggregate. Budgets come from `bazel query` rather than from the `size`
attribute alone, because an explicit `timeout` overrides what `size` implies.

Run it against a local run:

    bazel test //... --build_event_json_file=/tmp/bep.json
    python3 tools/check_test_budgets.py --bep /tmp/bep.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Bazel's four timeout buckets in ascending order, each paired with the `size`
# whose default timeout it is. `size` selects a bucket by default and an
# explicit `timeout` overrides it, so a budget is read off `timeout` — which
# `--xml:default_values` reports for every target either way. Ascending order is
# what makes "one bucket up" well defined, so the ladder is written once here
# and the lookups below are derived rather than restated.
BUCKETS = (
    ("short", "small", 60),
    ("moderate", "medium", 300),
    ("long", "large", 900),
    ("eternal", "enormous", 3600),
)
BUCKET_SECONDS = {timeout: seconds for timeout, _, seconds in BUCKETS}
# The bucket one step up, as (timeout, size). The largest has no successor.
NEXT_BUCKET = {lower[0]: upper[:2] for lower, upper in zip(BUCKETS, BUCKETS[1:])}


@dataclass(frozen=True)
class Budget:
    """What a BUILD file declares for one test target."""

    size: str
    timeout: str
    seconds: int
    path: str
    line: str


@dataclass(frozen=True)
class Measurement:
    """The worst run CI observed for one test target."""

    seconds: float
    cached: bool


def parse_budgets(xml_text: str, workspace: Path) -> dict[str, Budget]:
    """Read declared budgets out of `bazel query --output=xml` output.

    Requires `--xml:default_values`, which is what makes `timeout` present on a
    target that only declares `size`.
    """
    budgets: dict[str, Budget] = {}
    for rule in ET.fromstring(xml_text).findall("rule"):
        label = rule.get("name")
        if label is None:
            continue
        attrs = {
            s.get("name"): s.get("value") or ""
            for s in rule.findall("string")
            if s.get("name") in ("size", "timeout")
        }
        timeout = attrs.get("timeout", "")
        if timeout not in BUCKET_SECONDS:
            continue
        # Locations are absolute and carry a `line:col` suffix; annotations want
        # a repo-relative path and the line on its own.
        path, _, rest = rule.get("location", "").partition(":")
        line = rest.partition(":")[0]
        try:
            path = str(Path(path).resolve().relative_to(workspace))
        except ValueError:
            pass
        budgets[label] = Budget(
            size=attrs.get("size", ""),
            timeout=timeout,
            seconds=BUCKET_SECONDS[timeout],
            path=path,
            line=line,
        )
    return budgets


def overriding_timeout_flag(bep_text: str) -> str | None:
    """The `--test_timeout` in effect for the run, if one was set.

    A budget read off the `timeout` attribute is the *declared* one. `bazel
    query` cannot see `--test_timeout`, which overrides that attribute at
    execution — so a run carrying one is measured against a denominator bazel
    did not enforce, and every percentage here is quietly wrong. Nothing sets it
    today; this exists so that the day someone does, the check says so instead
    of reporting confident nonsense.

    The build event file records the rc-expanded command line, so the flag is
    visible however it arrived — `.bazelrc.ci`, a matrix leg, or a local
    `.bazelrc.user`.
    """
    for line in bep_text.splitlines():
        if '"optionsParsed"' not in line:
            continue
        options = json.loads(line).get("optionsParsed", {})
        for flag in options.get("cmdLine", []):
            if flag.startswith("--test_timeout"):
                return str(flag)
    return None


def parse_durations(bep_text: str) -> dict[str, Measurement]:
    """Read the worst per-attempt duration for each target out of a BEP file.

    Every shard, run and retry attempt gets the full timeout to itself, so the
    figure the budget has to cover is the maximum across them, never the sum.
    """
    worst: dict[str, Measurement] = {}
    for line in bep_text.splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        result_id = event.get("id", {}).get("testResult")
        if result_id is None:
            continue
        label = result_id.get("label")
        payload = event.get("testResult", {})
        millis = payload.get("testAttemptDurationMillis")
        if label is None or millis is None:
            continue
        cached = bool(payload.get("cachedLocally")) or bool(
            payload.get("executionInfo", {}).get("cachedRemotely")
        )
        seconds = int(millis) / 1000
        previous = worst.get(label)
        if previous is None or seconds > previous.seconds:
            worst[label] = Measurement(seconds=seconds, cached=cached)
    return worst


@dataclass(frozen=True)
class Row:
    """One target's measured duration against its declared budget."""

    label: str
    budget: Budget
    measurement: Measurement

    @property
    def ratio(self) -> float:
        return self.measurement.seconds / self.budget.seconds

    @property
    def remedy(self) -> str:
        """The one-line edit that restores the headroom.

        Both forms move the deadline; `size` also moves the target's scheduling
        weight, which is what decides between them on a leg that executes
        locally. The caller picks — this only names the next bucket.
        """
        nxt = NEXT_BUCKET.get(self.budget.timeout)
        if nxt is None:
            return "already at the largest bucket — split or shard the target"
        timeout, size = nxt
        return (
            f'size = "{size}", or timeout = "{timeout}" to move the '
            f"deadline without the scheduling weight"
        )


def collect(budgets: dict[str, Budget], durations: dict[str, Measurement]) -> list[Row]:
    """Join the two sources, worst offender first.

    A target present in only one source is skipped rather than guessed at: a
    budget with no duration was filtered out of this leg (`--test_tag_filters`),
    and a duration with no budget is not a target this checkout declares.
    """
    rows = [
        Row(label=label, budget=budgets[label], measurement=measurement)
        for label, measurement in durations.items()
        if label in budgets
    ]
    return sorted(rows, key=lambda row: row.ratio, reverse=True)


def format_table(rows: list[Row], leg: str) -> str:
    """A fixed-width listing, so a green run still shows where the margin is."""
    width = max(len(row.label) for row in rows)
    header = (
        f"{'target':{width}}  {'size':8} {'timeout':8} {'budget':>7} "
        f"{'worst':>8} {'used':>6}"
    )
    lines = [f"test time budgets — {leg} leg", header, "-" * len(header)]
    for row in rows:
        flag = " (cached)" if row.measurement.cached else ""
        lines.append(
            f"{row.label:{width}}  {row.budget.size:8} {row.budget.timeout:8} "
            f"{row.budget.seconds:6}s {row.measurement.seconds:7.1f}s "
            f"{row.ratio * 100:5.0f}%{flag}"
        )
    return "\n".join(lines)


def annotate(row: Row, leg: str, threshold: float, level: str) -> str:
    """A GitHub workflow annotation pinned to the target's BUILD line."""
    message = (
        f"{row.label} used {row.ratio * 100:.0f}% of its {row.budget.seconds}s "
        f"budget on the {leg} leg ({row.measurement.seconds:.1f}s), at or above "
        f"the {threshold * 100:.0f}% threshold. Raise it: {row.remedy}."
    )
    # No `:` or `,` in a property value — GitHub reads those as separators and
    # would truncate the title at the target name.
    title = f"test budget — {row.label.rsplit(':', 1)[-1]}"
    location = f"file={row.budget.path}" + (
        f",line={row.budget.line}" if row.budget.line else ""
    )
    return f"::{level} {location},title={title}::{message}"


def query_budgets_xml(bazelrc: str | None) -> str:
    """Ask bazel what every test target declares.

    `--xml:default_values` is what reports `timeout` on a target that declares
    only `size`; without it the budget of most targets would be unknown.
    """
    command = ["bazel"]
    if bazelrc:
        command.append(f"--bazelrc={bazelrc}")
    command += [
        "query",
        "tests(//...)",
        "--output=xml",
        "--xml:default_values",
    ]
    return subprocess.run(
        command, cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bep", required=True, help="build event json file from the test run"
    )
    parser.add_argument(
        "--query-xml",
        help="pre-fetched `bazel query --output=xml` output; queried if omitted",
    )
    parser.add_argument("--bazelrc", help="passed through to the bazel query")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="fraction of budget a target may use before it is flagged",
    )
    parser.add_argument("--leg", default="cpu", help="which CI leg produced --bep")
    parser.add_argument(
        "--mode",
        choices=("fail", "warn"),
        default="fail",
        help="`warn` annotates and exits 0; `fail` exits non-zero on a violation",
    )
    args = parser.parse_args(argv)

    # This step runs even when the test step went red, which is when a build
    # event file can be absent or empty. Say so and stand down: the leg already
    # has its failure, and burying it under a traceback from the check helps
    # nobody read which step actually broke.
    bep = Path(args.bep)
    if not bep.is_file():
        print(f"no build event file at {bep} — nothing to check")
        return 0
    bep_text = bep.read_text(encoding="utf-8")

    override = overriding_timeout_flag(bep_text)
    if override is not None:
        print(
            f"{override} was in effect, so the declared `timeout` attribute is "
            "not the budget bazel enforced and every percentage below would be "
            "wrong. Teach this script to read the override, or drop the flag."
        )
        return 1

    xml_text = (
        Path(args.query_xml).read_text(encoding="utf-8")
        if args.query_xml
        else query_budgets_xml(args.bazelrc)
    )
    budgets = parse_budgets(xml_text, REPO)
    durations = parse_durations(bep_text)
    rows = collect(budgets, durations)

    if not rows:
        print(f"no test results in {bep} — nothing to check")
        return 0

    print(format_table(rows, args.leg))

    violations = [row for row in rows if row.ratio >= args.threshold]
    if not violations:
        print(
            f"\nall {len(rows)} targets are under "
            f"{args.threshold * 100:.0f}% of their budget"
        )
        return 0

    level = "error" if args.mode == "fail" else "warning"
    print()
    for row in violations:
        print(annotate(row, args.leg, args.threshold, level))
    print(
        f"\n{len(violations)} target(s) at or above "
        f"{args.threshold * 100:.0f}% of budget on the {args.leg} leg."
    )
    return 1 if args.mode == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
