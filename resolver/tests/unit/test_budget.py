from __future__ import annotations

from ceph_autobuild_resolver.budget import Budget
from ceph_autobuild_resolver.providers.base import Usage


def test_iteration_cap_trips_has_capacity(cfg):
    budget = Budget.from_config(cfg)
    for _ in range(cfg.max_iterations):
        budget.record_iteration()
    assert not budget.has_capacity()
    assert budget.reason_for_stop() == "max_iterations"


def test_token_cap_trips_has_capacity(cfg):
    budget = Budget.from_config(cfg)
    budget.record_usage(Usage(input_tokens=cfg.run_token_budget, output_tokens=0))
    assert not budget.has_capacity()
    assert budget.reason_for_stop() == "token_budget_exceeded"


def test_unchanged_streak_trips(cfg):
    # cfg fixture sets max_unchanged_iterations=2
    budget = Budget.from_config(cfg)
    budget.record_unchanged_build()
    budget.record_unchanged_build()
    assert not budget.has_capacity()
    assert budget.reason_for_stop() == "no_progress"


def test_build_after_file_change_resets_streak_fully(cfg):
    # reset_unchanged_streak zeroes the streak completely, so it must
    # re-accumulate from scratch rather than tripping no_progress on the next
    # unchanged build. cfg fixture sets max_unchanged_iterations=2.
    budget = Budget.from_config(cfg)
    budget.record_unchanged_build()
    budget.reset_unchanged_streak()
    budget.record_unchanged_build()
    assert budget.unchanged_streak == 1
    assert budget.has_capacity()
    assert budget.reason_for_stop() == "ok"


def test_explicit_streak_reset(cfg):
    budget = Budget.from_config(cfg)
    budget.record_unchanged_build()
    budget.reset_unchanged_streak()
    assert budget.unchanged_streak == 0
    assert budget.has_capacity()
