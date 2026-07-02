"""Experiment logging. Each run produces a self-contained results directory
(resolved config, metrics, per-sample records, raw outputs, and logs) so that a
run is fully reproducible from its artifacts alone.

Named ``runlog`` rather than ``logging`` to avoid shadowing the standard library.
"""
