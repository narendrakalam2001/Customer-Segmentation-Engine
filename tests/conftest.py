# ============================================================
# PYTEST CONFTEST — Customer Segmentation Engine
# ============================================================
# Pytest imports conftest.py before collecting/importing any test
# module, so this is the earliest reliable place to set OMP_NUM_THREADS
# before scikit-learn (and its MKL-linked BLAS backend) gets imported
# anywhere in the test suite.
#
# Why: on Windows, KMeans has a known memory-leak warning when MKL
# spawns more threads than there are data chunks to process. This is
# a scikit-learn/Windows-MKL interaction, not a bug in this project —
# scikit-learn's own docs recommend this exact env var as the fix:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html
# ============================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
