# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog, and this project follows semantic versioning principles.

## [0.2.0] - 2026-04-12

### Added
- Reproducibility metadata attached to each simulation output through a dedicated metadata structure.
- Portable result serialization utilities to export trajectories, final states, parameters, and metadata.
- Multi-seed simulation helper for deterministic batch workflows.
- Command-line interface options for steps, seed, beta, initialization mode, dtype, and optional output path.

### Changed
- Public API now exposes reproducibility and workflow utilities for scripted usage.
- Documentation significantly expanded with reproducibility protocol and formula-to-code mapping.
- CI pipeline upgraded with lint, format, tests, and coverage quality gates.
- Project quality tooling integrated with Ruff and stricter pytest coverage policy.

### Scientific Validation
- Expanded test suite with additional mathematical checks on dreaming kernel behavior and input validation branches.
- Added regression tests for CLI behavior, metadata helpers, random initialization validation, and multi-seed equivalence.
- Coverage threshold raised to 90% with measured total coverage above threshold.

### Notes
- Core Monte Carlo equations and DLAM field factorization remain aligned with the original article implementation.

## [0.1.0] - 2026-04-12

### Added
- Initial publication-grade DLAM Monte Carlo core package.
- Core modules for data generation, dreaming kernel computation, field dynamics, and simulation orchestration.
- Basic test suite, CI workflow, and project metadata.
- Documentation, contributing guidelines, citation metadata, and MIT license.
