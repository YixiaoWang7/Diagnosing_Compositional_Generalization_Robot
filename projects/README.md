# Independently packaged applications

Applications belong here with their own package metadata and dependencies.
This directory is excluded from RoboPolicy wheels and source distributions.
Applications import public `robopolicy` interfaces; the library never imports them.

- [`corl2026_cg/`](corl2026_cg/README.md) implements the paper's task designs,
  demonstration preparation, training orchestration, GT-stage instructions, and
  simulator evaluation. Install it separately to use the `cg` command.
- [`examples/minimal_project/`](../examples/minimal_project/README.md) shows a small
  independent library consumer.

The [paper-to-code audit](../docs/validation/paper-release-audit.md) defines the remaining
publication and reproduction work for the `corl2026_cg` branch.
