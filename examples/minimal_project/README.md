# Independent example application

This package creates a small aligned HDF5 dataset and trains a state policy. It
imports only the public `robopolicy` surface. From the repository root:

```bash
uv pip install -e '.[dataset,logging]' -e examples/minimal_project
robopolicy-example --config examples/state_policy.json
python scripts/policies/train_library.py --resume runs/example/step-100.pt
```

The application owns dataset generation. Generated data belongs in `data/toy`,
outputs in `runs/example`, and neither belongs in version control. The library
owns training, checkpointing, normalization, and inference. Future applications
can be independent packages under `projects/` with their own task semantics and
extra dependencies; none are included in the library distribution.
