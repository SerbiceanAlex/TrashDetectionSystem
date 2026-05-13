# Models

Production checkpoints used by the web app live here:

```text
models/
├── detector/
│   └── A4-8010/
│       └── best.pt
└── classify/
    └── B2/
        └── best.pt
```

The `.pt` files are ignored by Git because they are binary artifacts. Keep `runs/`
for training experiments and copy the selected final checkpoint here when a model
becomes the production candidate.
