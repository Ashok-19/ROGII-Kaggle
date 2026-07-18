# ROGII Dashboard

Start the dashboard:

```bash
python tools/rogii.py dashboard --open
```

Default address: `http://127.0.0.1:8765`.

The server runs a manifest sync before serving state and re-syncs when experiment manifests change. The SQLite database is generated at `tracking/rogii.sqlite` and is intentionally ignored by Git.

Validate the project system:

```bash
python tools/rogii.py init
python tools/rogii.py sync
python tools/rogii.py validate
```

The dashboard is read-only. Edit manifests or use the CLI to record state; do not hand-edit generated database rows.

## Learning Lab

Open `http://127.0.0.1:8765/learn` for the beginner-first visual guide and real-well playground. It explains the target, geometry, gamma ray, typewell, formation surfaces, error decomposition, validation traps, and experiment idea space.

The playground reads downsampled views of local competition files through read-only endpoints:

- `/api/learning` — explanations, E001 metrics, the 773-well catalog, and breakthrough timeline.
- `/api/learning/well?well_id=<id>` — one real horizontal well, its typewell, formation surfaces, and E001 diagnostics.

The adjustable playground score is an educational sampled-row simulation. Exact experiment metrics remain sourced from validated manifests and E001 artifacts.
