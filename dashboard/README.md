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
