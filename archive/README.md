# ROGII Research Archive

Snapshot: 2026-07-18

This archive contains the evidence used to create the project roadmap and experiment system.

- Official competition snapshot, leaderboard, submissions, and copied-notebook dependency audit.
- Full text of the two winning working notes supplied by the user.
- All 132 competition discussion topics and 981 returned messages, normalized into 1 JSONL chunks.
- A source registry and claim registry that distinguish official facts, local measurements, controlled writeup evidence, and unverified participant claims.

## Search examples

```bash
rg -n "datum|oracle|worst well" archive/discussions
rg -n "negative result|real-LB" archive/writeups
python tools/rogii.py sync
```

## Evidence classes

- `official`: competition rules/pages.
- `local_data`: reproduced from local competition data.
- `writeup_controlled` / `writeup_real_lb`: author-reported controlled experiments with configurations or score pairs.
- `participant_claims`: useful leads, never proof.

The dashboard reads the compact source and claim registries. The JSONL files retain full searchable context.
