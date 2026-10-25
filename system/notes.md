# Notes

## System overview




| Part | File | Where |
|---|---|---|
| The blocks (responses, repeat, starting, header, save_model, finish) | `operations.py` 


| Part | File | Where | on |
|---|---|---|---|
| The blocks (responses, repeat, starting, header, save_model, finish) | `operations.py` | `class Operations` line 52; `responses` 103, `repeat` 126, `_loop` 132 (progress line, snapshot call, resume rule), `starting` 158, `header` 235, `save_model` 241, `finish` 247 | here|
| present / run_pass (moved, old names kept as delegates) | `operations.py` 86 / 164; delegates in `protocols.py` | |
