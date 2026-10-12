# Inspect a checkpoint

Checkpoint inspection reports the format and tensor metadata of a saved model without training or changing the checkpoint.

## Sub-features

- `checkpoint-json` emits machine-readable checkpoint metadata.
- `checkpoint-text` emits a human-readable summary.

## How to get to it (user POV)

- Run `python scripts/harness.py checkpoint inspect <path>` from the repository root.
- Add `--json` to request structured output.

## Driving it with the shell

Preconditions:

- The checkpoint exists and is readable.
- Choose an existing checkpoint for read-only inspection. Do not use a path that a concurrent run may overwrite.

- Action: `uv run python scripts/harness.py checkpoint inspect checkpoints/best.pt --json`. Proof: exit status is 0, JSON parses, and it includes `path`, `tensor_count`, and `checkpoint_api_version`. Replace the path only if `checkpoints/best.pt` is absent.
- Action: save stdout to the evidence directory and parse it with `uv run python -c 'import json,sys; d=json.load(sys.stdin); assert d["tensor_count"] > 0; print(d["checkpoint_api_version"], d["tensor_count"])'`. Proof: parsing and the positive tensor assertion succeed.

## Gotchas

- Checkpoint inspection loads the selected artifact. Large checkpoints can take time and memory.
- Metadata does not establish that the model behaves correctly on real inputs. Behavioral validation is a separate check.
- Do not print model tensors or copy checkpoint contents into evidence.
