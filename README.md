# ESMFold2 Apptainer Runtime

This repository provides an Apptainer definition and lightweight example
wrappers for running Biohub ESMFold2 locally with NVIDIA GPU support.

The upstream ESMFold2 model, ESMC-6B model, and `esm` Python package are not
authored here. This repository only packages pinned upstream artifacts into a
reproducible Apptainer runtime and adds convenience input/output helpers.

See [ATTRIBUTION.md](ATTRIBUTION.md) for upstream project links, license notes,
and citation guidance.

## Contents

- `esmfold2.def`: Apptainer build definition.
- `examples/esmfold2_predict.py`: CLI for AlphaFold3 JSON, PDB, mmCIF, and
  batch inputs.
- `examples/af3_inputs/`: small AlphaFold3 JSON examples.

Built images, model snapshots, and prediction outputs are intentionally ignored
by git.

## Build

Build the SIF image from the repository root:

```bash
apptainer build --force esmfold2.sif esmfold2.def
```

For large builds, point Apptainer temporary and cache directories at storage with
enough free space:

```bash
APPTAINER_TMPDIR=/path/to/tmp \
APPTAINER_CACHEDIR=/path/to/cache \
apptainer build --force esmfold2.sif esmfold2.def
```

The build downloads pinned versions of:

- `esm @ git+https://github.com/Biohub/esm.git@c94ed8d`
- `atomworks==2.2.0`
- `torch==2.9.1` from `https://download.pytorch.org/whl/cu128`
- `biohub/ESMFold2` revision `e1e189d0f5fb70c2693da2332eca4443c0ccccd6`
- `biohub/ESMC-6B` revision `89c554c46a44d825fbfbe3ce2a6bdc539770bdaa`

## Run

Run a single AlphaFold3 JSON input:

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --input examples/af3_inputs/hf_hhai_1mht.json \
  --output outputs/hf_hhai_1mht.cif \
  --metrics-json outputs/metrics.json
```

Inspect parsed inputs without loading the model:

```bash
apptainer exec ./esmfold2.sif esmfold2_predict \
  --input examples/af3_inputs/all_types_smoke.json \
  --list-inputs
```

Run a PDB or mmCIF input:

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --input input.cif \
  --output outputs/prediction.cif \
  --metrics-json outputs/prediction.metrics.json
```

Run a batch of `.json`, `.pdb`, `.cif`, and `.mmcif` files:

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --input-dir inputs \
  --output-dir outputs \
  --metrics-json outputs/metrics.json
```

Add `--recursive` to scan nested folders. Add `--full-metrics` to write large
metric tensors to `*_full_metrics.pkl` sidecars.

## Model Paths

By default, the image uses the ESMFold2 and ESMC-6B snapshots baked into
`/opt/esmfold2/models`. You can override the model and CCD cache with host-side
paths when desired:

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --model /path/to/ESMFold2 \
  --ccd-cache /path/to/ESMFold2 \
  --input input.json \
  --output outputs/prediction.cif
```

The same defaults can be overridden with `ESMFOLD2_MODEL_PATH` and
`ESMFOLD2_CCD_CACHE`.

## Outputs

Each prediction writes:

- a predicted mmCIF file,
- `*_summary_confidences.json`,
- `*_confidences.json`,
- optional `*_full_metrics.pkl` when `--full-metrics` is set.

`--metrics-json` writes a compact aggregate index for all predictions in a run.
