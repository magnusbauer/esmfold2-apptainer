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
- `examples/run_gpu_arch_tests.sh`: optional Slurm GPU architecture smoke tests.
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

The image uses a CUDA 12.8 devel base and installs xFormers, Transformer Engine,
and FlashAttention to avoid the pure-PyTorch fallback paths where supported. The
build downloads pinned versions of:

- `esm @ git+https://github.com/Biohub/esm.git@c94ed8d`
- `atomworks==2.2.0`
- `torch==2.9.1` from `https://download.pytorch.org/whl/cu128`
- `xformers==0.0.33.post2`
- `transformer-engine[pytorch]==2.15.0`
- `flash-attn==2.8.3`
- `biohub/ESMFold2` revision `e1e189d0f5fb70c2693da2332eca4443c0ccccd6`
- `biohub/ESMC-6B` revision `89c554c46a44d825fbfbe3ce2a6bdc539770bdaa`

## Run

Run a single AlphaFold3 JSON input and write an AF3-style output folder:

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --input examples/af3_inputs/hf_hhai_1mht.json \
  --output-dir outputs \
  --metrics-json outputs/metrics.json
```

Inspect parsed inputs without loading the model:

```bash
apptainer exec ./esmfold2.sif esmfold2_predict \
  --input examples/af3_inputs/all_types_smoke.json \
  --list-inputs
```

Run a PDB or mmCIF input. Structure files are parsed with AtomWorks to recover
protein, DNA, RNA, and ligand entities; ESMFold2 predicts a new structure.

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --input input.cif \
  --output-dir outputs \
  --metrics-json outputs/metrics.json
```

Run a batch of `.json`, `.pdb`, `.cif`, and `.mmcif` files. The model is loaded
once for all jobs in the batch:

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --input-dir inputs \
  --output-dir outputs \
  --metrics-json outputs/metrics.json
```

Add `--recursive` to scan nested folders. Add `--full-metrics` to write large
metric tensors to `*_full_metrics.pkl` sidecars.

## Multiple Samples

Use `--num-diffusion-samples` to produce multiple samples from one seed in one
model load:

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --input-dir examples/af3_inputs \
  --output-dir outputs \
  --metrics-json outputs/metrics.json \
  --full-metrics \
  --seed 0 \
  --num-diffusion-samples 2
```

Each job gets `seed-0_sample-0`, `seed-0_sample-1`, and a top-level selected
model chosen by `ranking_score`.

## Inputs

AlphaFold3 JSON is the preferred format for complexes because it can represent
proteins, DNA, RNA, ligands, residue modifications, seeds, and covalent bonds.
If a ligand is specified with `ccdCodes`, that CCD entry is passed through as
the ligand input. The runner does not convert CCD ligands to SMILES or
reinterpret ligand identity.

The model defaults in the CLI match the installed ESMFold2 builder defaults:
`--num-loops 3`, `--num-sampling-steps 200`, and
`--num-diffusion-samples 1`. AF3 `modelSeeds` are used unless `--seed` is
supplied.

## Outputs

With `--output-dir`, the runner writes one AF3-style job folder per input:

```text
outputs/<job_id>/
  <job_id>_model.cif
  <job_id>_confidences.json
  <job_id>_ranking_scores.csv
  <job_id>_full_metrics.pkl
  seed-0_sample-0/
    <job_id>_seed-0_sample-0_model.cif
    <job_id>_seed-0_sample-0_confidences.json
    <job_id>_seed-0_sample-0_full_metrics.pkl
```

`*_confidences.json` is the only confidence JSON sidecar; separate
`*_summary_confidences.json` files are not written. `--metrics-json` writes a
compact aggregate index for all predictions in the run. Top-level selected files
are hard links to the best seed/sample files when the filesystem supports hard
links.

Use `--output prediction.cif` for a simple flat single-CIF output instead of an
AF3-style output folder.

## Model Paths

By default, the image uses the ESMFold2 and ESMC-6B snapshots baked into
`/opt/esmfold2/models`. You can override the model and CCD cache with host-side
paths when desired:

```bash
apptainer exec --nv ./esmfold2.sif esmfold2_predict \
  --model /path/to/ESMFold2 \
  --ccd-cache /path/to/ESMFold2 \
  --input input.json \
  --output-dir outputs
```

The same defaults can be overridden with `ESMFOLD2_MODEL_PATH` and
`ESMFOLD2_CCD_CACHE`.

## GPU Architecture Tests

Submit one Slurm job for each configured GPU case:

```bash
examples/run_gpu_arch_tests.sh
```

The script covers Ada 4000, Blackwell B4000/B6000/B6000Q, L40, L40S, A100, and
A6000. Each case writes the same AF3-style output layout under its case folder,
plus `metrics.json`, `gpu_info.txt`, `command.txt`, `timing.txt`, and the Slurm
log. The script uses the normal model defaults unless
`ESMFOLD2_GPU_TEST_USE_MODEL_DEFAULTS=0` is set for a quick CUDA smoke test.
