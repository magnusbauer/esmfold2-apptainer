# Running ESMFold2

This page explains how to run the local ESMFold2 Apptainer image for structure
prediction. The runner is intended to tie into workflows in the same style as
AlphaFold3: use AF3-style JSON for rich complex inputs, and collect an mmCIF
prediction plus AF3-style confidence JSON sidecars as output.

## Image

Use the shared image when it is visible on the node:

```bash
IMAGE=/net/software/containers/users/magnusb/esmfold2.sif
```

On GPU compute nodes where `/net/software` is not mounted, use the repo-local
copy:

```bash
IMAGE=/net/scratch/magnusb/git/ESMFold2/esmfold2.sif
```

## Quick Start

Run from an interactive GPU shell or inside a Slurm job:

```bash
INPUT=/path/to/input.json
OUTDIR=/path/to/esmfold2_output
mkdir -p "${OUTDIR}"

apptainer exec --nv "${IMAGE}" esmfold2_predict \
  --input "${INPUT}" \
  --output-dir "${OUTDIR}" \
  --metrics-json "${OUTDIR}/metrics.json"
```

The output folder will contain:

- `<job_id>/<job_id>_model.cif`: the selected predicted structure in mmCIF
  format.
- `<job_id>/<job_id>_summary_confidences.json`: AF3-style summary confidence
  JSON for the selected structure.
- `<job_id>/<job_id>_confidences.json`: AF3-style confidence JSON for the
  selected structure.
- `<job_id>/<job_id>_ranking_scores.csv`: ranking scores for every seed/sample.
- `<job_id>/seed-*_sample-*/`: one folder per seed/sample, matching the AF3
  layout.
- `metrics.json`: aggregate metrics index for this run when `--metrics-json` is
  provided.

This means downstream code that already expects an AF3-like input/output shape
can usually be wired to ESMFold2 by pointing it at an AF3 JSON input and reading
the generated CIF plus confidence sidecars.

The top-level selected CIF and confidence files are hard links to the
best-ranked `seed-*_sample-*` files when the filesystem supports hard links.
They look like the normal AF3 top-level files but do not store duplicate file
contents.

## Recommended Input Format

AlphaFold3 JSON is the preferred input format for complexes because it can
represent proteins, DNA, RNA, ligands, residue modifications, seeds, and
covalent bonds. The runner also accepts `.pdb`, `.cif`, and `.mmcif`, but those
files are parsed only to recover the entities; ESMFold2 predicts a new
structure rather than reusing the input coordinates.

Minimal AF3-style JSON:

```json
{
  "name": "example_complex",
  "modelSeeds": [0],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MSEQUENCE"
      }
    },
    {
      "ligand": {
        "id": "L",
        "ccdCodes": ["ATP"]
      }
    }
  ],
  "dialect": "alphafold3",
  "version": 4
}
```

For a complete example, see:

```bash
/net/scratch/magnusb/git/ESMFold2/examples/af3_inputs/hf_hhai_1mht.json
```

## Check Inputs Without Running the Model

Use `--list-inputs` to validate parsing and show the chains and seeds without
loading the model:

```bash
apptainer exec "${IMAGE}" esmfold2_predict \
  --input "${INPUT}" \
  --list-inputs
```

This is useful before submitting a longer GPU job.

## Batch Runs

To run a folder of `.json`, `.pdb`, `.cif`, or `.mmcif` files while loading the
model only once:

```bash
INPUT_DIR=/path/to/inputs
OUTDIR=/path/to/esmfold2_batch_output

apptainer exec --nv "${IMAGE}" esmfold2_predict \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTDIR}" \
  --metrics-json "${OUTDIR}/metrics.json"
```

Add `--recursive` to scan nested folders. Add `--list-inputs` to inspect a batch
without loading the model.

## Multiple Seeds Or Samples

When an AF3 JSON contains multiple `modelSeeds`, or when you request multiple
diffusion samples with `--num-diffusion-samples`, each result gets its own
AF3-style folder:

```text
<job_id>/
  <job_id>_model.cif
  <job_id>_summary_confidences.json
  <job_id>_confidences.json
  <job_id>_ranking_scores.csv
  seed-1_sample-0/
    <job_id>_seed-1_sample-0_model.cif
    <job_id>_seed-1_sample-0_summary_confidences.json
    <job_id>_seed-1_sample-0_confidences.json
  seed-1_sample-1/
    <job_id>_seed-1_sample-1_model.cif
    <job_id>_seed-1_sample-1_summary_confidences.json
    <job_id>_seed-1_sample-1_confidences.json
```

The top-level files are the best-ranked seed/sample selected by
`ranking_score`. This mirrors AF3's layout, while hard links avoid storing the
selected model and confidence files twice when supported.

## Optional Host-Side Model Cache

The model weights are baked into the image. If you are running from the repo and
want to avoid reading model weights through the SIF mount, point the runner at
the host-side model snapshot:

```bash
apptainer exec --nv "${IMAGE}" esmfold2_predict \
  --model /net/scratch/magnusb/git/ESMFold2/models/ESMFold2 \
  --ccd-cache /net/scratch/magnusb/git/ESMFold2/models/ESMFold2 \
  --input "${INPUT}" \
  --output-dir "${OUTDIR}" \
  --metrics-json "${OUTDIR}/metrics.json"
```

The same paths can also be set with `ESMFOLD2_MODEL_PATH` and
`ESMFOLD2_CCD_CACHE`.

## Common Options

- `--seed N`: override AF3 `modelSeeds`. If omitted, AF3 JSON uses
  `modelSeeds`; PDB/mmCIF inputs use seed `0`.
- `--num-loops N`: ESMFold2 recycle loop count. The default is `3`.
- `--num-sampling-steps N`: diffusion sampling steps. The default is `200`.
- `--num-diffusion-samples N`: number of diffusion samples. The default is `1`.
- `--include-plddt`: include per-residue pLDDT values in confidence JSON.
- `--full-metrics`: write heavy tensor metrics to `*_full_metrics.pkl` next to
  each CIF. In AF3-style output folders, the top-level selected
  `<job_id>_full_metrics.pkl` is a hard link to the selected seed/sample pickle.
- `--output /path/to/prediction.cif`: write a simple flat single-CIF output
  instead of an AF3-style output folder.

## Example From This Repo

From `/net/scratch/magnusb/git/ESMFold2`:

```bash
IMAGE=/net/scratch/magnusb/git/ESMFold2/esmfold2.sif

apptainer exec --nv "${IMAGE}" esmfold2_predict \
  --input examples/af3_inputs/hf_hhai_1mht.json \
  --output-dir examples/run_outputs \
  --metrics-json examples/run_outputs/metrics.json
```

Expected outputs:

```text
examples/run_outputs/hf_hhai_1mht/hf_hhai_1mht_model.cif
examples/run_outputs/hf_hhai_1mht/hf_hhai_1mht_summary_confidences.json
examples/run_outputs/hf_hhai_1mht/hf_hhai_1mht_confidences.json
examples/run_outputs/hf_hhai_1mht/hf_hhai_1mht_ranking_scores.csv
examples/run_outputs/hf_hhai_1mht/seed-0_sample-0/hf_hhai_1mht_seed-0_sample-0_model.cif
examples/run_outputs/hf_hhai_1mht/seed-0_sample-0/hf_hhai_1mht_seed-0_sample-0_summary_confidences.json
examples/run_outputs/hf_hhai_1mht/seed-0_sample-0/hf_hhai_1mht_seed-0_sample-0_confidences.json
examples/run_outputs/metrics.json
```
