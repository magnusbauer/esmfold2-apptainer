# Scoring sequences with the ESM family of language models

The ESM family of language models is a set of large language models trained on
protein sequences. You can read about them in their
[github repo](https://github.com/facebookresearch/esm) and in the
[ESMFold Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1).
The long story short is that these models are designed to memorize patterns from
natural sequences, and it was discovered in the
[ESM Design Paper](https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1.abstract)
that language model perplexity is a decent metric for gauging whether or not a
protein will express and further be functional, assuming it already passes AF2
structural metrics.

Language model perplexity is itself a metric that gauges how likely the model
thinks a given sequence is. It is a measurement of how close to "natural" a
protein looks, according to the model.

## Why should you care?

Many of the metrics in the lab are structural. Structural metrics don't always
capture everything - for example designs that pass all AF2 metrics can still
fail to express because the sequence doesn't lend itself to the expression
process. ESM perplexity can capture things like this and help screen designs for
things like likelihood of expression.

## Running scoring locally

There is an existing script that can score sequences in a fasta file, and you
can run it like so:

```bash
apptainer run --nv -B /software/scripts/esm /software/containers/esm.sif /software/scripts/esm/score_sequences.py --input_fasta_file my_fasta_file.fasta --output_file sequences_scores.csv --local
```

This assumes that you are already logged in to a GPU node. It will score each of
your sequences in the fasta file and output a csv file with a column indicating
the header of the sequence in the fasta file and its ESM perplexity.

## Running scoring on SLURM

If you turn off the `--local` flag, the script will submit a SLURM job to the
cluster that scores your sequences instead.

## Script arguments

The script has some additional arguments to customize the job. Run the following
help command to understand what each of them mean:

```bash
apptainer run --nv -B /software/scripts/esm /software/containers/esm.sif /software/scripts/esm/score_sequences.py --help
```

## Understanding the output

Perplexity is a measure from 1-20, where 1 is the best and 20 is the worst. Most
natural sequences have perplexities around 6. A good threshold for filtering de
novo designs is probably somewhere between 6 and 10, depending on your sequences
and the type of problem you are trying to solve.

# Running ESMFold2 structure prediction

ESMFold2 is separate from ESM sequence scoring. Instead of scoring sequence
naturalness/perplexity, it predicts structures from AF3-style inputs and writes
AF3-style outputs.

## Running ESMFold2 locally

From a GPU node:

```bash
IMAGE=/net/scratch/magnusb/git/ESMFold2/esmfold2.sif
INPUT=/path/to/input.json
OUTDIR=/path/to/esmfold2_output

apptainer exec --nv "${IMAGE}" esmfold2_predict \
  --input "${INPUT}" \
  --output-dir "${OUTDIR}" \
  --metrics-json "${OUTDIR}/metrics.json"
```

The preferred input is AlphaFold3-style JSON. This lets you specify proteins,
DNA, RNA, ligands, modifications, seeds, and covalent bonds in a format similar
to AF3.

## Output layout

ESMFold2 writes one AF3-style folder per input:

```text
<job_id>/
  <job_id>_model.cif
  <job_id>_summary_confidences.json
  <job_id>_confidences.json
  <job_id>_ranking_scores.csv
  seed-0_sample-0/
    <job_id>_seed-0_sample-0_model.cif
    <job_id>_seed-0_sample-0_summary_confidences.json
    <job_id>_seed-0_sample-0_confidences.json
metrics.json
```

If multiple seeds or diffusion samples are run, each gets its own
`seed-*_sample-*` folder. The best-ranked sample is also placed in the parent
folder as `<job_id>_model.cif` with matching confidence files. These top-level
files are hard links when possible, so they do not duplicate storage.

## Full metrics

To save all exposed ESMFold2 metric tensors:

```bash
apptainer exec --nv "${IMAGE}" esmfold2_predict \
  --input "${INPUT}" \
  --output-dir "${OUTDIR}" \
  --metrics-json "${OUTDIR}/metrics.json" \
  --full-metrics
```

This adds `*_full_metrics.pkl` files next to the CIF outputs. In AF3-style
output folders, the top-level selected `<job_id>_full_metrics.pkl` is a hard
link to the selected seed/sample pickle when the filesystem supports hard links.

## Batch runs

To run a folder of `.json`, `.pdb`, `.cif`, or `.mmcif` inputs while loading the
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

## Check inputs without running the model

Use `--list-inputs` to validate parsing and show the chains and seeds without
loading the model:

```bash
apptainer exec "${IMAGE}" esmfold2_predict \
  --input "${INPUT}" \
  --list-inputs
```

## Ligand warning

The runner uses the ligand representation provided in the input JSON. If a
ligand is specified with `ccdCodes`, that CCD entry is passed through directly.
The runner does not convert CCD ligands to SMILES or reinterpret the ligand
identity, so make sure `ccdCodes` or `smiles` is the representation you actually
want to predict with.

## Common ESMFold2 options

- `--seed N`: override AF3 `modelSeeds`. If omitted, AF3 JSON uses
  `modelSeeds`; PDB/mmCIF inputs use seed `0`.
- `--num-loops N`: ESMFold2 recycle loop count. The default is `3`.
- `--num-sampling-steps N`: diffusion sampling steps. The default is `200`.
- `--num-diffusion-samples N`: number of diffusion samples. The default is `1`.
- `--include-plddt`: include per-residue pLDDT values in confidence JSON.
- `--full-metrics`: write heavy tensor metrics to `*_full_metrics.pkl` next to
  each CIF.
- `--output /path/to/prediction.cif`: write a simple flat single-CIF output
  instead of an AF3-style output folder.

