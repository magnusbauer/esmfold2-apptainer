# ESMFold2 Apptainer Runtime

This directory contains an Apptainer build for Biohub ESMFold2 with the model
weights baked in.

- `esmfold2.sif`: local runtime image.
- `esmfold2.def`: reproducible image definition.
- `examples/`: AF3 JSON examples and the unified prediction runner.
- `1mht_pred.cif`: previous guide validation output.

The shared copy for other users is expected at:

```bash
/net/software/containers/users/magnusb/esmfold2.sif
```

On GPU compute nodes that do not mount `/net/software`, use the repo-local copy:

```bash
/net/scratch/magnusb/git/ESMFold2/esmfold2.sif
```

## Main Command

Run from an interactive GPU shell or a Slurm job:

```bash
apptainer exec --nv /net/scratch/magnusb/git/ESMFold2/esmfold2.sif esmfold2_predict --input INPUT.json --output OUT.cif --metrics-json metrics.json
```

The default model path is the copy baked into the image. To avoid reading model
weights through the SIF mount, use the host-side snapshots in this repo:

```bash
apptainer exec --nv /net/scratch/magnusb/git/ESMFold2/esmfold2.sif esmfold2_predict --model /net/scratch/magnusb/git/ESMFold2/models/ESMFold2 --ccd-cache /net/scratch/magnusb/git/ESMFold2/models/ESMFold2 --input INPUT.json --output OUT.cif --metrics-json metrics.json
```

The same override can be set with `ESMFOLD2_MODEL_PATH` and
`ESMFOLD2_CCD_CACHE`.

`esmfold2_predict` accepts AlphaFold3 JSON, PDB, mmCIF, or a folder of those
files. Use AlphaFold3 JSON when ligands, DNA, RNA, residue modifications, seeds,
or covalent bonds must be represented exactly. Use `--output-dir` when you want
AF3-style output folders.

The model defaults in the CLI match the installed ESMFold2 builder defaults:
`--num-loops 3`, `--num-sampling-steps 200`, `--num-diffusion-samples 1`, and
AF3 `modelSeeds` are used unless `--seed` is supplied.

Each prediction writes AF3-style basic confidence sidecars next to the CIF:
`*_summary_confidences.json` and `*_confidences.json`. With `--output-dir`, the
runner writes one AF3-style job folder per input, with `seed-*_sample-*`
subfolders and top-level selected `*_model.cif`, `*_summary_confidences.json`,
and `*_confidences.json` files. The selected top-level files are hard links to
the best seed/sample when the filesystem supports hard links, so the AF3 shape
does not duplicate file contents. `--metrics-json` writes a small aggregate JSON
index. `--full-metrics` writes heavy tensors to `*_full_metrics.pkl` instead of
embedding them in JSON; in AF3-style output folders, the top-level selected
pickle is a hard link to the best seed/sample pickle.

## AlphaFold3 JSON

Run the Hugging Face HhaI/1MHT guide complex with AF3-style output:

```bash
apptainer exec --nv esmfold2.sif esmfold2_predict \
  --input examples/af3_inputs/hf_hhai_1mht.json \
  --output-dir examples/run_outputs/hf_hhai_1mht_full_metrics \
  --metrics-json examples/run_outputs/hf_hhai_1mht_full_metrics/metrics.json \
  --full-metrics
```

List parsed chains without loading the model:

```bash
apptainer exec esmfold2.sif esmfold2_predict \
  --input examples/af3_inputs/all_types_smoke.json \
  --list-inputs
```

## PDB Or mmCIF

PDB/mmCIF input is loaded with AtomWorks and converted into ESMFold2 protein,
DNA, RNA, and ligand inputs. Coordinates are used only to recover entities; the
model predicts a new structure.

```bash
apptainer exec --nv esmfold2.sif esmfold2_predict \
  --input input.cif \
  --output prediction.cif \
  --metrics-json prediction.metrics.json
```

## Batch

Batch mode scans `.json`, `.pdb`, `.cif`, and `.mmcif` files, loads the model
once for all jobs, and writes one AF3-style folder per input:

```bash
apptainer exec --nv esmfold2.sif esmfold2_predict \
  --input-dir inputs \
  --output-dir outputs \
  --metrics-json outputs/metrics.json
```

Use `--recursive` to scan nested folders and `--list-inputs` to inspect jobs
without loading the model.

For an input named `my_complex.json`, the batch layout is:

```text
outputs/my_complex/
  my_complex_model.cif
  my_complex_summary_confidences.json
  my_complex_confidences.json
  my_complex_ranking_scores.csv
  seed-0_sample-0/
    my_complex_seed-0_sample-0_model.cif
    my_complex_seed-0_sample-0_summary_confidences.json
    my_complex_seed-0_sample-0_confidences.json
```

## GPU Architecture Tests

Submit smoke tests for Ada, Blackwell, L40/L40S, A100, and A6000 nodes:

```bash
examples/run_gpu_arch_tests.sh
```

Outputs are written under `examples/gpu_test_outputs/<case>/`, including the
CIF, AF3-style confidence JSON, full metrics pickle, GPU info, command, and
Slurm log. The script uses the normal model defaults unless you explicitly set
`ESMFOLD2_GPU_TEST_USE_MODEL_DEFAULTS=0` for a quick CUDA smoke test.
If `models/ESMFold2` exists in this repo, the script automatically uses it with
`--model` and `--ccd-cache`; override with `ESMFOLD2_GPU_TEST_MODEL` and
`ESMFOLD2_GPU_TEST_CCD_CACHE`.

## Rebuild

```bash
APPTAINER_TMPDIR=/tmp apptainer build --force esmfold2.sif esmfold2.def
```

The image pins:

- `esm @ git+https://github.com/Biohub/esm.git@c94ed8d`
- `atomworks==2.2.0`
- `torch==2.9.1` from `https://download.pytorch.org/whl/cu128`
- `biohub/ESMFold2` revision `e1e189d0f5fb70c2693da2332eca4443c0ccccd6`
- `biohub/ESMC-6B` revision `89c554c46a44d825fbfbe3ce2a6bdc539770bdaa`
