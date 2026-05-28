# ESMFold2 Prediction Examples

Run these from `/net/scratch/magnusb/git/ESMFold2` or use the baked
`esmfold2_predict` command inside the Apptainer image.

## AF3 JSON

AlphaFold3 JSON is the preferred input format for complexes because it can
represent protein, DNA, RNA, ligands, residue modifications, seeds, and covalent
bonds.

```bash
apptainer exec --nv esmfold2.sif esmfold2_predict \
  --model /net/scratch/magnusb/git/ESMFold2/models/ESMFold2 \
  --ccd-cache /net/scratch/magnusb/git/ESMFold2/models/ESMFold2 \
  --input examples/af3_inputs/hf_hhai_1mht.json \
  --output examples/run_outputs/hf_hhai_1mht_full_metrics/1mht_pred.cif \
  --metrics-json examples/run_outputs/hf_hhai_1mht_full_metrics/metrics.json \
  --full-metrics
```

Inspect a syntax example containing protein, DNA, RNA, a CCD ligand, and a
SMILES ligand without loading the model:

```bash
apptainer exec esmfold2.sif esmfold2_predict \
  --input examples/af3_inputs/all_types_smoke.json \
  --list-inputs
```

The HhaI guide JSON uses AF3 1-based `basePosition` values and converts them to
the zero-based `Modification(position=...)` values expected by ESMFold2.

## PDB Or mmCIF

Structure files are parsed with AtomWorks, split into `ProteinInput`,
`DNAInput`, `RNAInput`, and `LigandInput`, then predicted as a new ESMFold2
structure.

```bash
apptainer exec --nv esmfold2.sif esmfold2_predict \
  --input input.cif \
  --output prediction_from_cif.cif \
  --metrics-json prediction_from_cif.metrics.json
```

List parsed entities without loading the model:

```bash
apptainer exec esmfold2.sif esmfold2_predict \
  --input input.cif \
  --list-inputs
```

## Batch

Batch mode scans `.json`, `.pdb`, `.cif`, and `.mmcif` files and loads the model
once.

```bash
apptainer exec --nv esmfold2.sif esmfold2_predict \
  --input-dir examples/af3_inputs \
  --output-dir examples/run_outputs/batch_af3 \
  --metrics-json examples/run_outputs/batch_af3/metrics.json
```

Add `--recursive` for nested folders. Add `--list-inputs` to inspect the batch
without loading the model.

Without extra flags the runner uses the installed ESMFold2 builder defaults:
`num_loops=3`, `num_sampling_steps=200`, and `num_diffusion_samples=1`.

## Metrics

Metrics JSON files contain `schema_version` and a `predictions` array. Each
prediction records the output path, chain ids and lengths, metadata, pLDDT
summary, pTM, and iPTM.

Add `--include-plddt` to include per-residue pLDDT values.

Every prediction writes AF3-style basic confidence sidecars next to the CIF:
`*_summary_confidences.json` and `*_confidences.json`. `--metrics-json` writes a
small aggregate JSON index.

Add `--full-metrics` to save every metric-like field exposed by the decoded
ESMFold2 result to `*_full_metrics.pkl`, including pLDDT, PAE, distogram logits,
pair-chain iPTM, residue indices, and entity ids when present. Full metrics are
pickled so the JSON files stay small.

## GPU Architecture Tests

Submit one Slurm job for each configured architecture case:

```bash
examples/run_gpu_arch_tests.sh
```

The script covers Ada 4000, Blackwell B4000/B6000/B6000Q, L40, L40S, A100, and
A6000. Outputs are written under `examples/gpu_test_outputs/<case>/`.

By default the script uses the normal model defaults. Set
`ESMFOLD2_GPU_TEST_USE_MODEL_DEFAULTS=0` only when you intentionally want a quick
CUDA smoke test rather than meaningful structures.

The script runs the repo SIF directly with normal `apptainer exec --nv`.
When `/net/scratch/magnusb/git/ESMFold2/models/ESMFold2` exists, it uses that
host-side model path automatically instead of reading weights from inside the
SIF. Override with `ESMFOLD2_GPU_TEST_MODEL` and
`ESMFOLD2_GPU_TEST_CCD_CACHE`.
