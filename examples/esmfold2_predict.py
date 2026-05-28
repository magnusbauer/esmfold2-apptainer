#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from esm.models.esmfold2 import ESMFold2InputBuilder

from _esmfold2_common import DEFAULT_CCD_CACHE, DEFAULT_MODEL
from _esmfold2_common import DEFAULT_NUM_DIFFUSION_SAMPLES
from _esmfold2_common import DEFAULT_NUM_LOOPS
from _esmfold2_common import DEFAULT_NUM_SAMPLING_STEPS
from _esmfold2_common import load_model
from _esmfold2_common import result_records_from_items
from _esmfold2_common import write_metrics_json
from _esmfold2_inputs import PreparedJob
from _esmfold2_inputs import discover_input_paths
from _esmfold2_inputs import read_input_job
from _esmfold2_inputs import resolve_bonded_atom_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ESMFold2 from AlphaFold3 JSON or AtomWorks-loaded PDB/mmCIF inputs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Single .json, .pdb, .cif, or .mmcif input.")
    source.add_argument("--input-dir", type=Path, help="Folder of .json, .pdb, .cif, or .mmcif inputs.")
    parser.add_argument("--format", default="auto", choices=["auto", "af3json", "pdb", "cif"])
    parser.add_argument("--output", type=Path, help="Output mmCIF path for --input.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for --input-dir. Defaults to <input-dir>/esmfold2_outputs.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --input-dir.")
    parser.add_argument(
        "--list-inputs",
        action="store_true",
        help="Parse and list inputs without loading the model.",
    )
    parser.add_argument("--metrics-json", type=Path, help="Optional path to write prediction metrics as JSON.")
    parser.add_argument(
        "--include-plddt",
        action="store_true",
        help="Include pLDDT values in confidence JSON output.",
    )
    parser.add_argument(
        "--full-metrics",
        action="store_true",
        help="Write all exposed metric tensors to *_full_metrics.pkl next to each CIF. Basic JSON stays small.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id or local model directory.")
    parser.add_argument("--ccd-cache", default=DEFAULT_CCD_CACHE, help="Directory containing ccd.pkl.")
    parser.add_argument("--esmc-precision", default="bf16", choices=["bf16", "fp32", "fp8"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-loops", type=int, default=DEFAULT_NUM_LOOPS)
    parser.add_argument(
        "--num-sampling-steps",
        "--num-diffusion-steps",
        dest="num_sampling_steps",
        type=int,
        default=DEFAULT_NUM_SAMPLING_STEPS,
        help=(
            "Diffusion sampling steps. --num-diffusion-steps is an alias. "
            f"Default: {DEFAULT_NUM_SAMPLING_STEPS}."
        ),
    )
    parser.add_argument("--num-diffusion-samples", type=int, default=DEFAULT_NUM_DIFFUSION_SAMPLES)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override AF3 modelSeeds. Structure inputs default to seed 0.",
    )
    args = parser.parse_args()
    if args.input and not args.list_inputs and args.output is None:
        parser.error("--output is required with --input unless --list-inputs is used")
    if args.input_dir and args.output is not None:
        parser.error("--output is only valid with --input; use --output-dir for batch runs")
    return args


def load_jobs(args: argparse.Namespace) -> tuple[list[PreparedJob], Path | None]:
    if args.input:
        return [read_input_job(args.input, input_format=args.format)], None

    output_dir = args.output_dir or args.input_dir / "esmfold2_outputs"
    paths = discover_input_paths(args.input_dir, output_dir, recursive=args.recursive)
    if not paths:
        raise ValueError(f"{args.input_dir} does not contain .json, .pdb, .cif, or .mmcif inputs")
    jobs = [read_input_job(path, input_format=args.format) for path in paths]
    seen: set[str] = set()
    for job in jobs:
        original = job.job_id
        suffix = 2
        while job.job_id in seen:
            job.job_id = f"{original}_{suffix}"
            job.complex_id = job.job_id
            suffix += 1
        seen.add(job.job_id)
    return jobs, output_dir


def finalize_job(job: PreparedJob, ccd_cache: str) -> PreparedJob:
    job.structure_input = resolve_bonded_atom_pairs(
        job.structure_input,
        job.bonded_atom_pairs,
        ccd_cache=ccd_cache,
    )
    if job.bonded_atom_pairs:
        job.metadata["bonded_atom_pair_count"] = len(job.bonded_atom_pairs)
    return job


def seeds_for_job(job: PreparedJob, seed_override: int | None) -> list[int]:
    if seed_override is not None:
        return [seed_override]
    return job.seeds or [0]


def base_output_for_job(job: PreparedJob, args: argparse.Namespace, output_dir: Path | None) -> Path:
    if args.output is not None:
        return args.output
    if output_dir is None:
        raise ValueError(f"No output directory for {job.job_id}")
    return output_dir / f"{job.job_id}.cif"


def output_for_seed(base_output: Path, seed: int, seed_count: int) -> Path:
    if seed_count == 1:
        return base_output
    return base_output.with_name(f"{base_output.stem}.seed_{seed}{base_output.suffix}")


def print_job(job: PreparedJob, seeds: list[int], output: Path | None) -> None:
    chains = []
    for chain in job.chain_records:
        label = f"{chain['id']}:{chain['type']}"
        if "length" in chain:
            label += f":{chain['length']}"
        elif "ccd" in chain:
            label += f":{','.join(chain['ccd'])}"
        chains.append(label)
    output_text = str(output) if output is not None else ""
    print(
        f"{job.job_id}\t{job.input_format}\t{job.input_path}\t{output_text}\t"
        f"seeds={','.join(str(seed) for seed in seeds)}\tchains={';'.join(chains)}"
    )


def main() -> None:
    args = parse_args()
    jobs, output_dir = load_jobs(args)
    finalized_jobs = [finalize_job(job, args.ccd_cache) for job in jobs]

    if args.list_inputs:
        for job in finalized_jobs:
            output = None
            if args.output is not None or output_dir is not None:
                output = base_output_for_job(job, args, output_dir)
            print_job(job, seeds_for_job(job, args.seed), output)
        return

    output_dir.mkdir(parents=True, exist_ok=True) if output_dir is not None else None
    print(f"Loading model once for {len(finalized_jobs)} job(s)", flush=True)
    model = load_model(args.model, args.esmc_precision, args.device)
    builder = ESMFold2InputBuilder(ccd_cache=args.ccd_cache)

    all_records = []
    for job in finalized_jobs:
        base_output = base_output_for_job(job, args, output_dir)
        seeds = seeds_for_job(job, args.seed)
        for seed in seeds:
            output = output_for_seed(base_output, seed, len(seeds))
            metadata = dict(job.metadata)
            metadata["job_id"] = job.job_id
            metadata["seed"] = seed
            print(f"Running {job.job_id} seed={seed}: {job.input_path} -> {output}", flush=True)
            result = builder.fold(
                model,
                job.structure_input,
                num_loops=args.num_loops,
                num_sampling_steps=args.num_sampling_steps,
                num_diffusion_samples=args.num_diffusion_samples,
                seed=seed,
                complex_id=job.complex_id,
            )
            records = result_records_from_items(
                result,
                output=str(output),
                chain_records=job.chain_records,
                complex_id=job.complex_id,
                include_plddt=args.include_plddt,
                full_metrics=args.full_metrics,
                metadata=metadata,
            )
            all_records.extend(records)

    if args.metrics_json:
        write_metrics_json(args.metrics_json, all_records)


if __name__ == "__main__":
    main()
