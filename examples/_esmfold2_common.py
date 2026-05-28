from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

DEFAULT_MODEL = os.environ.get("ESMFOLD2_MODEL_PATH", "/opt/esmfold2/models/ESMFold2")
DEFAULT_CCD_CACHE = os.environ.get(
    "ESMFOLD2_CCD_CACHE",
    os.environ.get("ESMFOLD2_CCD_PATH", "/opt/esmfold2/models/ESMFold2"),
)
DEFAULT_NUM_LOOPS = 3
DEFAULT_NUM_SAMPLING_STEPS = 200
DEFAULT_NUM_DIFFUSION_SAMPLES = 1
CHAIN_IDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def clean_protein_sequence(sequence: str) -> str:
    sequence = "".join(sequence.split()).upper().replace("*", "")
    if not sequence:
        raise ValueError("Encountered an empty sequence")
    if "|" in sequence:
        raise ValueError("Use separate FASTA records or structure chains instead of '|' separators")
    return sequence


def default_chain_id(index: int) -> str:
    if index < len(CHAIN_IDS):
        return CHAIN_IDS[index]
    return f"chain_{index + 1}"


def load_model(model: str, esmc_precision: str, device: str):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run the container with apptainer exec --nv.")

    loaded = ESMFold2Model.from_pretrained(
        model,
        esmc_precision=esmc_precision,
        local_files_only=Path(model).exists(),
    ).eval()
    if device == "cuda":
        loaded = loaded.cuda()
    return loaded


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _tensor_summary(value: Any) -> dict[str, float]:
    tensor = _as_cpu_tensor(value).float()
    if tensor.numel() == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.as_tensor(value)


def _tensor_shape(value: Any) -> list[int]:
    return list(_as_cpu_tensor(value).shape)


def _tensor_list(value: Any) -> list[float]:
    tensor = _as_cpu_tensor(value).float()
    return [float(item) for item in tensor.reshape(-1).tolist()]


def _tensor_nested_list(value: Any) -> Any:
    tensor = _as_cpu_tensor(value)
    if tensor.dtype.is_floating_point:
        tensor = tensor.float()
    return tensor.tolist()


def _tensor_numpy(value: Any) -> np.ndarray:
    tensor = _as_cpu_tensor(value)
    return tensor.numpy() if torch.is_tensor(tensor) else np.asarray(tensor)


def write_metrics_json(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(
            {
                "schema_version": 2,
                "format": "esmfold2_af3_style_basic_metrics",
                "predictions": records,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _sidecar_paths(model_path: Path) -> tuple[Path, Path, Path]:
    base = model_path.with_suffix("")
    return (
        base.with_name(f"{base.name}_summary_confidences.json"),
        base.with_name(f"{base.name}_confidences.json"),
        base.with_name(f"{base.name}_full_metrics.pkl"),
    )


def _chain_ids(chain_records: list[dict[str, Any]]) -> list[str]:
    return [str(chain["id"]) for chain in chain_records]


def _chain_pair_iptm_payload(value: Any | None, chain_records: list[dict[str, Any]]) -> dict[str, Any]:
    if value is None:
        return {}
    matrix = _tensor_nested_list(value)
    payload: dict[str, Any] = {"chain_pair_iptm": matrix}
    if isinstance(matrix, list):
        payload["chain_ptm"] = [
            float(row[index])
            for index, row in enumerate(matrix)
            if isinstance(row, list) and index < len(row)
        ]
        chain_iptm = []
        for index, row in enumerate(matrix):
            if not isinstance(row, list):
                continue
            off_diagonal = [
                float(value)
                for column, value in enumerate(row)
                if column != index
            ]
            chain_iptm.append(float(np.mean(off_diagonal)) if off_diagonal else None)
        payload["chain_iptm"] = chain_iptm
    payload["chain_ids"] = _chain_ids(chain_records)
    return payload


def _chain_token_lengths(chain_records: list[dict[str, Any]]) -> list[int]:
    lengths: list[int] = []
    for chain in chain_records:
        if "length" in chain:
            lengths.append(int(chain["length"]))
        elif "ccd" in chain:
            lengths.append(len(chain["ccd"]))
        else:
            lengths.append(1)
    return lengths


def _chain_pair_pae_min(value: Any | None, chain_records: list[dict[str, Any]]) -> list[list[float]] | None:
    if value is None:
        return None
    tensor = _as_cpu_tensor(value).float()
    if tensor.ndim != 2:
        return None
    lengths = _chain_token_lengths(chain_records)
    if sum(lengths) != tensor.shape[0] or tensor.shape[0] != tensor.shape[1]:
        return None
    starts = np.cumsum([0] + lengths)
    matrix: list[list[float]] = []
    for row_index, row_length in enumerate(lengths):
        row: list[float] = []
        row_start = int(starts[row_index])
        row_end = row_start + row_length
        for col_index, col_length in enumerate(lengths):
            col_start = int(starts[col_index])
            col_end = col_start + col_length
            block = tensor[row_start:row_end, col_start:col_end]
            row.append(float(block.min().item()) if block.numel() else float("nan"))
        matrix.append(row)
    return matrix


def _ranking_score(ptm: float | None, iptm: float | None) -> float | None:
    if ptm is None or iptm is None:
        return None
    return 0.8 * iptm + 0.2 * ptm


def summary_confidences_from_item(
    item: Any,
    *,
    chain_records: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    plddt_summary = _tensor_summary(item.plddt)
    ptm = _to_float(item.ptm)
    iptm = _to_float(item.iptm)
    payload: dict[str, Any] = {
        "ptm": ptm,
        "iptm": iptm,
        "ranking_score": _ranking_score(ptm, iptm),
        "plddt_mean": plddt_summary["mean"],
        "plddt_min": plddt_summary["min"],
        "plddt_max": plddt_summary["max"],
        "chain_ids": _chain_ids(chain_records),
    }
    payload.update(_chain_pair_iptm_payload(item.pair_chains_iptm, chain_records))
    chain_pair_pae_min = _chain_pair_pae_min(item.pae, chain_records)
    if chain_pair_pae_min is not None:
        payload["chain_pair_pae_min"] = chain_pair_pae_min
    if metadata:
        payload["metadata"] = metadata
    return payload


def confidences_from_item(
    item: Any,
    *,
    include_plddt: bool,
    chain_records: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = summary_confidences_from_item(
        item,
        chain_records=chain_records,
        metadata=metadata,
    )
    payload["schema_version"] = "esmfold2_af3_style_confidences_v1"
    payload["note"] = (
        "Large ESMFold2 metric tensors are stored in *_full_metrics.pkl when "
        "--full-metrics is used; this JSON intentionally contains only basic metrics."
    )
    if include_plddt:
        payload["plddt"] = _tensor_list(item.plddt)
    return payload


def full_metrics_from_item(
    item: Any,
    *,
    chain_records: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "esmfold2_full_metrics_pickle_v1",
        "chain_records": chain_records,
        "metadata": metadata or {},
        "metrics": {},
    }
    for name, attr in (
        ("plddt", "plddt"),
        ("pae", "pae"),
        ("distogram_logits", "distogram"),
        ("pair_chains_iptm", "pair_chains_iptm"),
        ("residue_index", "residue_index"),
        ("entity_id", "entity_id"),
        ("output_embedding_sequence", "output_embedding_sequence"),
        ("output_embedding_pair_pooled", "output_embedding_pair_pooled"),
        ("sae_features", "sae_features"),
    ):
        value = getattr(item, attr, None)
        if value is not None:
            payload["metrics"][name] = _tensor_numpy(value)
    return payload


def result_records_from_items(
    result: Any,
    *,
    output: str,
    chain_records: list[dict[str, Any]],
    complex_id: str,
    include_plddt: bool = False,
    full_metrics: bool = False,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    results = result if isinstance(result, list) else [result]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        if len(results) == 1:
            path = output_path
        else:
            path = output_path.with_name(f"{output_path.stem}.sample_{index + 1}{output_path.suffix}")
        with open(path, "w") as handle:
            handle.write(item.complex.to_mmcif())
        summary_path, confidences_path, full_metrics_path = _sidecar_paths(path)
        summary_confidences = summary_confidences_from_item(
            item,
            chain_records=chain_records,
            metadata=metadata,
        )
        confidences = confidences_from_item(
            item,
            include_plddt=include_plddt,
            chain_records=chain_records,
            metadata=metadata,
        )
        _write_json(summary_path, summary_confidences)
        _write_json(confidences_path, confidences)
        if full_metrics:
            with open(full_metrics_path, "wb") as handle:
                pickle.dump(
                    full_metrics_from_item(
                        item,
                        chain_records=chain_records,
                        metadata=metadata,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        record = {
            "complex_id": complex_id,
            "sample_index": index + 1,
            "output": str(path),
            "summary_confidences_json": str(summary_path),
            "confidences_json": str(confidences_path),
            "chains": chain_records,
            "metrics": summary_confidences,
        }
        if metadata:
            record["metadata"] = metadata
        if full_metrics:
            record["full_metrics_pickle"] = str(full_metrics_path)
        print(
            f"{path}: pLDDT mean={summary_confidences['plddt_mean']:.3f}, "
            f"pTM={summary_confidences['ptm']:.3f}, ipTM={summary_confidences['iptm']:.3f}",
            flush=True,
        )
        records.append(record)
    return records


def fold_protein_sequences_with_model(
    loaded_model,
    sequences: Iterable[tuple[str, str]],
    *,
    builder: ESMFold2InputBuilder | None = None,
    output: str,
    ccd_cache: str,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    seed: int | None,
    complex_id: str,
    include_plddt: bool = False,
    full_metrics: bool = False,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entries = [
        ProteinInput(id=chain_id, sequence=clean_protein_sequence(sequence))
        for chain_id, sequence in sequences
    ]
    if not entries:
        raise ValueError("No protein sequences were provided")

    spi = StructurePredictionInput(sequences=entries)
    if builder is None:
        builder = ESMFold2InputBuilder(ccd_cache=ccd_cache)
    result = builder.fold(
        loaded_model,
        spi,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
        complex_id=complex_id,
    )

    chain_records = [
        {"id": entry.id, "length": len(entry.sequence)}
        for entry in entries
    ]

    return result_records_from_items(
        result,
        output=output,
        chain_records=chain_records,
        complex_id=complex_id,
        include_plddt=include_plddt,
        full_metrics=full_metrics,
        metadata=metadata,
    )


def fold_protein_sequences(
    sequences: Iterable[tuple[str, str]],
    *,
    model: str,
    output: str,
    ccd_cache: str,
    esmc_precision: str,
    device: str,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    seed: int | None,
    complex_id: str,
    metrics_json: str | None = None,
    include_plddt: bool = False,
    full_metrics: bool = False,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    loaded_model = load_model(model, esmc_precision, device)
    records = fold_protein_sequences_with_model(
        loaded_model,
        sequences,
        output=output,
        ccd_cache=ccd_cache,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
        complex_id=complex_id,
        include_plddt=include_plddt,
        full_metrics=full_metrics,
        metadata=metadata,
    )
    if metrics_json:
        write_metrics_json(metrics_json, records)
    return records


def add_common_args(
    parser,
    *,
    include_output: bool = True,
    include_complex_id: bool = True,
) -> None:
    if include_output:
        parser.add_argument("--output", required=True, help="Output mmCIF path.")
    parser.add_argument("--metrics-json", help="Optional path to write prediction metrics as JSON.")
    parser.add_argument(
        "--include-plddt",
        action="store_true",
        help="Include pLDDT values in confidence JSON output.",
    )
    parser.add_argument(
        "--full-metrics",
        action="store_true",
        help=(
            "Write all exposed metric tensors to *_full_metrics.pkl next to each CIF. "
            "Basic JSON metrics stay small."
        ),
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
    )
    parser.add_argument("--num-diffusion-samples", type=int, default=DEFAULT_NUM_DIFFUSION_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    if include_complex_id:
        parser.add_argument("--complex-id", default="pred")
