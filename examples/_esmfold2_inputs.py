from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Any

from esm.models.esmfold2 import (
    CovalentBond,
    DNAInput,
    LigandInput,
    Modification,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)
from esm.models.esmfold2.conformers import load_ccd
from esm.models.esmfold2.constants import DNA_1TO3, PROTEIN_1TO3, RNA_1TO3
from esm.models.esmfold2.prepare_input import build_chains_from_input
from esm.models.esmfold2.processor import clean_esmfold2_input
from esm.models.esmfold2.types import MSA


AF3_JSON_SUFFIXES = {".json"}
PDB_SUFFIXES = {".pdb"}
CIF_SUFFIXES = {".cif", ".mmcif"}
SUPPORTED_SUFFIXES = AF3_JSON_SUFFIXES | PDB_SUFFIXES | CIF_SUFFIXES
STANDARD_PROTEIN_CODES = set(PROTEIN_1TO3.values())
STANDARD_DNA_CODES = set(DNA_1TO3.values())
STANDARD_RNA_CODES = set(RNA_1TO3.values())


@dataclass
class PreparedJob:
    job_id: str
    complex_id: str
    input_path: Path
    input_format: str
    structure_input: StructurePredictionInput
    chain_records: list[dict[str, Any]]
    metadata: dict[str, Any]
    seeds: list[int]
    bonded_atom_pairs: list[Any] | None = None


def sanitize_job_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return sanitized or "prediction"


def infer_input_format(path: Path, requested: str = "auto") -> str:
    requested = requested.lower()
    if requested != "auto":
        if requested not in {"af3json", "pdb", "cif"}:
            raise ValueError("--format must be auto, af3json, pdb, or cif")
        return requested
    suffix = path.suffix.lower()
    if suffix in AF3_JSON_SUFFIXES:
        return "af3json"
    if suffix in PDB_SUFFIXES:
        return "pdb"
    if suffix in CIF_SUFFIXES:
        return "cif"
    raise ValueError(f"Unsupported input suffix for {path}")


def discover_input_paths(input_dir: Path, output_dir: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths: list[Path] = []
    output_resolved = output_dir.resolve()
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            paths.append(path)
    return paths


def _as_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _as_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _chain_ids(value: Any, context: str) -> str | list[str]:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{context}.id must not be empty")
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return list(value)
    raise ValueError(f"{context}.id must be a non-empty string or list of strings")


def _iter_chain_ids(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _clean_sequence(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context}.sequence must be a string")
    sequence = "".join(value.split()).upper().replace("*", "")
    if not sequence:
        raise ValueError(f"{context}.sequence must not be empty")
    if "|" in sequence or ":" in sequence:
        raise ValueError(f"{context}.sequence must use separate AF3 entries instead of chain separators")
    return sequence


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string when provided")
    return value


def _reject_non_empty(mapping: dict[str, Any], key: str, context: str, reason: str) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if value in (None, "", []):
        return
    raise ValueError(f"{context}.{key} is not supported: {reason}")


def _one_based_position(value: Any, context: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a 1-based positive integer")
    return value - 1


def _protein_modifications(items: Any, sequence_length: int, context: str) -> list[Modification] | None:
    if items in (None, []):
        return None
    mods: list[Modification] = []
    for index, item in enumerate(_as_list(items, f"{context}.modifications")):
        item = _as_dict(item, f"{context}.modifications[{index}]")
        ccd = item.get("ptmType")
        if not isinstance(ccd, str) or not ccd:
            raise ValueError(f"{context}.modifications[{index}].ptmType must be a CCD string")
        position = _one_based_position(item.get("ptmPosition"), f"{context}.modifications[{index}].ptmPosition")
        if position >= sequence_length:
            raise ValueError(f"{context}.modifications[{index}] position exceeds sequence length")
        mods.append(Modification(position=position, ccd=ccd))
    return mods


def _nucleotide_modifications(items: Any, sequence_length: int, context: str) -> list[Modification] | None:
    if items in (None, []):
        return None
    mods: list[Modification] = []
    for index, item in enumerate(_as_list(items, f"{context}.modifications")):
        item = _as_dict(item, f"{context}.modifications[{index}]")
        ccd = item.get("modificationType")
        if not isinstance(ccd, str) or not ccd:
            raise ValueError(f"{context}.modifications[{index}].modificationType must be a CCD string")
        position = _one_based_position(item.get("basePosition"), f"{context}.modifications[{index}].basePosition")
        if position >= sequence_length:
            raise ValueError(f"{context}.modifications[{index}] position exceeds sequence length")
        mods.append(Modification(position=position, ccd=ccd))
    return mods


def _load_inline_or_path_msa(entry: dict[str, Any], input_dir: Path, context: str) -> MSA | None:
    inline = _optional_string(entry.get("unpairedMsa"), f"{context}.unpairedMsa")
    path_value = _optional_string(entry.get("unpairedMsaPath"), f"{context}.unpairedMsaPath")
    if inline and path_value:
        raise ValueError(f"{context} cannot provide both unpairedMsa and unpairedMsaPath")
    if inline:
        return MSA.from_a3m(StringIO(inline), remove_insertions=True)
    if path_value:
        msa_path = Path(path_value)
        if not msa_path.is_absolute():
            msa_path = input_dir / msa_path
        if msa_path.suffix.lower() not in {".a3m", ".fa", ".faa", ".fas", ".fasta"}:
            raise ValueError(f"{context}.unpairedMsaPath must point to an uncompressed A3M/FASTA file")
        return MSA.from_a3m(msa_path, remove_insertions=True)
    return None


def _parse_protein(entry: dict[str, Any], input_dir: Path, context: str) -> ProteinInput:
    chain_id = _chain_ids(entry.get("id"), context)
    sequence = _clean_sequence(entry.get("sequence"), context)
    _reject_non_empty(entry, "pairedMsa", context, "ESMFold2 examples only accept unpaired protein MSA for now")
    _reject_non_empty(entry, "pairedMsaPath", context, "ESMFold2 examples only accept unpaired protein MSA for now")
    _reject_non_empty(entry, "templates", context, "templates are not passed to ESMFold2")
    return ProteinInput(
        id=chain_id,
        sequence=sequence,
        modifications=_protein_modifications(entry.get("modifications"), len(sequence), context),
        msa=_load_inline_or_path_msa(entry, input_dir, context),
    )


def _parse_dna(entry: dict[str, Any], context: str) -> DNAInput:
    chain_id = _chain_ids(entry.get("id"), context)
    sequence = _clean_sequence(entry.get("sequence"), context)
    return DNAInput(
        id=chain_id,
        sequence=sequence,
        modifications=_nucleotide_modifications(entry.get("modifications"), len(sequence), context),
    )


def _parse_rna(entry: dict[str, Any], context: str) -> RNAInput:
    chain_id = _chain_ids(entry.get("id"), context)
    sequence = _clean_sequence(entry.get("sequence"), context)
    _reject_non_empty(entry, "unpairedMsa", context, "RNAInput does not accept MSA")
    _reject_non_empty(entry, "unpairedMsaPath", context, "RNAInput does not accept MSA")
    return RNAInput(
        id=chain_id,
        sequence=sequence,
        modifications=_nucleotide_modifications(entry.get("modifications"), len(sequence), context),
    )


def _parse_ligand(entry: dict[str, Any], context: str) -> LigandInput:
    chain_id = _chain_ids(entry.get("id"), context)
    smiles = _optional_string(entry.get("smiles"), f"{context}.smiles")
    ccd_codes = entry.get("ccdCodes")
    has_smiles = smiles not in (None, "")
    has_ccd = ccd_codes not in (None, [])
    if has_smiles == has_ccd:
        raise ValueError(f"{context} must provide exactly one of smiles or ccdCodes")
    if has_ccd:
        if not isinstance(ccd_codes, list) or not ccd_codes or not all(isinstance(code, str) and code for code in ccd_codes):
            raise ValueError(f"{context}.ccdCodes must be a non-empty list of CCD strings")
        return LigandInput(id=chain_id, ccd=list(ccd_codes))
    return LigandInput(id=chain_id, smiles=smiles)


def _sequence_record(entry: ProteinInput | DNAInput | RNAInput | LigandInput, kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chain_id in _iter_chain_ids(entry.id):
        record: dict[str, Any] = {"id": chain_id, "type": kind}
        if isinstance(entry, (ProteinInput, DNAInput, RNAInput)):
            record["length"] = len(entry.sequence)
            if entry.modifications:
                record["modifications"] = [
                    {"position": mod.position + 1, "esm_position": mod.position, "ccd": mod.ccd}
                    for mod in entry.modifications
                ]
            if isinstance(entry, ProteinInput) and entry.msa is not None:
                record["msa_depth"] = entry.msa.depth
        elif isinstance(entry, LigandInput):
            if entry.ccd is not None:
                record["ccd"] = entry.ccd
            if entry.smiles is not None:
                record["smiles"] = entry.smiles
        records.append(record)
    return records


def chain_records_from_input(input_spec: StructurePredictionInput) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in input_spec.sequences:
        if isinstance(entry, ProteinInput):
            records.extend(_sequence_record(entry, "protein"))
        elif isinstance(entry, DNAInput):
            records.extend(_sequence_record(entry, "dna"))
        elif isinstance(entry, RNAInput):
            records.extend(_sequence_record(entry, "rna"))
        elif isinstance(entry, LigandInput):
            records.extend(_sequence_record(entry, "ligand"))
    return records


def _parse_af3_sequences(data: dict[str, Any], input_dir: Path) -> list[ProteinInput | DNAInput | RNAInput | LigandInput]:
    sequences = _as_list(data.get("sequences"), "sequences")
    if not sequences:
        raise ValueError("sequences must not be empty")
    parsed: list[ProteinInput | DNAInput | RNAInput | LigandInput] = []
    for index, wrapper in enumerate(sequences):
        wrapper = _as_dict(wrapper, f"sequences[{index}]")
        keys = [key for key in ("protein", "dna", "rna", "ligand") if key in wrapper]
        if len(keys) != 1:
            raise ValueError(f"sequences[{index}] must contain exactly one of protein, dna, rna, or ligand")
        key = keys[0]
        entry = _as_dict(wrapper[key], f"sequences[{index}].{key}")
        context = f"sequences[{index}].{key}"
        if key == "protein":
            parsed.append(_parse_protein(entry, input_dir, context))
        elif key == "dna":
            parsed.append(_parse_dna(entry, context))
        elif key == "rna":
            parsed.append(_parse_rna(entry, context))
        elif key == "ligand":
            parsed.append(_parse_ligand(entry, context))
    return parsed


def _af3_seeds(data: dict[str, Any]) -> list[int]:
    seeds = data.get("modelSeeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("AF3 JSON modelSeeds must be a non-empty list")
    normalized: list[int] = []
    for seed in seeds:
        if not isinstance(seed, int):
            raise ValueError("AF3 JSON modelSeeds must contain integers")
        normalized.append(seed)
    return normalized


def read_af3_json(path: Path) -> PreparedJob:
    with open(path) as handle:
        data = _as_dict(json.load(handle), str(path))

    dialect = data.get("dialect")
    if dialect not in (None, "alphafold3"):
        raise ValueError(f"{path} has unsupported dialect {dialect!r}; expected alphafold3")
    version = data.get("version")
    if version is not None and (not isinstance(version, int) or version < 1 or version > 4):
        raise ValueError(f"{path} has unsupported AF3 input version {version!r}; expected 1 through 4")
    _reject_non_empty(data, "userCCD", str(path), "custom CCD injection into ESMFold2 ccd.pkl is not implemented")
    _reject_non_empty(data, "userCCDPath", str(path), "custom CCD injection into ESMFold2 ccd.pkl is not implemented")

    input_spec = StructurePredictionInput(sequences=_parse_af3_sequences(data, path.parent))
    name = data.get("name") or path.stem
    if not isinstance(name, str):
        raise ValueError(f"{path}.name must be a string when provided")
    bonded_atom_pairs = data.get("bondedAtomPairs")
    if bonded_atom_pairs in (None, []):
        bonded_atom_pairs = None
    elif not isinstance(bonded_atom_pairs, list):
        raise ValueError("bondedAtomPairs must be a list when provided")

    return PreparedJob(
        job_id=sanitize_job_id(path.stem),
        complex_id=sanitize_job_id(name),
        input_path=path,
        input_format="af3json",
        structure_input=input_spec,
        chain_records=chain_records_from_input(input_spec),
        metadata={
            "input_format": "af3json",
            "input": str(path),
            "af3_name": name,
            "af3_version": version,
            "af3_dialect": dialect,
        },
        seeds=_af3_seeds(data),
        bonded_atom_pairs=bonded_atom_pairs,
    )


def _smiles_chain_ids(input_spec: StructurePredictionInput) -> set[str]:
    ids: set[str] = set()
    for entry in input_spec.sequences:
        if isinstance(entry, LigandInput) and entry.smiles is not None:
            ids.update(_iter_chain_ids(entry.id))
    return ids


def _parse_atom_ref(value: Any, context: str) -> tuple[str, int, str]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must be [chain_id, residue_index, atom_name]")
    chain_id, residue_index, atom_name = value
    if not isinstance(chain_id, str) or not chain_id:
        raise ValueError(f"{context}[0] chain_id must be a non-empty string")
    if not isinstance(residue_index, int) or residue_index < 1:
        raise ValueError(f"{context}[1] residue_index must be a 1-based positive integer")
    if not isinstance(atom_name, str) or not atom_name:
        raise ValueError(f"{context}[2] atom_name must be a non-empty string")
    return chain_id, residue_index - 1, atom_name


def resolve_bonded_atom_pairs(
    input_spec: StructurePredictionInput,
    bonded_atom_pairs: list[Any] | None,
    *,
    ccd_cache: str | Path,
) -> StructurePredictionInput:
    if not bonded_atom_pairs:
        return input_spec

    smiles_ids = _smiles_chain_ids(input_spec)
    load_ccd(Path(ccd_cache))
    cleaned = clean_esmfold2_input(input_spec)
    chains, tokens, atoms = build_chains_from_input(cleaned)
    chain_by_id = {chain.chain_id: chain for chain in chains}
    residue_atoms: dict[tuple[int, int], list[Any]] = {}
    for atom in atoms:
        if not atom.is_valid or atom.token_index >= len(tokens):
            continue
        token = tokens[atom.token_index]
        residue_atoms.setdefault((token.asym_id, token.residue_index), []).append(atom)

    def resolve(ref: Any, context: str) -> tuple[str, int, int]:
        chain_id, res_idx, atom_name = _parse_atom_ref(ref, context)
        if chain_id in smiles_ids:
            raise ValueError(f"{context} references SMILES ligand chain {chain_id}; AF3 atom names cannot be mapped safely")
        chain = chain_by_id.get(chain_id)
        if chain is None:
            raise ValueError(f"{context} references unknown chain {chain_id!r}")
        atoms_for_residue = residue_atoms.get((chain.asym_id, res_idx), [])
        matches = [idx for idx, atom in enumerate(atoms_for_residue) if atom.name == atom_name]
        if len(matches) != 1:
            raise ValueError(
                f"{context} atom {atom_name!r} on chain {chain_id} residue {res_idx + 1} "
                f"resolved to {len(matches)} atoms"
            )
        return chain_id, res_idx, matches[0]

    covalent_bonds: list[CovalentBond] = []
    for index, pair in enumerate(bonded_atom_pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"bondedAtomPairs[{index}] must contain exactly two atom refs")
        chain_id1, res_idx1, atom_idx1 = resolve(pair[0], f"bondedAtomPairs[{index}][0]")
        chain_id2, res_idx2, atom_idx2 = resolve(pair[1], f"bondedAtomPairs[{index}][1]")
        covalent_bonds.append(
            CovalentBond(
                chain_id1=chain_id1,
                res_idx1=res_idx1,
                atom_idx1=atom_idx1,
                chain_id2=chain_id2,
                res_idx2=res_idx2,
                atom_idx2=atom_idx2,
            )
        )

    return replace(input_spec, covalent_bonds=covalent_bonds)


def read_structure_with_atomworks(path: Path, input_format: str) -> PreparedJob:
    try:
        from atomworks.io import parse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AtomWorks is required for PDB/mmCIF input. Rebuild the image with atomworks==2.2.0."
        ) from exc

    result = parse(
        str(path),
        file_type=None if input_format == "auto" else input_format,
        ccd_mirror_path=None,
        add_missing_atoms=False,
        build_assembly=None,
        remove_waters=True,
    )
    return _atomworks_result_to_job(path, input_format, result)


def _get_attr_or_key(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _iter_records(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _chain_items(value: Any) -> list[tuple[str | None, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [(str(key), item) for key, item in value.items()]
    if isinstance(value, list):
        return [(None, item) for item in value]
    if isinstance(value, tuple):
        return [(None, item) for item in value]
    return []


def _bool_method(value: Any, name: str) -> bool:
    method = getattr(value, name, None)
    if callable(method):
        return bool(method())
    return False


def _residue_names(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in list(value)]


def _modifications_from_residue_names(
    res_names: list[str],
    sequence_length: int,
    standard_codes: set[str],
) -> list[Modification] | None:
    mods: list[Modification] = []
    for position, residue_name in enumerate(res_names[:sequence_length]):
        if residue_name and residue_name not in standard_codes and residue_name != "UNK":
            mods.append(Modification(position=position, ccd=residue_name))
    return mods or None


def _atomworks_result_to_job(path: Path, input_format: str, result: Any) -> PreparedJob:
    chain_info = _get_attr_or_key(result, "chain_info", "chains", "polymer_chains")
    sequences: list[ProteinInput | DNAInput | RNAInput | LigandInput] = []

    for index, (dict_chain_id, chain) in enumerate(_chain_items(chain_info)):
        chain_id = (
            dict_chain_id
            or _get_attr_or_key(chain, "chain_id", "asym_id", "auth_asym_id", "id")
            or chr(ord("A") + index)
        )
        chain_id = str(chain_id)
        chain_type = _get_attr_or_key(chain, "chain_type", "molecule_type", "polymer_type", "entity_type", "type")
        mol_type = str(chain_type or "").lower()
        sequence = _get_attr_or_key(chain, "sequence", "canonical_sequence", "processed_entity_canonical_sequence")
        res_names = _residue_names(_get_attr_or_key(chain, "res_name", "residue_names"))

        if _bool_method(chain_type, "is_non_polymer") or "non-polymer" in mol_type:
            ccd_codes = [name for name in res_names if name and name not in {"HOH", "WAT"}]
            if not ccd_codes:
                raise ValueError(f"AtomWorks non-polymer chain {chain_id} in {path} does not expose CCD residue names")
            sequences.append(LigandInput(id=chain_id, ccd=ccd_codes))
            continue

        if sequence is None:
            raise ValueError(f"AtomWorks did not expose a sequence for chain {chain_id} in {path}")
        sequence = _clean_sequence(str(sequence), f"{path}:{chain_id}")
        if "hybrid" in mol_type:
            raise ValueError(
                f"AtomWorks chain {chain_id} in {path} is a DNA/RNA hybrid chain; "
                "provide AF3 JSON with separate dna/rna entries instead"
            )

        if _bool_method(chain_type, "is_protein") or "protein" in mol_type or "polypeptide" in mol_type:
            sequences.append(
                ProteinInput(
                    id=chain_id,
                    sequence=sequence,
                    modifications=_modifications_from_residue_names(
                        res_names,
                        len(sequence),
                        STANDARD_PROTEIN_CODES,
                    ),
                )
            )
        elif "deoxyribo" in mol_type or "dna" in mol_type:
            sequences.append(
                DNAInput(
                    id=chain_id,
                    sequence=sequence,
                    modifications=_modifications_from_residue_names(
                        res_names,
                        len(sequence),
                        STANDARD_DNA_CODES,
                    ),
                )
            )
        elif "polyribonucleotide" in mol_type or ("ribo" in mol_type and "deoxyribo" not in mol_type) or "rna" in mol_type:
            sequences.append(
                RNAInput(
                    id=chain_id,
                    sequence=sequence,
                    modifications=_modifications_from_residue_names(
                        res_names,
                        len(sequence),
                        STANDARD_RNA_CODES,
                    ),
                )
            )
        else:
            raise ValueError(
                f"AtomWorks chain {chain_id} in {path} has unsupported or ambiguous molecule type {mol_type!r}"
            )

    if not sequences:
        raise ValueError(f"AtomWorks did not find protein, DNA, RNA, or ligand inputs in {path}")

    input_spec = StructurePredictionInput(sequences=sequences)
    return PreparedJob(
        job_id=sanitize_job_id(path.stem),
        complex_id=sanitize_job_id(path.stem),
        input_path=path,
        input_format=input_format,
        structure_input=input_spec,
        chain_records=chain_records_from_input(input_spec),
        metadata={
            "input_format": input_format,
            "input": str(path),
            "structure_loader": "atomworks",
            "note": "Coordinates are parsed only to recover entity inputs; ESMFold2 predicts a new structure.",
        },
        seeds=[0],
    )


def read_input_job(path: Path, *, input_format: str = "auto") -> PreparedJob:
    resolved = infer_input_format(path, input_format)
    if resolved == "af3json":
        return read_af3_json(path)
    return read_structure_with_atomworks(path, resolved)
