"""
Modified HDF5 DataLoader for preprocessed radar pulse data from the UCL SEI Dataset

This module provides two dataloaders for the simple preprocessed HDF5 format:
1. SimplePulseDatasetPyTorch - For PyTorch models (2D data: time × channels)
2. AugmentedPulseDataset     - To enhance the base dataset with Augmentations for SNR evaluation and contrastive loss

The base (and by inheritance, the AugmentedPulseDataset) loaders support:
- Filtering:
    - Temporal (collect day) filtering
    - Environment (chamber or field) filtering
    - Temperature (hot or cold) filtering
    - Emitter (blade1, blade2...) filtering
    - Waveform (pulse or chirped pulse) filtering
- Flexible data loading strategies
- Loading of variable pulse sequence lengths
- Output of Metadata (RFSoC Sampling Freq)
- Output of Pulse Descriptor Words per pulse (Time-of-arrival, centre-frequency, SNR)

ADC selection is always done via adc_names - there is no separate ADC filter.
adc_names picks which ADCs are included; adc_mode controls how they combine:

When mode="contrastive", alternate views of the same pulse (via different ADCs) are
presented as a stacked tensor like this:
    - (n_adc, len(iq), 2) stacked tensor, one row per adc_names entry

When mode="single", each ADC's view of a physical pulse becomes its own separate
item - the dataset is simply len(adc_names) times longer, e.g. for 4000 matched
pulses across 2 ADCs, contrastive gives 4000 items shaped (2, len(iq), 2), while
single gives 8000 items shaped (len(iq), 2).

When contrastive mode is used with SNR augmentations, the tensor is presented as:
    - (n_adc * n_snrs, len(iq), 2)
"""

import bisect
import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from collections import defaultdict
import warnings
from .augmentations import DegradeToSNR, AddMagnitude, NormaliseByPeakMagnitude, Flatten

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. SimplePulseDatasetPyTorch will not work.")


class SimplePulseDatasetPyTorch(Dataset):
    """
    PyTorch Dataset for loading simple preprocessed HDF5 radar pulse data.

    Features:
    - Load from directory of HDF5 chunks (variable number of pulses per chunk)
    - Filter by day, environment, temp, emitter, or waveform
    - Optional: Load all data into memory for faster training
    - Lazy loading by default to minimize memory usage
    - Ability to limit number of pulses per emitter
    - Optionally return the full pulse_descriptor_words row and/or file-level
      metadata alongside each pulse (return_pdw / return_metadata)
    - Ability to return stacked pulses aligned by time-of-arrival in
      "contrastive" mode. Enabling multi-view training from multiple ADCs.

    No-ADC mode (adc_names=None, default): returns pulses in shape
    (len(iq), 2), one row per pulse, with no per-ADC awareness at all -
    every file matching the label filters is treated as one pool of pulses.

    Multi-ADC mode (adc_names=[...]) is always aligned by time-of-arrival,
    using the fixed TOA_FIELD and TOA_TOLERANCE constants (see module
    docstring for the matching algorithm and why TOA is used instead of
    position). This is the only way to select or combine specific ADCs -
    there is no separate ADC filter.

    `adc_mode` controls what __getitem__ hands back for each matched group:
        adc_mode="contrastive": __getitem__ returns the full stack,
            pulse shape (n_adc, len(iq), 2), one row per adc_names entry -
            all views of one physical pulse together.
            A view of a physical pulses is dropped if a single SNR in the group
            goes below "SNR_THRESHOLD_DB" reported SNR prior to augmentation.
        adc_mode="single" (default): __getitem__ returns one ADC's view at a
            time, pulse shape (len(iq), 2), same as no-ADC mode. Dataset
            length is multiplied by len(adc_names); labels gain an 'adc' key
            so you can tell which ADC a given item came from.
    """

    TOA_FIELD = "time_of_arrival"   # fixed - not a constructor parameter, alignment always uses this field
    TOA_TOLERANCE = 2e-6            # fixed - max |toa_a - toa_b| for two ADC readings to count as the same pulse
    SNR_FIELD = "snr"               # fixed - pulse_descriptor_words field checked against SNR_THRESHOLD_DB
    SNR_THRESHOLD_DB = 31           # fixed - a matched group is dropped entirely if ANY ADC's view is below this

    def __init__(
        self,
        hdf5_dir: str,

        day_filter: Optional[List[str]] = None,
        environment_filter: Optional[List[str]] = None,
        temp_filter: Optional[List[str]] = None,
        blade_filter: Optional[List[str]] = None,
        waveform_filter: Optional[List[str]] = None,
        adc_names: Optional[List[str]] = None,
        adc_mode: Literal["single", "contrastive"] = "single",

        verbose: bool = False,

        load_to_memory: bool = False,
        max_pulses_per_emitter: Optional[int] = None,

        return_pdw: bool = False,
        return_metadata: bool = False,
    ):
        """
        Args:
            *_filter: a list passed to whitelist devices and environment settings
                to be passed by the dataloader, enabling training on only specific,
                devices, and/or waveforms. By default "None" is passed.
            adc_names: the ONLY way to select or combine specific ADCs. If given,
                builds a multi-ADC-aligned index instead of a per-pulse index.
                Alignment is always by time-of-arrival
                (pulse_descriptor_words["time_of_arrival"]), nearest-TOA matching
                within the fixed TOA_TOLERANCE, per (day, environment, temp,
                emitter, pulse_type) acquisition. If omitted, no per-ADC
                distinction is made at all - every matching file's pulses are
                pooled together.
            adc_mode: only used when adc_names is given - see class docstring.
                "single" (default) flattens to one ADC view per item;
                "contrastive" stacks every adc_names entry together per item.
            verbose: if True, print per-acquisition alignment diagnostics
                (pulse counts per ADC, matched groups, dropped counts) when
                building a multi-ADC index.
            load_to_memory: Load pulses to memory, allowing faster training.
            max_pulses_per_emitter: Limit number of pulses used from each emitter/adc.
            return_pdw/medata: Return pulse descriptor words (ToA, power, freq, etc)
                or collect meta-data (RFSoC parameters), source file for pulse.

        """

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SimplePulseDatasetPyTorch")
        if adc_mode not in ("single", "contrastive"):
            raise ValueError(f"adc_mode must be 'single' or 'contrastive', got {adc_mode!r}")

        self.hdf5_dir = Path(hdf5_dir)

        self.day_filter = set(day_filter) if day_filter else None
        self.environment_filter = set(environment_filter) if environment_filter else None
        self.temp_filter = set(temp_filter) if temp_filter else None
        self.blade_filter = set(blade_filter) if blade_filter else None
        self.waveform_filter = set(waveform_filter) if waveform_filter else None
        self.adc_names = list(adc_names) if adc_names else None
        self.multi_adc = self.adc_names is not None
        self.adc_mode = adc_mode

        self.verbose = verbose
        self.alignment_stats: Dict[Tuple, Dict[str, Any]] = {}

        self.load_to_memory = load_to_memory
        self.max_pulses_per_emitter = max_pulses_per_emitter

        self.return_pdw = return_pdw
        self.return_metadata = return_metadata

        self.hdf5_files = sorted(self.hdf5_dir.rglob('*.h5'))
        if len(self.hdf5_files) == 0:
            raise ValueError(f"No .h5 files found in {hdf5_dir}")

        self.metadata: Optional[Dict[str, Any]] = None
        if self.return_metadata:
            with h5py.File(self.hdf5_files[0], 'r') as f:
                self.metadata = self._read_metadata(f)

        if self.multi_adc:
            self._build_multi_adc_index_by_toa(max_pulses_per_emitter)
        else:
            self._build_index(max_pulses_per_emitter)

        if self.load_to_memory:
            self._load_to_memory()

    @staticmethod
    def _decode_attr(x):
        return x.decode() if isinstance(x, bytes) else x

    @classmethod
    def _read_metadata(cls, f: h5py.File) -> Dict[str, Any]:
        """Generic read of every attribute under /metadata."""
        return {key: cls._decode_attr(value) for key, value in f["metadata"].attrs.items()}

    @classmethod
    def _read_pdw_row(cls, f: h5py.File, pulse_idx: int) -> Dict[str, Any]:
        """Generic read of every field under /pulse_descriptor_words for one pulse."""
        pdw_grp = f["pulse_descriptor_words"]
        row = {}
        for name, dataset in pdw_grp.items():
            row[name] = dataset[pulse_idx] if pulse_idx < dataset.shape[0] else None
        return row

    @staticmethod
    def _read_pdw_columns(f: h5py.File, fields: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Read whole columns from "pulse_descriptor_words"."""
        pdw_grp = f["pulse_descriptor_words"]
        out: Dict[str, Optional[np.ndarray]] = {}
        for name in fields:
            out[name] = pdw_grp[name][:] if name in pdw_grp else None
        return out

    @classmethod
    def _read_labels(cls, f: h5py.File) -> Dict[str, str]:
        labels_grp = f["labels"]
        return {
            "day": cls._decode_attr(labels_grp.attrs["day"]),
            "environment": cls._decode_attr(labels_grp.attrs["environment"]),
            "temp": cls._decode_attr(labels_grp.attrs["temp"]),
            "emitter": cls._decode_attr(labels_grp.attrs["emitter"]),
            "pulse_type": cls._decode_attr(labels_grp.attrs["pulse_type"]),
        }

    def _passes_label_filters(self, labels: Dict[str, str]) -> bool:
        if self.day_filter and labels["day"] not in self.day_filter:
            return False
        if self.environment_filter and labels["environment"] not in self.environment_filter:
            return False
        if self.temp_filter and labels["temp"] not in self.temp_filter:
            return False
        if self.blade_filter and labels["emitter"] not in self.blade_filter:
            return False
        if self.waveform_filter and labels["pulse_type"] not in self.waveform_filter:
            return False
        return True

    def _build_index(self, max_pulses_per_emitter: Optional[int] = None):
        """Build index mapping global pulse indices to file/pulse locations. (no-ADC mode)"""
        self.pulse_index = []
        emitter_counts = defaultdict(int)

        for file_idx, filepath in enumerate(self.hdf5_files):
            try:
                with h5py.File(filepath, 'r') as f:
                    source_file = self._decode_attr(f["metadata"].attrs["source_file"])

                    n_pulses = f['pulses'].shape[0]
                    labels = self._read_labels(f)

                    emitter_key = (labels["day"], labels["environment"], labels["temp"],
                                   labels["emitter"], labels["pulse_type"], source_file)

                    if max_pulses_per_emitter is not None and emitter_counts[emitter_key] >= max_pulses_per_emitter:
                        continue

                    for pulse_idx in range(n_pulses):
                        if max_pulses_per_emitter is not None and emitter_counts[emitter_key] >= max_pulses_per_emitter:
                            break
                        if not self._passes_label_filters(labels):
                            continue
                        pulse_labels = {**labels, "source_file": source_file}
                        self.pulse_index.append((file_idx, pulse_idx, pulse_labels))
                        emitter_counts[emitter_key] += 1

            except Exception as e:
                warnings.warn(f"Error reading {filepath.name}: {e}")
                continue

        print(f"Dataset initialized: {len(self.pulse_index)} pulses from {len(self.hdf5_files)} HDF5 files")

    def _scan_adc_files(self, need_columns: Optional[List[str]] = None):
        """
        Shared first pass for the multi-ADC index builder: walk every HDF5 file, work out
        which requested ADC (if any) it belongs to, apply label filters, and group results by
        (adc, acquisition_key). Returns:
            per_adc: {adc_name: {acq_key: [ per-file entries ]}}
                each entry is a dict with at least file_idx, n_pulses, and (if need_columns
                was given) a "columns" dict of the requested pdw columns for that file.
            acq_labels: {acq_key: labels dict}
            seen_identifiers: every source_file/filename stem seen, for diagnostics.
        """
        adc_set_lower = {a.lower() for a in self.adc_names}
        per_adc: Dict[str, Dict[Tuple, List[dict]]] = {adc: defaultdict(list) for adc in self.adc_names}
        acq_labels: Dict[Tuple, Dict[str, str]] = {}
        seen_identifiers = set()

        for file_idx, filepath in enumerate(self.hdf5_files):
            try:
                with h5py.File(filepath, 'r') as f:
                    source_file = self._decode_attr(f["metadata"].attrs["source_file"])
                    candidates = {Path(source_file).stem.lower(), filepath.stem.lower()}
                    seen_identifiers.update(candidates)

                    matched = candidates & adc_set_lower
                    if not matched:
                        continue
                    adc = next(a for a in self.adc_names if a.lower() in matched)

                    labels = self._read_labels(f)
                    if not self._passes_label_filters(labels):
                        continue

                    acq_key = (labels["day"], labels["environment"], labels["temp"],
                               labels["emitter"], labels["pulse_type"])
                    acq_labels[acq_key] = labels

                    entry = {"file_idx": file_idx, "n_pulses": f['pulses'].shape[0]}
                    if need_columns:
                        entry["columns"] = self._read_pdw_columns(f, need_columns)
                    per_adc[adc][acq_key].append(entry)

            except Exception as e:
                warnings.warn(f"Error reading {filepath.name}: {e}")
                continue

        return per_adc, acq_labels, seen_identifiers

    def _raise_if_missing_or_disjoint(self, per_adc, seen_identifiers):
        missing = [adc for adc in self.adc_names if not per_adc[adc]]
        if missing:
            preview = sorted(seen_identifiers)[:20]
            raise ValueError(
                f"No files matched ADC name(s) {missing} in {self.hdf5_dir} (matched against "
                f"each file's metadata source_file stem AND its own filename stem). "
                f"Identifiers actually seen in this directory: {preview}"
                f"{' ...' if len(seen_identifiers) > 20 else ''}. "
                f"Check adc_names against these, and that day/environment/temp/blade/waveform "
                f"filters aren't excluding every file for that ADC."
            )

        common_keys = set.intersection(*(set(per_adc[adc].keys()) for adc in self.adc_names))
        if not common_keys:
            raise ValueError(
                f"ADCs {self.adc_names} each matched files, but share no common "
                f"(day, environment, temp, emitter, pulse_type) acquisition after filtering. "
                f"Per-ADC acquisitions found: "
                f"{ {adc: sorted(per_adc[adc].keys()) for adc in self.adc_names} }"
            )
        return common_keys

    def _build_multi_adc_index_by_toa(self, max_pulses_per_emitter: Optional[int] = None):
        """
        TOA-based alignment (the only alignment mode). Index over physical pulses,
        matched across self.adc_names by nearest time-of-arrival rather than by position.
        This is because pulse_ids may become misaligned if one ADC drops a pulse.

        self.pulse_index is a list of dicts:
            {"labels": {...}, adc0: (file_idx, pulse_idx), adc1: (...), ...}

        Matching algorithm, per (day, environment, temp, emitter, pulse_type) acquisition:
          1. Sort every ADC's pulses (across however many files/chunks make up that ADC's
             recording of this acquisition) by TOA_FIELD, ascending.
          2. Pick the ADC with the fewest pulses in this acquisition as the base sequence.
          3. For each base pulse, binary-search each other ADC's sorted toa array for the
             closest value within TOA_TOLERANCE that hasn't already been claimed by an
             earlier base pulse (a "used" flag per other-ADC pulse enforces a 1:1 match, so
             two close-in-time base pulses can't both grab the same other-ADC pulse).
          4. A base pulse is kept only if every other ADC produced a match; otherwise it's
             dropped and counted (n_dropped_no_match, reported in alignment_stats and, if
             verbose=True, printed per acquisition).
          5. Once a group is fully matched across every requested ADC, its per-ADC SNR
             (pulse_descriptor_words[SNR_FIELD]) is checked against SNR_THRESHOLD_DB. If
             ANY ADC's view of that physical pulse is below SNR_THRESHOLD_DB, the entire
             group is dropped (dropped_low_snr, reported in alignment_stats and, if
             verbose=True, printed per acquisition). This applies uniformly regardless of
             adc_mode: adc_mode="single" is expanded from this same group index, so a
             physical pulse failing the check on any ADC contributes no views at all, from
             any ADC, in either mode.
        """
        need_columns = [self.TOA_FIELD, self.SNR_FIELD]
        per_adc_files, acq_labels, seen_identifiers = self._scan_adc_files(need_columns=need_columns)

        # Expand file-level entries into flat per-pulse dicts per (adc, acq_key).
        per_adc: Dict[str, Dict[Tuple, List[dict]]] = {adc: defaultdict(list) for adc in self.adc_names}
        for adc, by_key in per_adc_files.items():
            for acq_key, entries in by_key.items():
                for entry in entries:
                    toa_col = entry["columns"][self.TOA_FIELD]
                    snr_col = entry["columns"][self.SNR_FIELD]
                    for pulse_idx in range(entry["n_pulses"]):
                        per_adc[adc][acq_key].append({
                            "file_idx": entry["file_idx"],
                            "pulse_idx": pulse_idx,
                            "toa": float(toa_col[pulse_idx]),
                            "snr": float(snr_col[pulse_idx]),
                        })

        common_keys = self._raise_if_missing_or_disjoint(per_adc, seen_identifiers)

        self.pulse_index = []
        self.alignment_stats = {}

        for key in sorted(common_keys):
            for adc in self.adc_names:
                per_adc[adc][key].sort(key=lambda p: p["toa"])

            counts = {adc: len(per_adc[adc][key]) for adc in self.adc_names}

            if len(set(counts.values())) > 1:
                warnings.warn(
                    f"Pulse count mismatch across ADCs for {key}: {counts}. "
                    f"TOA alignment will match what it can within TOA_TOLERANCE="
                    f"{self.TOA_TOLERANCE} and drop the rest - see alignment_stats[{key}] "
                    f"for exact matched/dropped counts once the index is built."
                )

            base_adc = min(counts, key=counts.get)
            other_adcs = [a for a in self.adc_names if a != base_adc]

            toa_arrays = {adc: [p["toa"] for p in per_adc[adc][key]] for adc in other_adcs}
            used = {adc: [False] * counts[adc] for adc in other_adcs}

            matched_groups = []
            n_dropped_no_match = 0
            n_dropped_low_snr = 0

            for base_pulse in per_adc[base_adc][key]:
                base_toa = base_pulse["toa"]
                group = {base_adc: base_pulse}
                ok = True
                for adc in other_adcs:
                    arr = toa_arrays[adc]
                    i = bisect.bisect_left(arr, base_toa)
                    best_j, best_dist = None, None
                    for j in (i - 1, i, i + 1):
                        if 0 <= j < len(arr) and not used[adc][j]:
                            dist = abs(arr[j] - base_toa)
                            if dist <= self.TOA_TOLERANCE and (best_dist is None or dist < best_dist):
                                best_j, best_dist = j, dist
                    if best_j is None:
                        ok = False
                        break
                    group[adc] = per_adc[adc][key][best_j]
                    used[adc][best_j] = True
                if not ok:
                    n_dropped_no_match += 1
                    continue

                if any(group[a]["snr"] < self.SNR_THRESHOLD_DB for a in self.adc_names):
                    n_dropped_low_snr += 1
                    continue

                matched_groups.append(group)

            if max_pulses_per_emitter is not None:
                matched_groups = matched_groups[:max_pulses_per_emitter]

            labels = acq_labels[key]
            for group in matched_groups:
                self.pulse_index.append({
                    "labels": labels,
                    **{adc: (group[adc]["file_idx"], group[adc]["pulse_idx"]) for adc in self.adc_names},
                })

            self.alignment_stats[key] = {
                "counts": counts,
                "base_adc": base_adc,
                "matched": len(matched_groups),
                "dropped_no_toa_match": n_dropped_no_match,
                "dropped_low_snr": n_dropped_low_snr,
            }

            if self.verbose:
                print(
                    f"[align:toa] {key}: counts={counts} base={base_adc!r} tolerance={self.TOA_TOLERANCE} "
                    f"-> {len(matched_groups)} matched, {n_dropped_no_match} base pulses dropped "
                    f"(no ADC match within tolerance), {n_dropped_low_snr} groups dropped "
                    f"(an ADC's SNR was below {self.SNR_THRESHOLD_DB}dB)"
                )

        total_dropped_no_match = sum(s["dropped_no_toa_match"] for s in self.alignment_stats.values())
        total_dropped_low_snr = sum(s["dropped_low_snr"] for s in self.alignment_stats.values())
        print(
            f"Dataset initialized: {len(self.pulse_index)} pulse-groups x {len(self.adc_names)} ADCs "
            f"from {len(self.hdf5_files)} HDF5 files (aligned by time_of_arrival, "
            f"tolerance={self.TOA_TOLERANCE}, {total_dropped_no_match} pulses dropped for lack of a "
            f"cross-ADC match within tolerance, {total_dropped_low_snr} groups dropped for an ADC's "
            f"SNR being below {self.SNR_THRESHOLD_DB}dB)"
        )

    def _load_to_memory(self):
        """Load all filtered pulses (and pdw rows, if requested) into memory."""
        print("Loading all data to memory...")

        cache: Dict[Tuple[int, int], Tuple[Any, Any]] = {}
        by_file: Dict[int, set] = defaultdict(set)

        if self.multi_adc:
            for entry in self.pulse_index:
                for adc in self.adc_names:
                    by_file[entry[adc][0]].add(entry[adc][1])
        else:
            for file_idx, pulse_idx, _labels in self.pulse_index:
                by_file[file_idx].add(pulse_idx)

        for file_idx, pulse_idxs in by_file.items():
            filepath = self.hdf5_files[file_idx]
            with h5py.File(filepath, 'r') as f:
                for pulse_idx in pulse_idxs:
                    pulse = torch.from_numpy(f['pulses'][pulse_idx])
                    pdw = self._read_pdw_row(f, pulse_idx) if self.return_pdw else None
                    cache[(file_idx, pulse_idx)] = (pulse, pdw)

        if self.multi_adc:
            self._mem = [{adc: cache[entry[adc]] for adc in self.adc_names} for entry in self.pulse_index]
        else:
            self._mem = [cache[(file_idx, pulse_idx)] for file_idx, pulse_idx, _ in self.pulse_index]

        print(f"Loaded {len(self.pulse_index)} entries to memory")

    def __len__(self) -> int:
        if self.multi_adc and self.adc_mode == "single":
            return len(self.pulse_index) * len(self.adc_names)
        return len(self.pulse_index)

    def __getitem__(self, idx: int):
        """
        No-ADC mode returns:
            pulse:  (len(iq), 2) tensor [I, Q]
            labels: dict with 'day', 'environment', 'temp', 'emitter', 'pulse_type'
            pdw:    (only if return_pdw=True) dict of pdw fields for this pulse

        Multi-ADC mode, adc_mode="contrastive" returns:
            pulse:  (n_adc, len(iq), 2) tensor, one row per adc_names entry,
                    each row an independent digitisation of the SAME physical pulse
            labels: dict (no 'source_file' - ambiguous across ADCs)
            pdw:    (only if return_pdw=True) list of n_adc pdw dicts, in adc_names order

        Multi-ADC mode, adc_mode="single" (default) returns ONE ADC's view per call:
            pulse:  (len(iq), 2) tensor, same shape as no-ADC mode
            labels: dict as above, plus an 'adc' key naming which ADC this is
            pdw:    (only if return_pdw=True) a single dict (not a list)
        """
        if self.multi_adc and self.adc_mode == "single":
            n_adc = len(self.adc_names)
            group_idx, adc_idx = idx // n_adc, idx % n_adc
            adc = self.adc_names[adc_idx]
            entry = self.pulse_index[group_idx]

            if self.load_to_memory:
                pulse, pdw = self._mem[group_idx][adc]
            else:
                file_idx, pulse_idx = entry[adc]
                pulse, pdw = self._read_one(self.hdf5_files[file_idx], pulse_idx)

            pulse = pulse.reshape(-1, 2)
            labels = {**entry["labels"], "adc": adc}
            output = (pulse, labels)
            if self.return_pdw:
                output = output + (pdw,)
            return output

        elif self.multi_adc:  # adc_mode == "contrastive"
            entry = self.pulse_index[idx]
            views = self._mem[idx] if self.load_to_memory else {
                adc: self._read_one(self.hdf5_files[entry[adc][0]], entry[adc][1])
                for adc in self.adc_names
            }
            pulses = torch.stack([views[adc][0].reshape(-1, 2) for adc in self.adc_names], dim=0)
            output = (pulses, entry["labels"])
            if self.return_pdw:
                pdws = [views[adc][1] for adc in self.adc_names]
                output = output + (pdws,)
            return output

        else:
            file_idx = self.pulse_index[idx][0]
            if self.load_to_memory:
                pulse, pdw = self._mem[idx]
            else:
                _, pulse_idx, _ = self.pulse_index[idx]
                pulse, pdw = self._read_one(self.hdf5_files[file_idx], pulse_idx)
            labels = self.pulse_index[idx][2]
            pulse = pulse.reshape(-1, 2)

            output = (pulse, labels)
            if self.return_pdw:
                output = output + (pdw,)
            return output

    def _read_one(self, filepath: Path, pulse_idx: int):
        with h5py.File(filepath, 'r') as f:
            pulse = torch.from_numpy(f['pulses'][pulse_idx])
            pdw = self._read_pdw_row(f, pulse_idx) if self.return_pdw else None
        return pulse, pdw


class AugmentedPulseDataset(Dataset):
    """
    SNR-degradation augmentation.
    Wraps a SimplePulseDatasetPyTorch (or Subset of one) constructed with
    return_pdw=True. Modes "single" / "contrastive" are unchanged.

    ADC-aware: if the wrapped base was constructed with adc_names=[...] AND
    adc_mode="contrastive" (the base's default), each base item is already
    (n_adc, C, L) pulse + list of n_adc pdw dicts. This class detects that
    (walking through Subset the same way it already does for return_pdw)
    and applies _augment_pulse to each ADC view independently, using THAT
    view's own noise_floor/pulse_power, then re-stacks along the ADC
    dimension - "single" mode picks one SNR target per call and stacks
    (n_adc, C, L); "contrastive" mode stacks every (adc, snr) pair as
    (n_adc * n_snr, C, L), ADC-major then SNR order. This is unaffected by
    the base dataset's TOA alignment details - by the time a group reaches
    this class it's just an (n_adc, C, L) stack + matching pdw list.

    If the base was constructed with adc_names=[...] AND adc_mode="single", each
    base item is already a flattened single pulse + single pdw (one ADC per
    item) - this class treats it exactly like a single-ADC base, no ADC
    stacking logic applies, since the base already did the flattening.
    """

    def __init__(
        self,
        base_dataset: SimplePulseDatasetPyTorch,
        target_snrs_db: List[Union[float, str]],
        add_magnitude: bool = False,
        normalise: bool = False,
        mode: Literal["single", "contrastive"] = "single",
        return_pdw: bool = False,
    ):
        if mode not in ("single", "contrastive"):
            raise ValueError(f"mode must be 'single' or 'contrastive', got {mode!r}")

        if not self._base_has_pdw(base_dataset):
            raise ValueError(
                "AugmentedPulseDataset requires base_dataset to have been constructed "
                "with return_pdw=True - noise_floor/pulse_power (from pdw) are needed "
                "to degrade SNR."
            )

        self.base_dataset = base_dataset
        self.degrade_snr = DegradeToSNR()
        self.target_snrs_db = target_snrs_db
        self.add_magnitude = add_magnitude
        self.normalise = normalise
        self.mode = mode
        self.return_pdw = return_pdw
        self.snr_copies = len(target_snrs_db)
        self.n_views = self.snr_copies

        base_multi_adc = self._base_attr(base_dataset, "multi_adc", False)
        base_adc_mode = self._base_attr(base_dataset, "adc_mode", "contrastive")
        
        # Only treat the base as ADC-stacked if it's actually handing us stacks;
        # a multi_adc base in adc_mode="single" has already flattened to one ADC per item.
        self.multi_adc = base_multi_adc and base_adc_mode == "contrastive"
        self.n_adc = len(self._base_attr(base_dataset, "adc_names", [])) if self.multi_adc else None

    @staticmethod
    def _base_attr(dataset, name, default):
        """Walk through Subset wrapping (e.g. from random_split) to find an attribute on the underlying base dataset."""
        seen = dataset
        while not hasattr(seen, name) and hasattr(seen, "dataset"):
            seen = seen.dataset
        return getattr(seen, name, default)

    @classmethod
    def _base_has_pdw(cls, dataset) -> bool:
        return bool(cls._base_attr(dataset, "return_pdw", False))

    def __len__(self):
        if self.mode == "contrastive":
            return len(self.base_dataset)
        else:  # single
            return len(self.base_dataset) * self.n_views

    # Order is important here, we need to be careful how we apply augmentations/corrections
    def _augment_pulse(self, pulse, noise_floor, pulse_power, snr_idx):
        target_snr = self.target_snrs_db[snr_idx]

        pulse, achieved_snr = self.degrade_snr(
            pulse, noise_floor_db=noise_floor, pulse_power_db=pulse_power, target_snr_db=target_snr,
        )

        if not torch.isfinite(pulse).all():
            raise RuntimeError("NaN generated by DegradeToSNR")

        if self.add_magnitude:
            pulse = AddMagnitude()(pulse)

        if self.normalise:
            peak = pulse[2].max()
            if not torch.isfinite(peak):
                raise ValueError("NaN generated before/during magnitude calculation")
            pulse = NormaliseByPeakMagnitude()(pulse)

        return pulse, achieved_snr

    @staticmethod
    def _update_pdw(pdw: Dict[str, Any], achieved_snr: float) -> Dict[str, Any]:
        """Copy of a pulse's pdw with 'snr' overwritten to the SNR that pulse was actually degraded to."""
        pdw = dict(pdw)
        pdw["snr"] = achieved_snr
        return pdw

    def _augment_adc_stack(self, base_pulse, pdw_list, snr_idxs):
        """base_pulse: (n_adc, C, L); pdw_list: list of n_adc dicts. Returns stacked views and matching pdw list, ADC-major then SNR."""
        views, pdws = [], [] if self.return_pdw else None
        for adc_idx in range(self.n_adc):
            adc_pulse = base_pulse[adc_idx]
            adc_pdw = pdw_list[adc_idx]
            for snr_idx in snr_idxs:
                pulse, achieved_snr = self._augment_pulse(
                    adc_pulse.clone(), adc_pdw["noise_floor"], adc_pdw["pulse_power"], snr_idx,
                )
                views.append(pulse)
                if self.return_pdw:
                    pdws.append(self._update_pdw(adc_pdw, achieved_snr))
        return torch.stack(views, dim=0), pdws

    def __getitem__(self, idx):
        if self.mode == "contrastive":
            base_pulse, labels, pdw = self.base_dataset[idx]

            if self.multi_adc:
                views, pdws = self._augment_adc_stack(base_pulse, pdw, range(self.snr_copies))
                output = (views, labels)
                if self.return_pdw:
                    output = output + (pdws,)
                return output

            noise_floor = pdw["noise_floor"]
            pulse_power = pdw["pulse_power"]

            views = []
            pdws = [] if self.return_pdw else None
            for snr_idx in range(self.snr_copies):
                pulse, achieved_snr = self._augment_pulse(
                    base_pulse.clone(), noise_floor, pulse_power, snr_idx,
                )
                views.append(pulse)
                if self.return_pdw:
                    pdws.append(self._update_pdw(pdw, achieved_snr))

            views = torch.stack(views, dim=0)
            output = (views, labels)
            if self.return_pdw:
                output = output + (pdws,)
            return output

        # ------------------------------
        # Single mode
        # ------------------------------
        base_idx = idx // self.snr_copies
        snr_idx = idx % self.snr_copies

        base_pulse, labels, pdw = self.base_dataset[base_idx]

        if self.multi_adc:
            views, pdws = self._augment_adc_stack(base_pulse, pdw, [snr_idx])
            output = (views, labels)
            if self.return_pdw:
                output = output + (pdws,)
            return output

        noise_floor = pdw["noise_floor"]
        pulse_power = pdw["pulse_power"]
        snr = pdw["snr"]

        pulse, achieved_snr = self._augment_pulse(
            base_pulse.clone(), noise_floor, pulse_power, snr_idx,
        )

        output = (pulse, labels)
        if self.return_pdw:
            output = output + (self._update_pdw(pdw, achieved_snr),)

        return output
