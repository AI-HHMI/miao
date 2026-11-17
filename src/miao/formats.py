"""Pyramid metadata for the stores miao reads besides OME-NGFF zarr: N5 and Neuroglancer precomputed.

miao reads both formats through tensorstore, like zarr. This module describes them with the same
`OmeMetadata` that the OME-NGFF reader returns, so the rest of miao does not need to know which
format a volume is in. The formats differ in where a level's voxel size and origin are recorded.
This module finds them.

**N5** has no standard multiscale metadata. A multiscale group holds datasets `s0`, `s1`, ... (a
pyramid may start at `s1` when its `s0` was never written), or the group is itself one array. Each
tool writes voxel sizes in its own way. miao determines a level's voxel size (in nm) as follows:
  - If it can, it multiplies a base voxel size by the level's downsampling factors. The base is
    the group's `pixelResolution` (n5-viewer, render exports) or `resolution`, else the first
    level's `pixelResolution`, else the first level's `resolution` divided by its factors. The
    factors are the level's `downsamplingFactors`, else the group's `scales` entry for the level.
  - Otherwise it uses the level's own `resolution` (FAFB's s0, the FlyEM hemibrain export).
A level that has both a `resolution` and factors must agree with base x factors, as FAFB's levels
do. Dimensions are x, y, z in the order N5 lists them (tensorstore's n5 driver keeps that order),
unless the group names them in `axes`. A `pixelResolution` in micrometers is converted to nm.

**Neuroglancer precomputed** describes every level in one `info` file: voxel size in nm, size and
voxel offset. The axes are always x, y, z, plus a channel dimension, which miao drops when there is
only one channel. A volume is the directory that holds the `info` file. Its levels are named by
their keys.

miao assumes that both kinds of pyramid were built by block averaging. The voxel centers of a
downsampled level then sit half a voxel further in than level 0's, so level k's translation is
(voxel_k - voxel_0) / 2, plus its offset. Most lmd-v0.0.1 zarr stores record the same translation
per level.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from miao.zarr_meta import OmeMetadata, ScaleMetadata, StoreFormat, read_ome_metadata

_NM_PER_UNIT = {"nm": 1.0, "nanometer": 1.0, "nanometers": 1.0,
                "um": 1e3, "µm": 1e3, "micrometer": 1e3, "micrometers": 1e3, "micron": 1e3}


def read_metadata(
    path: str | Path,
    group_key: str,
    store_format: StoreFormat,
    requested_scales: list[int] | None = None,
) -> OmeMetadata:
    """Pyramid metadata for the multiscale group (N5) or volume (precomputed) at path/group_key."""
    if store_format in ("zarr2", "zarr3"):
        return read_ome_metadata(path, group_key, store_format, requested_scales)
    if store_format == "n5":
        return read_n5_metadata(path, group_key, requested_scales)
    if store_format == "precomputed":
        return read_precomputed_metadata(path, group_key, requested_scales)
    raise ValueError(f"Unknown store format {store_format!r}")


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _space_axes(names: list[str]) -> list[dict]:
    return [{"name": n, "type": "channel"} if n == "c"
            else {"name": n, "type": "space", "unit": "nanometer"} for n in names]


def _select(levels: list, requested: list[int] | None, where: Path) -> list[int]:
    if requested is None:
        return list(range(len(levels)))
    for i in requested:
        if not 0 <= i < len(levels):
            raise IndexError(f"Requested scale level {i} but only {len(levels)} exist at {where}")
    return list(requested)


# --- N5 ------------------------------------------------------------------------------------------


def _resolution_nm(value, where: Path) -> list[float] | None:
    """A `pixelResolution` ({"dimensions": [...], "unit": ...}) or a `resolution` ([...]) in nm."""
    if value is None:
        return None
    if isinstance(value, dict):
        unit = str(value.get("unit", "nm")).lower()
        if unit not in _NM_PER_UNIT:
            raise ValueError(f"{where}: pixelResolution unit {unit!r} is not a length miao knows")
        return [float(v) * _NM_PER_UNIT[unit] for v in value["dimensions"]]
    return [float(v) for v in value]


def _n5_levels(group: Path) -> list[tuple[int | None, str, dict]]:
    """(level number, dataset path, attributes) per level, finest first: the group's s0, s1, ...
    datasets, or the group itself when it is an array (number None, path "")."""
    own = _json(group / "attributes.json")
    if "dimensions" in own:
        return [(None, "", own)]
    found = []
    if group.is_dir():
        for child in group.iterdir():
            m = re.fullmatch(r"s(\d+)", child.name)
            attrs = _json(child / "attributes.json") if m else {}
            if "dimensions" in attrs:
                found.append((int(m.group(1)), child.name, attrs))
    if not found:
        raise ValueError(f"No N5 dataset at {group}, and no s0, s1, ... datasets inside it")
    return sorted(found)


def _n5_voxel_sizes(
    group_attrs: dict, levels: list, where: Path
) -> tuple[list[list[float]], list[float]]:
    """Each level's voxel size in nm, and the base (level-0) voxel size the pyramid is built on."""
    scales = group_attrs.get("scales")

    def factors(n, attrs):
        f = attrs.get("downsamplingFactors")
        if f is None and scales is not None and n is not None and n < len(scales):
            f = scales[n]
        return None if f is None else [float(v) for v in f]

    n0, _, first = levels[0]
    f0 = factors(n0, first)
    base = (_resolution_nm(group_attrs.get("pixelResolution"), where)
            or _resolution_nm(group_attrs.get("resolution"), where)
            or _resolution_nm(first.get("pixelResolution"), where))
    if base is None and "resolution" in first:
        own = _resolution_nm(first["resolution"], where)
        base = own if f0 is None else [r / f for r, f in zip(own, f0)]

    sizes = []
    for n, name, attrs in levels:
        f = factors(n, attrs)
        own = _resolution_nm(attrs.get("resolution"), where)
        if f is not None and base is not None:
            voxel = [b * x for b, x in zip(base, f)]
            if own is not None and not np.allclose(own, voxel, rtol=1e-6):
                raise ValueError(
                    f"{where / name}: resolution {own} disagrees with the base voxel size {base} "
                    f"times its downsampling factors {f}")
        elif own is not None:
            voxel = own
        elif base is not None and n in (0, None):
            voxel = base
        else:
            raise ValueError(
                f"{where / name}: no voxel size. Expected `resolution`, or `pixelResolution` / "
                "`resolution` on the group with `downsamplingFactors` (or the group's `scales`).")
        sizes.append(voxel)
    return sizes, (base if base is not None else sizes[0])


def read_n5_metadata(
    path: str | Path, group_key: str, requested_scales: list[int] | None = None
) -> OmeMetadata:
    """Pyramid metadata for an N5 multiscale group (or single dataset) at path/group_key."""
    group = Path(path) / group_key
    levels = _n5_levels(group)
    group_attrs = {} if levels[0][0] is None else _json(group / "attributes.json")
    sizes, base = _n5_voxel_sizes(group_attrs, levels, group)

    ndim = len(levels[0][2]["dimensions"])
    names = group_attrs.get("axes") or (["x", "y", "z"] if ndim == 3 else None)
    if names is None or len(names) != ndim:
        raise ValueError(f"{group}: {ndim}-dimensional N5 without an `axes` attribute naming them")
    names = [str(n).lower() for n in names]

    scales: dict[int, ScaleMetadata] = {}
    for i in _select(levels, requested_scales, group):
        _, name, attrs = levels[i]
        offset = attrs.get("offset")
        if offset is not None and any(float(o) != 0 for o in offset):
            raise ValueError(f"{group / name}: non-zero `offset` {offset} is not supported")
        scales[i] = ScaleMetadata(
            path=name,
            scale_factors=sizes[i],
            shape=[int(v) for v in attrs["dimensions"]],
            chunks=[int(v) for v in attrs["blockSize"]],
            dtype=np.dtype(attrs["dataType"]),
            translation=[(v - b) / 2.0 for v, b in zip(sizes[i], base)],
        )
    return OmeMetadata(
        axes=_space_axes(names),
        axis_names=names,
        scales=scales,
        zarr_version="n5",
        outer_scale=[1.0] * ndim,
        outer_translation=[0.0] * ndim,
    )


# --- Neuroglancer precomputed ----------------------------------------------------------------------


def read_precomputed_metadata(
    path: str | Path, group_key: str, requested_scales: list[int] | None = None
) -> OmeMetadata:
    """Pyramid metadata for the Neuroglancer precomputed volume (an `info` directory) at
    path/group_key."""
    volume = Path(path) / group_key
    info = _json(volume / "info")
    if "scales" not in info:
        raise ValueError(f"No Neuroglancer precomputed `info` with scales at {volume}")
    levels = sorted(info["scales"], key=lambda s: float(np.prod(s["resolution"])))
    channels = int(info.get("num_channels", 1))
    names = ["x", "y", "z"] + (["c"] if channels > 1 else [])
    res0 = [float(v) for v in levels[0]["resolution"]]

    scales: dict[int, ScaleMetadata] = {}
    for i in _select(levels, requested_scales, volume):
        s = levels[i]
        res = [float(v) for v in s["resolution"]]
        offset = [float(v) for v in s.get("voxel_offset", [0, 0, 0])]
        extra = [channels] if channels > 1 else []
        scales[i] = ScaleMetadata(
            path=s["key"],
            scale_factors=res + ([1.0] if extra else []),
            shape=[int(v) for v in s["size"]] + extra,
            chunks=[int(v) for v in s["chunk_sizes"][0]] + extra,
            dtype=np.dtype(info["data_type"]),
            translation=[o * r + (r - r0) / 2.0 for o, r, r0 in zip(offset, res, res0)]
            + ([0.0] if extra else []),
        )
    return OmeMetadata(
        axes=_space_axes(names),
        axis_names=names,
        scales=scales,
        zarr_version="precomputed",
        outer_scale=[1.0] * len(names),
        outer_translation=[0.0] * len(names),
    )
