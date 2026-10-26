"""Bounded physical-coordinate voxel-boundary meshes from real segmentation labels."""

import math

import nibabel as nib
import numpy as np

from pharma_agent.imaging import PALETTE


def mask_surface(image, labels, face_budget=14000, sample_limit=128):
    data = np.asarray(image.dataobj)
    steps = np.ones(3, dtype=int)
    spacing = nib.affines.voxel_sizes(image.affine)

    def coarsen():
        # Preserve already thick slices: reduce the axis with the finest physical sampling first.
        candidates = np.where(steps < np.asarray(data.shape), spacing * steps, np.inf)
        axis = int(np.argmin(candidates))
        if not np.isfinite(candidates[axis]):
            return False
        steps[axis] += 1
        return True

    while math.prod(np.ceil(np.asarray(data.shape) / steps).astype(int)) > sample_limit**3:
        if not coarsen():
            break
    while True:
        sampled = data[tuple(slice(None, None, int(step)) for step in steps)]
        boundaries, count = [], 0
        # Count first, allocate vertices only once the output fits the budget.
        # A noisy/checkerboard mask must not allocate millions of face arrays.
        for value, name in labels.items():
            value = int(value)
            if not value:
                continue
            occupied = sampled == value
            padded = np.pad(occupied, 1)
            for axis in range(3):
                for sign in (-1, 1):
                    offset = [slice(1, -1)] * 3
                    offset[axis] = slice(0, -2) if sign < 0 else slice(2, None)
                    selector = occupied & ~padded[tuple(offset)]
                    face_count = int(np.count_nonzero(selector))
                    count += face_count
                    if face_count and count <= face_budget:
                        boundaries.append((value, name, axis, sign, np.argwhere(selector)))
                    if count > face_budget:
                        break
                if count > face_budget:
                    break
            if count > face_budget:
                break
        if count <= face_budget or not coarsen():
            break
    grouped = {}
    inverse = np.linalg.inv(image.affine[:3, :3]).T
    for value, name, axis, sign, centers in boundaries:
        other = [v for v in range(3) if v != axis]
        corners = np.zeros((4, 3), dtype=float)
        corners[:, axis] = sign * 0.5
        corners[:, other[0]] = [-0.5, 0.5, 0.5, -0.5]
        corners[:, other[1]] = [-0.5, -0.5, 0.5, 0.5]
        world = nib.affines.apply_affine(
            image.affine, ((centers[:, None, :] + corners) * steps).reshape(-1, 3)
        ).reshape(-1, 4, 3)
        normal = np.cross(world[0, 1] - world[0, 0], world[0, 2] - world[0, 0])
        if np.dot(normal, inverse[:, axis] * sign) < 0:
            world = world[:, [0, 3, 2, 1], :]
        record = grouped.setdefault(
            value, {"id": value, "name": name, "color": PALETTE[(value - 1) % len(PALETTE)], "quads": []}
        )
        record["quads"].extend(world.round(3).reshape(-1).tolist())
    return {
        "space": "RAS",
        "units": image.header.get_xyzt_units()[0],
        "stride": int(max(steps)),
        "sampling": steps.tolist(),
        "preview_only": True,
        "method": "voxel_boundary",
        "face_count": count,
        "segments": list(grouped.values()),
    }
