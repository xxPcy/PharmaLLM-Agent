"""Private CPU imaging pipeline. No image pixels or DICOM identity enter LLM context."""

import gzip
import io
import json
import logging
import warnings
import math
import os
import tempfile
import zipfile
from datetime import timedelta

import nibabel as nib
import numpy as np
import pydicom
from nibabel.processing import resample_from_to, resample_to_output
from PIL import Image, ImageOps

from pharma_agent.db import now
from pharma_agent.errors import AppError


class _ContentFreeDicomLog(logging.Filter):
    def filter(self, record):
        # pydicom may include rejected tag values in warnings, including patient identity.
        record.msg, record.args = "DICOM parser diagnostic; content omitted", ()
        record.exc_info, record.exc_text, record.stack_info = None, None, None
        return True


logging.getLogger("pydicom").addFilter(_ContentFreeDicomLog())

PALETTE = [(65, 190, 163), (241, 178, 88), (115, 155, 240), (220, 119, 154), (170, 145, 226)]


def bad(message, code="INVALID_IMAGE"):
    return AppError(code, message, 422)


def clean_name(name):
    return str(name or "未命名文件").replace("\\", "/").split("/")[-1].replace("\x00", "")[:180]


class ImagingStore:
    def __init__(self, storage, settings):
        self.storage, self.settings = storage, settings

    def metadata(self, asset):
        path = self.storage.path(asset.id + ".meta.json")
        if path.exists():
            meta = json.loads(path.read_text())
        else:
            meta = {"name": asset.id, "kind": "attachment", "status": asset.status}
        return {
            **meta,
            "asset_id": asset.id,
            "mime": asset.mime,
            "size_bytes": asset.size_bytes,
            "status": asset.status,
            "created_at": asset.created_at.isoformat(),
        }

    def save_meta(self, asset_id, meta):
        # Atomic replacement; names and derived geometry never reach ordinary logs.
        target = self.storage.path(asset_id + ".meta.json")
        fd, temporary = tempfile.mkstemp(dir=self.storage.root, prefix=".meta-")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(meta, handle, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def purge(self, asset_id):
        for suffix in (".meta.json", ".volume.nii.gz", ".display.nii.gz", ".preview.png"):
            self.storage.remove(asset_id + suffix)

    def check_volume(self, image):
        if len(image.shape) != 3 or min(image.shape) < 2:
            raise bad("需要真正的三维体数据；二维或多通道/时间序列请先转换、拆分")
        if math.prod(image.shape) > self.settings.imaging_max_voxels:
            raise bad("体素数量超过配置上限", "IMAGE_TOO_LARGE")
        affine = image.affine
        if affine is None or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
            raise bad("影像缺少有效的空间坐标变换")
        if not (image.header["sform_code"] or image.header["qform_code"]):
            raise bad("NIfTI 未声明 sform/qform，无法可靠定位方向；请先补全空间信息")
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise bad("影像含有非有限像素值")
        return data

    def nifti(self, content):
        if content.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as handle:
                content = handle.read(self.settings.imaging_max_expanded_bytes + 1)
        if len(content) > self.settings.imaging_max_expanded_bytes:
            raise bad("影像解压后超过配置上限", "IMAGE_TOO_LARGE")
        if len(content) < 348:
            raise bad("NIfTI 文件不完整")
        header_size = int.from_bytes(content[:4], "little")
        if header_size not in {348, 540}:
            header_size = int.from_bytes(content[:4], "big")
        cls = nib.Nifti2Image if header_size == 540 else nib.Nifti1Image
        image = cls.from_bytes(content)
        self.check_volume(image)
        return image

    def dicom(self, files):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self._dicom(files)

    def _dicom(self, files):
        if not 1 <= len(files) <= self.settings.imaging_max_series_files:
            raise bad("DICOM 序列文件数量超出上限")
        datasets = [pydicom.dcmread(io.BytesIO(body)) for _, body in files]
        first = datasets[0]
        if any(
            int(getattr(d, "NumberOfFrames", 1)) != 1 or int(getattr(d, "SamplesPerPixel", 1)) != 1
            for d in datasets
        ):
            raise bad("暂不支持增强多帧或彩色 DICOM；请转换为三维 NIfTI")
        if len(files) == 1:
            if int(first.Rows) * int(first.Columns) > self.settings.imaging_max_voxels:
                raise bad("图像尺寸超过上限")
            pixels = first.pixel_array.astype(np.float32)
            preview = self.gray(pixels, None, None)
            if getattr(first, "PhotometricInterpretation", "") == "MONOCHROME1":
                preview = 255 - preview
            return (
                None,
                {
                    "kind": "dicom_slice",
                    "format": "DICOM 单层",
                    "dimensions": list(pixels.shape[::-1]),
                    "series_count": 1,
                    "slice_count": 1,
                    "modality": str(getattr(first, "Modality", "unknown")),
                },
                preview,
            )
        uid = getattr(first, "SeriesInstanceUID", None)
        if not uid or any(getattr(d, "SeriesInstanceUID", None) != uid for d in datasets):
            raise bad("一次只能上传一个 DICOM 序列，请按 Series 分组")
        orient = np.asarray(first.ImageOrientationPatient, dtype=float)
        row, col = orient[:3], orient[3:]
        if not np.allclose(
            [np.linalg.norm(row), np.linalg.norm(col), np.dot(row, col)], [1, 1, 0], atol=1e-4
        ):
            raise bad("DICOM 方向余弦无效")
        normal = np.cross(row, col)
        spacing = np.asarray(first.PixelSpacing, dtype=float)
        if not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise bad("DICOM 像素间距无效")
        for d in datasets:
            if (
                (d.Rows, d.Columns) != (first.Rows, first.Columns)
                or not np.allclose(d.ImageOrientationPatient, orient, atol=1e-4)
                or not np.allclose(d.PixelSpacing, spacing, atol=1e-4)
            ):
                raise bad("DICOM 序列尺寸、间距或方向不一致")
        datasets.sort(key=lambda d: float(np.dot(np.asarray(d.ImagePositionPatient, float), normal)))
        positions = np.asarray([d.ImagePositionPatient for d in datasets], float)
        delta = np.diff(positions, axis=0)
        step = delta.mean(axis=0)
        dz = float(np.dot(step, normal))
        if dz <= 0 or not np.allclose(delta, step, atol=max(0.01, dz * 0.01)):
            raise bad("DICOM 层距不均匀、缺层或重复切片，请先整理序列")
        shape = (int(first.Columns), int(first.Rows), len(datasets))
        if math.prod(shape) > self.settings.imaging_max_voxels:
            raise bad("DICOM 体素数量超过上限", "IMAGE_TOO_LARGE")
        data = np.stack(
            [
                d.pixel_array.astype(np.float32).T * float(getattr(d, "RescaleSlope", 1))
                + float(getattr(d, "RescaleIntercept", 0))
                for d in datasets
            ],
            axis=2,
        )
        lps = np.eye(4)
        lps[:3, 0], lps[:3, 1], lps[:3, 2], lps[:3, 3] = (
            row * spacing[1],
            col * spacing[0],
            step,
            positions[0],
        )
        affine = np.diag([-1, -1, 1, 1]) @ lps
        image = nib.Nifti1Image(data, affine)
        image.header.set_xyzt_units("mm")
        return (
            image,
            {
                "series_count": 1,
                "slice_count": len(datasets),
                "modality": str(getattr(first, "Modality", "unknown")),
            },
            None,
        )

    def archive(self, content):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [
                m for m in archive.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX/")
            ]
            if (
                not members
                or len(members) > self.settings.imaging_max_series_files
                or sum(m.file_size for m in members) > self.settings.imaging_max_expanded_bytes
            ):
                raise bad("ZIP 文件数量或解压后大小超过上限", "IMAGE_TOO_LARGE")
            # Never extract paths; read bounded members into memory.
            result, total = [], 0
            for member in members:
                with archive.open(member) as handle:
                    body = handle.read(self.settings.imaging_max_expanded_bytes - total + 1)
                total += len(body)
                if total > self.settings.imaging_max_expanded_bytes:
                    raise bad("ZIP 解压后大小超过上限", "IMAGE_TOO_LARGE")
                result.append((member.filename, body))
            return result

    def prepare(self, asset_id, files, modality="unknown", conversation_id=None):
        """Parse before publishing DB asset; callers compensate files on any error."""
        name = clean_name(files[0][0]) if len(files) == 1 else f"DICOM 序列（{len(files)} 层）.zip"
        body = files[0][1]
        image, extra, preview = None, {}, None
        try:
            if len(files) > 1 or name.lower().endswith(".zip"):
                image, extra, preview = self.dicom(files if len(files) > 1 else self.archive(body))
                mime, kind, fmt = "application/zip", "volume", "DICOM 序列"
                if len(files) > 1:
                    output = io.BytesIO()
                    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                        for i, (_, part) in enumerate(files):
                            archive.writestr(f"slice-{i:05}.dcm", part)
                    body = output.getvalue()
            elif name.lower().endswith((".nii", ".nii.gz")):
                image = self.nifti(body)
                mime, kind, fmt = (
                    "application/gzip" if name.lower().endswith(".gz") else "application/octet-stream",
                    "volume",
                    "NIfTI",
                )
            elif body.startswith(b"\x89PNG\r\n\x1a\n") or body.startswith(b"\xff\xd8\xff"):
                with Image.open(io.BytesIO(body)) as photo:
                    if photo.width * photo.height > self.settings.imaging_max_voxels:
                        raise bad("图片尺寸超过上限")
                    fmt = photo.format
                    normalized = ImageOps.exif_transpose(photo).convert("RGB")
                    self.storage.write(asset_id + ".preview.png", self.png(normalized))
                    extra = {"dimensions": list(normalized.size)}
                mime, kind = "image/png" if fmt == "PNG" else "image/jpeg", "image"
            elif name.lower().endswith((".txt", ".json")):
                text = body.decode("utf-8")
                if name.lower().endswith(".json"):
                    json.loads(text)
                mime, kind, fmt = (
                    ("application/json" if name.lower().endswith(".json") else "text/plain"),
                    "attachment",
                    "JSON" if name.lower().endswith(".json") else "TXT",
                )
            else:
                image, extra, preview = self.dicom(files)
                mime, kind, fmt = "application/dicom", "dicom_slice", "DICOM 单层"
            actual_modality = extra.pop("modality", "unknown")
            if actual_modality != "unknown" and modality != "unknown":
                declared = "MR" if modality in {"T1", "T2", "FLAIR", "ADC", "DWI"} else modality
                if declared != actual_modality:
                    raise bad("声明的模态与 DICOM 采集模态不一致", "MODALITY_MISMATCH")
            modality_source = (
                "user" if modality != "unknown" else "dicom" if actual_modality != "unknown" else "unknown"
            )
            extra["acquisition_modality"] = actual_modality
            if modality == "unknown" and actual_modality != "unknown":
                modality = actual_modality
            meta = {
                "name": name,
                "kind": kind,
                "format": fmt,
                "status": "ready",
                "conversation_id": conversation_id,
                "modality": modality,
                "modality_source": modality_source,
                "retention_days": self.settings.asset_retention_days,
                "expires_at": (now() + timedelta(days=self.settings.asset_retention_days)).isoformat(),
                **extra,
            }
            if image is not None:
                meta.update(self.save_volume(asset_id, image))
                meta["kind"] = "volume"
            if preview is not None:
                self.storage.write(asset_id + ".preview.png", self.png(Image.fromarray(preview)))
            key = asset_id + ".bin"
            file_hash = self.storage.write(key, body)
            self.save_meta(asset_id, meta)
            return key, file_hash, mime, len(body), meta
        except AppError:
            self.storage.remove(asset_id + ".bin")
            self.purge(asset_id)
            raise
        except Exception:
            self.storage.remove(asset_id + ".bin")
            self.purge(asset_id)
            raise bad("文件解析失败。请检查格式、空间信息或 DICOM 解码器；增强多帧请先转 NIfTI") from None

    def save_volume(self, asset_id, image, mask=False, target=None):
        data = self.check_volume(image)
        self.storage.write(asset_id + ".volume.nii.gz", gzip.compress(image.to_bytes()))
        canonical = nib.as_closest_canonical(image)
        # Orthogonal world axes: resample oblique/sheared acquisitions; retain original for inference.
        matrix = canonical.affine[:3, :3]
        if target:
            display = resample_from_to(image, target, order=0)
        elif np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-5):
            display = canonical
        else:
            spacing = nib.affines.voxel_sizes(canonical.affine)
            corners = nib.affines.apply_affine(
                canonical.affine,
                np.array(np.meshgrid(*[(0, n - 1) for n in canonical.shape])).reshape(3, -1).T,
            )
            out_shape = np.ceil(np.ptp(corners, axis=0) / spacing + 1).astype(int)
            if math.prod(out_shape) > self.settings.imaging_max_voxels:
                raise bad("重采样尺寸超过上限", "IMAGE_TOO_LARGE")
            display = resample_to_output(image, voxel_sizes=spacing, order=0 if mask else 1)
        self.storage.write(asset_id + ".display.nii.gz", gzip.compress(display.to_bytes()))
        low, high = np.percentile(data, [1, 99]).tolist()
        units = image.header.get_xyzt_units()[0]
        return {
            "dimensions": list(image.shape),
            "spacing": nib.affines.voxel_sizes(image.affine).tolist(),
            "units": units,
            "orientation": list(nib.aff2axcodes(image.affine)),
            "display_dimensions": list(display.shape),
            "display_spacing": nib.affines.voxel_sizes(display.affine).tolist(),
            "display_orientation": list(nib.aff2axcodes(display.affine)),
            "intensity_range": [low, high],
            "channels": 1,
            "display_resampled": not np.allclose(display.affine, image.affine),
        }

    def load(self, asset_id, display=True):
        try:
            return nib.load(
                self.storage.path(asset_id + (".display.nii.gz" if display else ".volume.nii.gz"))
            )
        except FileNotFoundError:
            raise AppError("ASSET_UNAVAILABLE", "影像文件已删除或不可读取", 404) from None

    @staticmethod
    def gray(data, window, level):
        lo, hi = np.percentile(data, [1, 99])
        width = max(float(window) if window is not None else float(hi - lo), 1e-6)
        center = float(level) if level is not None else float((lo + hi) / 2)
        return (np.clip((data - center + width / 2) / width, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def png(image):
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def slice(self, asset_id, plane, index, window=None, level=None, mask_id=None, opacity=0.45, labels=None):
        image = self.load(asset_id)
        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
        if not 0 <= index < image.shape[axis]:
            raise bad("切片编号超出范围")
        # Display x grows toward the positive world axis, top toward positive y/z.
        pixels = np.flipud(np.take(np.asarray(image.dataobj), index, axis=axis).T)
        rgb = np.repeat(self.gray(pixels, window, level)[..., None], 3, axis=2)
        if mask_id:
            mask = self.load(mask_id)
            if mask.shape != image.shape or not np.allclose(mask.affine, image.affine, atol=1e-4):
                raise bad("分割结果与影像显示空间不一致", "MASK_SPACE_MISMATCH")
            values = np.flipud(np.take(np.asarray(mask.dataobj), index, axis=axis).T)
            for label in sorted(np.unique(values)):
                value = int(label)
                if value == 0 or (labels is not None and value not in labels):
                    continue
                color = np.array(PALETTE[(value - 1) % len(PALETTE)])
                selector = values == label
                rgb[selector] = (rgb[selector] * (1 - opacity) + color * opacity).astype(np.uint8)
        return self.png(Image.fromarray(rgb))

    def accept_mask(self, asset_id, content, source_id, labels):
        mask, source = self.nifti(content), self.load(source_id, display=False)
        if mask.shape != source.shape or not np.allclose(mask.affine, source.affine, atol=1e-4):
            raise bad("服务返回的 mask 与原影像物理空间不一致", "MASK_SPACE_MISMATCH")
        if mask.header.get_xyzt_units()[0] != source.header.get_xyzt_units()[0]:
            raise bad("分割结果与源影像的空间单位不一致", "MASK_SPACE_MISMATCH")
        data = mask.get_fdata(dtype=np.float32)
        if not np.allclose(data, np.rint(data)) or not set(np.unique(data).astype(int)).issubset(
            {int(v) for v in labels}
        ):
            raise bad("分割标签不在模型标签白名单中", "MASK_LABEL_MISMATCH")
        display = self.load(source_id)
        meta = self.save_volume(asset_id, mask, mask=True, target=(display.shape, display.affine))
        units = source.header.get_xyzt_units()[0]
        unit_scale = {"mm": 1, "meter": 1000, "micron": 0.001}.get(units)
        measurements = []
        for value, name in labels.items():
            if int(value) == 0:
                continue
            count = int(np.count_nonzero(data == int(value)))
            measurements.append(
                {
                    "label": int(value),
                    "name": name,
                    "voxel_count": count,
                    **(
                        {
                            "volume_mm3": count
                            * abs(float(np.linalg.det(source.affine[:3, :3])))
                            * unit_scale**3,
                            "basis": "标签体素数 × 原始 affine 行列式；单位换算为 mm³",
                        }
                        if unit_scale
                        else {}
                    ),
                }
            )
        meta.update(
            kind="mask",
            format="NIfTI",
            name="segmentation.nii.gz",
            source_asset_id=source_id,
            labels=labels,
            measurements=measurements,
            status="ready",
        )
        self.save_meta(asset_id, meta)
        return meta
