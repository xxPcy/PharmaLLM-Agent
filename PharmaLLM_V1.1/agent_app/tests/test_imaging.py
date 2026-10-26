"""Synthetic CPU-only geometry and HTTP contract tests; never call real inference."""

import gzip
import io
import zipfile
from unittest.mock import AsyncMock

import httpx
import nibabel as nib
import numpy as np
import pydicom
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from pharma_agent.errors import AppError
from pharma_agent.imaging import ImagingStore
from pharma_agent.segmentation import SegmentationGateway
from pharma_agent.settings import Settings
from pharma_agent.storage import LocalStorage


@pytest.fixture
def store(tmp_path):
    settings = Settings(_env_file=None, jwt_secret="synthetic-geometry-" * 4)
    return ImagingStore(LocalStorage(tmp_path / "assets"), settings)


def volume(shape=(9, 11, 7), affine=None):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    image = nib.Nifti1Image(data, np.diag([0.7, 1.2, 2.5, 1]) if affine is None else affine)
    image.header.set_xyzt_units("mm")
    return image


def test_nifti_metadata_slice_and_no_phi(store):
    original = volume(affine=np.array([[-0.7, 0, 0, 10], [0, 1.2, 0, 20], [0, 0, 2.5, -4], [0, 0, 0, 1]]))
    _, _, _, _, meta = store.prepare(
        "asset_source", [("scan.nii.gz", gzip.compress(original.to_bytes()))], "T2"
    )
    assert meta["orientation"] == ["L", "A", "S"]
    assert meta["display_orientation"] == ["R", "A", "S"]
    assert np.allclose(meta["spacing"], [0.7, 1.2, 2.5])
    for plane, index, expected in [("axial", 3, (9, 11)), ("coronal", 4, (9, 7)), ("sagittal", 2, (11, 7))]:
        content = store.slice("asset_source", plane, index)
        assert Image.open(io.BytesIO(content)).size == expected
    with pytest.raises(AppError):
        store.slice("asset_source", "axial", 999)
    store.purge("asset_source")
    assert not store.storage.path("asset_source.display.nii.gz").exists()


def test_oblique_image_is_resampled_to_world_axes(store):
    angle = 0.3
    affine = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 5],
            [np.sin(angle), np.cos(angle), 0, -2],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ]
    )
    meta = store.save_volume("asset_oblique", volume(affine=affine))
    result = store.load("asset_oblique")
    assert meta["display_resampled"]
    assert np.allclose(result.affine[:3, :3], np.diag(np.diag(result.affine[:3, :3])))
    assert nib.aff2axcodes(result.affine) == ("R", "A", "S")


def test_mask_alignment_labels_and_real_measurements(store):
    image = volume()
    store.save_volume("asset_source", image)
    data = np.zeros(image.shape, dtype=np.uint8)
    data[1:4, 2:5, 1:3] = 1
    mask = nib.Nifti1Image(data, image.affine)
    mask.header.set_xyzt_units("mm")
    meta = store.accept_mask(
        "asset_mask", gzip.compress(mask.to_bytes()), "asset_source", {"0": "background", "1": "actual_label"}
    )
    assert meta["measurements"][0]["voxel_count"] == 18
    assert meta["measurements"][0]["volume_mm3"] == pytest.approx(18 * 0.7 * 1.2 * 2.5, rel=1e-5)
    assert store.slice("asset_source", "axial", 2, mask_id="asset_mask") != store.slice(
        "asset_source", "axial", 2
    )
    assert store.slice("asset_source", "axial", 2, mask_id="asset_mask", labels=[]) == store.slice(
        "asset_source", "axial", 2
    )
    shifted = mask.affine.copy()
    shifted[0, 3] += 1
    with pytest.raises(AppError) as error:
        store.accept_mask(
            "asset_shifted",
            nib.Nifti1Image(data, shifted).to_bytes(),
            "asset_source",
            {"0": "background", "1": "actual_label"},
        )
    assert error.value.code == "MASK_SPACE_MISMATCH"
    with pytest.raises(AppError) as error:
        store.accept_mask(
            "asset_badlabels", mask.to_bytes(), "asset_source", {"0": "background", "2": "another_label"}
        )
    assert error.value.code == "MASK_LABEL_MISMATCH"


def test_unknown_units_do_not_become_mm3(store):
    source = volume()
    source.header.set_xyzt_units("unknown")
    store.save_volume("asset_source", source)
    mask = nib.Nifti1Image(np.ones(source.shape, dtype=np.uint8), source.affine)
    meta = store.accept_mask("asset_mask", mask.to_bytes(), "asset_source", {0: "background", 1: "structure"})
    assert "volume_mm3" not in meta["measurements"][0]


def dicom_bytes(z, uid, patient="PRIVATE_TEST_NAME"):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = uid
    ds.PatientName = patient
    ds.PatientID = "PRIVATE_TEST_ID"
    ds.Rows = 6
    ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [5, 10, z]
    ds.PixelSpacing = [1.2, 0.7]
    ds.Modality = "MR"
    ds.PixelData = (np.arange(48, dtype=np.uint16).reshape(6, 8) + int(z)).tobytes()
    output = io.BytesIO()
    ds.save_as(output, enforce_file_format=True)
    return output.getvalue()


def test_dicom_series_uses_lps_to_ras_and_true_spacing(store):
    uid = generate_uid()
    parts = [(f"scan{i}.dcm", dicom_bytes(z, uid)) for i, z in enumerate([8, 2, 5])]
    _, _, _, _, meta = store.prepare("asset_dicom", parts, "T2")
    assert meta["dimensions"] == [8, 6, 3]
    assert meta["modality"] == "T2"
    assert meta["acquisition_modality"] == "MR"
    assert np.allclose(meta["spacing"], [0.7, 1.2, 3])
    source = store.load("asset_dicom", display=False)
    assert np.allclose(source.affine[:3, 3], [-5, -10, 2])
    assert np.allclose(source.affine[:3, :3], np.diag([-0.7, -1.2, 3]))
    assert "PRIVATE_TEST" not in str(meta)


def test_dicom_single_is_2d_and_mixed_missing_slices_rejected(store):
    uid = generate_uid()
    _, _, _, _, meta = store.prepare("asset_single", [("one.dcm", dicom_bytes(2, uid))])
    assert meta["kind"] == "dicom_slice" and meta["slice_count"] == 1
    assert not store.storage.path("asset_single.volume.nii.gz").exists()
    for parts in [
        [("a", dicom_bytes(2, uid)), ("b", dicom_bytes(5, generate_uid()))],
        [("a", dicom_bytes(2, uid)), ("b", dicom_bytes(5, uid)), ("c", dicom_bytes(11, uid))],
    ]:
        with pytest.raises(AppError):
            store.prepare("asset_invalid", parts)


def test_bounded_zip_and_gzip_and_2d_nifti(store):
    store.settings.imaging_max_expanded_bytes = 1024
    with pytest.raises(AppError):
        store.nifti(gzip.compress(b"a" * 2048))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../../escaped", b"a" * 2048)
    with pytest.raises(AppError):
        store.archive(output.getvalue())
    assert not store.storage.root.parent.joinpath("escaped").exists()
    with pytest.raises(AppError):
        store.nifti(nib.Nifti1Image(np.zeros((4, 4), dtype=np.float32), np.eye(4)).to_bytes())


def gateway(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n- id: fixture-t2\n  name: Synthetic fixture\n  version: fixture-1\n  task_types: [segmentation]\n  modalities: [T2]\n  channels: 1\n  labels: {0: background, 1: fixture_label}\n"
    )
    return SegmentationGateway(
        Settings(
            _env_file=None,
            jwt_secret="synthetic-gateway-" * 4,
            segmentation_base_url="http://synthetic-service",
            segmentation_models_config=config,
        )
    )


def test_segmentation_unconfigured_and_input_compatibility(tmp_path):
    g = gateway(tmp_path)
    meta = {"kind": "volume", "dimensions": [4, 4, 4], "modality": "T2", "channels": 1}
    assert g.validate(meta, "fixture-t2", "segmentation").id == "fixture-t2"
    for key, value, code in [
        ("kind", "image", "NOT_3D_VOLUME"),
        ("modality", "T1", "MODALITY_MISMATCH"),
        ("channels", 2, "CHANNEL_MISMATCH"),
    ]:
        with pytest.raises(AppError) as error:
            g.validate({**meta, key: value}, "fixture-t2", "segmentation")
        assert error.value.code == code
    with pytest.raises(AppError):
        g.validate(meta, "arbitrary-model", "segmentation")
    g.settings.segmentation_base_url = ""
    with pytest.raises(AppError) as error:
        g.validate(meta, "fixture-t2", "segmentation")
    assert error.value.code == "SEGMENTATION_UNCONFIGURED"


@pytest.mark.parametrize("failure", [False, True])
async def test_independent_service_contract_and_failure(tmp_path, failure):
    g = gateway(tmp_path)
    path = tmp_path / "image.nii.gz"
    path.write_bytes(b"synthetic-volume")
    phases = AsyncMock()

    def handler(request):
        if request.method == "POST":
            assert request.headers["Idempotency-Key"] == "job-fixture"
            assert b"synthetic-volume" in request.content
            return httpx.Response(202, json={"task_id": "fixture-task"})
        if request.url.path.endswith("/mask"):
            return httpx.Response(200, content=b"synthetic-mask")
        return httpx.Response(
            200,
            json={
                "status": "failed" if failure else "completed",
                "model_id": "fixture-t2",
                "model_version": "fixture-1",
            },
        )

    if failure:
        with pytest.raises(AppError) as error:
            await g.infer(
                path,
                g.models["fixture-t2"],
                "segmentation",
                "job-fixture",
                phases,
                httpx.MockTransport(handler),
            )
        assert error.value.code == "SEGMENTATION_FAILED"
    else:
        assert (
            await g.infer(
                path,
                g.models["fixture-t2"],
                "segmentation",
                "job-fixture",
                phases,
                httpx.MockTransport(handler),
            )
            == b"synthetic-mask"
        )
        phases.assert_any_await("queued", "fixture-task")
        phases.assert_any_await("postprocessing", "fixture-task")


@pytest.mark.parametrize("code", ["CUDA_OUT_OF_MEMORY", "untrusted-error"])
async def test_remote_cuda_oom_is_actionable_without_exposing_private_errors(tmp_path, code):
    g = gateway(tmp_path)
    path = tmp_path / "image.nii.gz"
    path.write_bytes(b"synthetic-volume")

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"task_id": "fixture-task"})
        assert not request.url.path.endswith("/mask")
        return httpx.Response(
            200,
            json={"status": "failed", "error_code": code, "error_message": "private-patient-detail-fixture"},
        )

    with pytest.raises(AppError) as error:
        await g.infer(
            path,
            g.models["fixture-t2"],
            "segmentation",
            "job-fixture",
            AsyncMock(),
            httpx.MockTransport(handler),
        )
    assert "private-patient" not in error.value.message
    if code == "CUDA_OUT_OF_MEMORY":
        assert error.value.code == "SEGMENTATION_CUDA_OUT_OF_MEMORY"
        assert "显存不足" in error.value.message
    else:
        assert error.value.code == "SEGMENTATION_FAILED"


async def test_conversation_job_filter_uses_portable_json_string_expression():
    from types import SimpleNamespace
    from contextlib import asynccontextmanager
    from unittest.mock import patch
    from sqlalchemy.dialects import postgresql
    from pharma_agent.imaging_api import conversation_jobs
    from pharma_agent.security import Actor

    captured = []

    class Session:
        async def scalars(self, query):
            captured.append(query)
            return SimpleNamespace(all=lambda: [])

    @asynccontextmanager
    async def sessions():
        yield Session()

    svc = SimpleNamespace(sessions=sessions)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=svc)))
    with patch("pharma_agent.imaging_api.require_owned", new=AsyncMock()):
        result = await conversation_jobs(
            "conv_fixture", request, Actor("owner_fixture", "tenant_fixture", "member")
        )
    assert result == []
    compiled = captured[0].compile(dialect=postgresql.dialect())
    assert "->>" in str(compiled) and "tenant_id" in str(compiled) and "user_id" in str(compiled)
    assert {"owner_fixture", "tenant_fixture", "conv_fixture"}.issubset(set(compiled.params.values()))


async def test_medical_volume_tool_never_uses_an_unattached_asset():
    from pharma_agent.schemas import SegmentationInput
    from pharma_agent.segmentation import create_segmentation_draft

    args = SegmentationInput(file_id="asset_" + "a" * 32, model_id="fixture", task_type="segmentation")
    with pytest.raises(AppError) as error:
        await create_segmentation_draft(None, None, args, "conv_fixture", "key", [])
    assert error.value.code == "ATTACHMENT_REQUIRED"


def test_mask_rejects_a_numeric_affine_with_different_physical_units(store):
    image = volume()
    store.save_volume("asset_source", image)
    mask = nib.Nifti1Image(np.ones(image.shape, dtype=np.uint8), image.affine)
    mask.header.set_xyzt_units("meter")
    with pytest.raises(AppError) as failure:
        store.accept_mask("asset_mask", mask.to_bytes(), "asset_source", {0: "background", 1: "structure"})
    assert failure.value.code == "MASK_SPACE_MISMATCH"


def test_dicom_library_diagnostics_do_not_expose_tag_values(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="pydicom"):
        logging.getLogger("pydicom").warning("Bad patient tag: %s", "PRIVATE_TEST_PATIENT_ID")
    assert "PRIVATE_TEST_PATIENT_ID" not in caplog.text
    assert "content omitted" in caplog.text


@pytest.mark.parametrize("offline", [False, True])
async def test_model_identity_comes_from_bound_role_and_health(offline):
    from types import SimpleNamespace
    from pharma_agent.imaging_api import model_info

    bound = {
        "provider": "openai_compatible",
        "model_name": "bound-model-fixture",
        "revision": "revision-fixture",
    }
    models = SimpleNamespace(
        snapshot=lambda: {
            "roles": {"medical_answer": "bound"},
            "models": {"bound": bound},
            "config_version": "fixture",
        },
        health=AsyncMock(
            side_effect=AppError("MODEL_UNAVAILABLE", "synthetic unavailable", 503) if offline else None
        ),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                services=SimpleNamespace(models=models, graph_policy={"version": "fixture"})
            )
        )
    )
    result = await model_info(request, None)
    assert result["model_id"] == "bound-model-fixture"
    assert result["revision"] == "revision-fixture"
    assert result["status"] == ("unavailable" if offline else "ready")
