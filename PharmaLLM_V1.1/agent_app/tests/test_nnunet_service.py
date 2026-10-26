"""Exercise the local HTTP contract with a fake subprocess; no network weights or GPU."""

import asyncio
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock

import httpx
import nibabel as nib
import numpy as np
import pytest

from pharma_agent.nnunet_service import create_segmentation_app
from pharma_agent.settings import Settings


@pytest.mark.parametrize("failure", [None, "generic", "cuda_oom"])
async def test_service_auth_idempotency_and_exported_mask(tmp_path, monkeypatch, failure):
    from pharma_agent import nnunet_service as module

    model = tmp_path / "model"
    (model / "fold_4").mkdir(parents=True)
    (model / "fold_4/checkpoint_best.pth").write_bytes(b"fixture-never-loaded")
    manifest = {
        "id": "fixture-t1",
        "version": "fixture-v1",
        "fold": 4,
        "checkpoint": "checkpoint_best.pth",
        "task_types": ["femoral_head_segmentation"],
    }
    (model / "model.json").write_text(json.dumps(manifest))
    settings = Settings(
        _env_file=None,
        jwt_secret="fixture-service-jwt-" * 3,
        segmentation_api_key="fixture-seg-secret-" * 3,
        nnunet_model_folder=model,
        nnunet_python=Path(sys.executable),
        segmentation_service_root=tmp_path / "tasks",
        segmentation_load_mode="per_task",
    )
    monkeypatch.setenv("NNUNET_DEVICE", "cpu")
    commands = []

    async def subprocess(*command, **kwargs):
        commands.append(command)
        assert "nnunet_case_runner.py" in command[2]
        assert command[-1] == "cpu" and kwargs["start_new_session"]
        source = nib.load(command[command.index("--input") + 1])
        if not failure:
            result = nib.Nifti1Image(np.ones(source.shape, dtype=np.uint8), source.affine)
            nib.save(result, command[command.index("--output") + 1] + ".nii.gz")
        stream = asyncio.StreamReader()
        stream.feed_data(b"PHARMA_STAGE:preprocessing\nPHARMA_STAGE:inference\nPHARMA_STAGE:postprocessing\n")
        if failure == "cuda_oom":
            stream.feed_data(b"PHARMA_ERROR:CUDA_OUT_OF_MEMORY\nprivate traceback fixture\n")
        stream.feed_eof()

        class Process:
            stdout = stream

            async def wait(self):
                return 1 if failure else 0

        return Process()

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", subprocess)
    monkeypatch.setattr(module, "stop_child", AsyncMock())
    app = create_segmentation_app(settings)
    image = nib.Nifti1Image(np.arange(5 * 7 * 4, dtype=np.float32).reshape(5, 7, 4), np.diag([1, 2, 3, 1]))
    image.header.set_xyzt_units("mm")
    body = gzip.compress(image.to_bytes())
    headers = {
        "Authorization": "Bearer " + settings.segmentation_api_key.get_secret_value(),
        "Idempotency-Key": "fixture-request",
    }
    fields = {
        "request_id": "fixture-request",
        "model_id": "fixture-t1",
        "task_type": "femoral_head_segmentation",
    }
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture", trust_env=False
        ) as client,
    ):
        assert (await client.get("/health")).status_code == 401
        assert (await client.get("/health", headers=headers)).status_code == 200
        response = await client.post(
            "/v1/segmentations",
            data=fields,
            files={"image": ("image.nii.gz", body, "application/gzip")},
            headers=headers,
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        for _ in range(30):
            state = (await client.get("/v1/segmentations/" + task_id, headers=headers)).json()
            if state["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert state["status"] == ("failed" if failure else "completed")
        again = await client.post(
            "/v1/segmentations", data=fields, files={"image": ("image.nii.gz", body)}, headers=headers
        )
        assert again.json()["task_id"] == task_id and len(commands) == 1
        mask = await client.get(f"/v1/segmentations/{task_id}/mask", headers=headers)
        if failure:
            expected = "CUDA_OUT_OF_MEMORY" if failure == "cuda_oom" else "NNUNET_INFERENCE_FAILED"
            assert mask.status_code == 409 and state["error_code"] == expected
            assert "private traceback" not in json.dumps(state)
        else:
            assert mask.status_code == 200
            parsed = nib.Nifti1Image.from_bytes(gzip.decompress(mask.content))
            assert parsed.shape == image.shape and np.allclose(parsed.affine, image.affine)
            assert parsed.header.get_xyzt_units()[0] == "mm"
            assert np.all(np.asarray(parsed.dataobj) == 1)
        other = nib.Nifti1Image(np.zeros(image.shape, dtype=np.float32), image.affine)
        other.header.set_xyzt_units("mm")
        conflict = await client.post(
            "/v1/segmentations",
            data=fields,
            files={"image": ("image.nii.gz", gzip.compress(other.to_bytes()))},
            headers=headers,
        )
        assert conflict.status_code == 409


async def test_stop_child_terminates_only_its_own_process_group():
    from pharma_agent.nnunet_service import stop_child

    victim = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(20)", start_new_session=True
    )
    independent = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(20)", start_new_session=True
    )
    try:
        await stop_child(victim)
        assert victim.returncode is not None
        assert independent.returncode is None
    finally:
        await stop_child(victim)
        await stop_child(independent)
