"""Real CPU-only child IPC and synthetic NIfTI. No nnU-Net, torch, GPU or live services."""

import asyncio
import gzip
import io
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import nullcontext
from types import ModuleType

import httpx
import nibabel as nib
import numpy as np
import pytest

from pharma_agent.nnunet_service import create_segmentation_app
from pharma_agent.resident_model import ResidentModel
from pharma_agent.settings import Settings


@pytest.mark.parametrize("scenario", ["reuse", "cuda_oom", "crash", "wrong_request", "startup_failed"])
async def test_resident_http_lifecycle_reuses_process_and_never_fakes_readiness(
    tmp_path, monkeypatch, scenario
):
    model = tmp_path / "model"
    (model / "fold_4").mkdir(parents=True)
    (model / "fold_4/checkpoint_best.pth").write_bytes(b"never-loaded")
    manifest = {
        "id": "fixture",
        "version": "v1",
        "fold": 4,
        "checkpoint": "checkpoint_best.pth",
        "task_types": ["femoral_head_segmentation"],
    }
    (model / "model.json").write_text(json.dumps(manifest))
    helper = tmp_path / "fake_resident.py"
    helper.write_text("""import json,sys,shutil
def emit(**d): print('PHARMA_RESIDENT:'+json.dumps(d),flush=True)
mode=sys.argv[1]
print('SYNTHETIC_MODEL_LOADED_ONCE',flush=True)
if mode=='startup_failed':
 emit(event='startup_failed',error_code='MODEL_LOAD_FAILED');sys.exit(2)
emit(event='ready',model_id='fixture',model_version='v1')
for i,line in enumerate(sys.stdin):
 d=json.loads(line);rid=d['request_id']
 if mode=='crash':sys.exit(3)
 if mode=='cuda_oom' and i==0:
  emit(event='failed',request_id=rid,error_code='CUDA_OUT_OF_MEMORY');continue
 if mode=='wrong_request':
  emit(event='completed',request_id='wrong-task');continue
 for phase in ['preprocessing','inference','postprocessing']:emit(event='phase',request_id=rid,phase=phase)
 shutil.copyfile(d['input'],d['output']+'.nii.gz')
 emit(event='completed',request_id=rid)
""")
    launches = []

    def command(self):
        launches.append(self)
        return [sys.executable, "-u", str(helper), scenario]

    monkeypatch.setattr(ResidentModel, "command", command)
    settings = Settings(
        _env_file=None,
        jwt_secret="resident-test-jwt-" * 3,
        segmentation_api_key="resident-test-key-" * 3,
        nnunet_model_folder=model,
        nnunet_python=Path(sys.executable),
        segmentation_service_root=tmp_path / "tasks",
    )
    monkeypatch.setenv("NNUNET_DEVICE", "cpu")
    app = create_segmentation_app(settings)
    headers = {"Authorization": "Bearer " + settings.segmentation_api_key.get_secret_value()}
    image = nib.Nifti1Image(np.ones((4, 5, 6), np.uint8), np.diag([1, 2, 3, 1]))
    image.header.set_xyzt_units("mm")
    body = gzip.compress(image.to_bytes())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture", trust_env=False
        ) as client,
    ):
        assert (await client.get("/health")).status_code == 401
        for _ in range(100):
            response = await client.get("/health", headers=headers)
            if response.json()["status"] != "loading":
                break
            await asyncio.sleep(0.02)
        if scenario == "startup_failed":
            assert response.status_code == 503 and response.json()["error_code"] == "MODEL_LOAD_FAILED"
            return
        assert response.status_code == 200 and response.json()["load_mode"] == "resident"
        pid = launches[0].process.pid
        for index in range(2 if scenario in {"reuse", "cuda_oom"} else 1):
            fields = {
                "request_id": f"request-{index}",
                "model_id": "fixture",
                "task_type": "femoral_head_segmentation",
            }
            response = await client.post(
                "/v1/segmentations",
                files={"image": ("image.nii.gz", body)},
                data=fields,
                headers={**headers, "Idempotency-Key": fields["request_id"]},
            )
            assert response.status_code == 202
            task_id = response.json()["task_id"]
            for _ in range(100):
                state = (await client.get("/v1/segmentations/" + task_id, headers=headers)).json()
                if state["status"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.02)
            expected = (
                "completed" if scenario == "reuse" or scenario == "cuda_oom" and index == 1 else "failed"
            )
            assert state["status"] == expected
            if expected == "completed":
                mask = await client.get(f"/v1/segmentations/{task_id}/mask", headers=headers)
                parsed = nib.Nifti1Image.from_bytes(gzip.decompress(mask.content))
                assert parsed.shape == image.shape and np.allclose(parsed.affine, image.affine)
                assert parsed.header.get_xyzt_units()[0] == "mm"
            elif scenario == "cuda_oom":
                assert state["error_code"] == "CUDA_OUT_OF_MEMORY"
            else:
                assert state["error_code"] in {"RESIDENT_UNAVAILABLE", "RESIDENT_PROTOCOL_ERROR"}
                assert (await client.get("/health", headers=headers)).status_code == 503
                assert (
                    await client.get(f"/v1/segmentations/{task_id}/mask", headers=headers)
                ).status_code == 409
        assert len(launches) == 1 and launches[0].process.pid == pid
        assert (tmp_path / "tasks/resident.log").read_text().count("SYNTHETIC_MODEL_LOADED_ONCE") == 1
    assert launches[0].process.returncode is not None


def test_resident_ipc_rejects_paths_outside_private_case_directory(tmp_path, capsys):
    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/nnunet_case_runner.py"))
    session = SimpleNamespace(manifest={"id": "fixture", "version": "v1"}, device="cpu", predict=Mock())
    task = "seg_" + "a" * 32
    stream = io.StringIO(
        json.dumps(
            {
                "request_id": task,
                "input": str(tmp_path / "outside.nii.gz"),
                "output": str(tmp_path / task / "mask"),
            }
        )
        + "\n"
    )
    runner["resident_loop"](session, tmp_path.resolve(), stream)
    session.predict.assert_not_called()
    assert '"event": "failed"' in capsys.readouterr().out


def test_runner_preloads_network_on_device_once_before_reusing_it(tmp_path, monkeypatch):
    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/nnunet_case_runner.py"))
    manifest = {"id": "fixture", "version": "v1", "fold": 4, "checkpoint": "never-read.pth"}
    (tmp_path / "model.json").write_text(json.dumps(manifest))
    network = SimpleNamespace(eval=Mock())
    network.to = Mock(return_value=network)
    configuration = SimpleNamespace(
        preprocessor_class=lambda **kw: SimpleNamespace(run_case=lambda *a: (np.ones((1, 2, 2, 2)), None, {}))
    )
    predictor = SimpleNamespace(
        network=network,
        initialize_from_trained_model_folder=Mock(),
        configuration_manager=configuration,
        plans_manager=object(),
        dataset_json={},
        predict_logits_from_preprocessed_data=Mock(return_value=SimpleNamespace(cpu=lambda: "cpu-logits")),
    )
    torch = SimpleNamespace(
        set_num_threads=Mock(),
        set_num_interop_threads=Mock(),
        device=lambda d: d,
        from_numpy=lambda data: data,
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            synchronize=Mock(), mem_get_info=lambda: (1000000, 2000000), memory_allocated=lambda: 1000
        ),
    )
    export = Mock()
    modules = {
        "nnunetv2": ModuleType("nnunetv2"),
        "nnunetv2.inference": ModuleType("nnunetv2.inference"),
        "nnunetv2.inference.predict_from_raw_data": SimpleNamespace(
            nnUNetPredictor=Mock(return_value=predictor)
        ),
        "nnunetv2.inference.export_prediction": SimpleNamespace(export_prediction_from_logits=export),
        "torch": torch,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    model = runner["ModelSession"](tmp_path, "cuda", preload=True)
    network.to.assert_called_once_with("cuda")
    torch.cuda.synchronize.assert_called_once()
    for _ in range(2):
        model.predict(tmp_path / "input.nii.gz", tmp_path / "mask", Mock())
    predictor.initialize_from_trained_model_folder.assert_called_once()
    assert network.to.call_count == 1 and export.call_count == 2
