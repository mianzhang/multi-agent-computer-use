import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import vm_utils as vu


@pytest.mark.parametrize(
    ("system_name", "expected"),
    [
        ("Linux", ["-T", "ws"]),
        ("Windows", ["-T", "ws"]),
        ("Darwin", ["-T", "fusion"]),
    ],
)
def test_vmrun_type_args_by_platform(monkeypatch, system_name, expected):
    monkeypatch.setattr(vu.platform, "system", lambda: system_name)
    assert vu._vmrun_type_args() == expected


def test_vmrun_type_args_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setattr(vu.platform, "system", lambda: "Plan9")
    with pytest.raises(RuntimeError, match="Unsupported OS for vmrun: Plan9"):
        vu._vmrun_type_args()


def test_list_vm_snapshots_filters_noise(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vu,
        "_run_vmrun",
        lambda args, command, label, timeout=60: SimpleNamespace(
            stdout="\nWarning: ignore me\nTotal snapshots: 2\n snap_a \n\nsnap_b\n"
        ),
    )

    snapshots = vu._list_vm_snapshots(
        argparse.Namespace(osworld_root=str(tmp_path)),
        "/vm/example.vmx",
        "task-a",
    )

    assert snapshots == ["snap_a", "snap_b"]


def test_orchestrator_alloc_and_release_vm_updates_registry(tmp_path):
    args = argparse.Namespace(osworld_root=str(tmp_path))
    registry_path = tmp_path / vu.VMWARE_REGISTRY_FILENAME
    registry_path.write_text("pool/a.vmx|free\npool/b.vmx|busy\n", encoding="utf-8")

    chosen = vu._orchestrator_alloc_vm(args)

    assert chosen == "pool/a.vmx"
    allocated = registry_path.read_text(encoding="utf-8")
    assert "pool/a.vmx|free" not in allocated

    vu._orchestrator_release_vm(args, chosen)
    released = registry_path.read_text(encoding="utf-8")
    assert "pool/a.vmx|free" in released


def test_take_subtask_snapshot_reuses_existing_snapshot_on_no_change(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vu,
        "_create_vm_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("The state of the virtual machine has not changed")
        ),
    )
    monkeypatch.setattr(vu, "_list_vm_snapshots", lambda args, vm_path, label: ["snap_old", "snap_latest"])

    snap_name = vu._take_subtask_snapshot(
        argparse.Namespace(osworld_root=str(tmp_path)),
        "/vm/example.vmx",
        "subtask_1",
    )

    assert snap_name == "snap_latest"


def test_delete_vm_tree_removes_overlay_and_sidecar(tmp_path):
    overlay = tmp_path / "overlay.qcow2"
    sidecar = Path(str(overlay) + ".conn.json")
    overlay.write_text("overlay", encoding="utf-8")
    sidecar.write_text("{}", encoding="utf-8")

    deleted = vu._delete_vm_tree(str(overlay), "cleanup")

    assert deleted is True
    assert not overlay.exists()
    assert not sidecar.exists()


def test_take_initial_vm_screenshot_keeps_successful_apptainer_overlay(monkeypatch, tmp_path):
    overlay_path = tmp_path / "scout_overlay.qcow2"
    overlay_path.write_text("overlay", encoding="utf-8")
    killed = []
    captured = {}

    monkeypatch.setattr(vu, "_apptainer_create_overlay", lambda label="": str(overlay_path))
    monkeypatch.setattr(vu, "_apptainer_kill_qemu_for_overlay", lambda path, label: killed.append((path, label)))

    def fake_take_vm_screenshot(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output"]).write_text("png-bytes", encoding="utf-8")
        return True

    monkeypatch.setattr(vu, "take_vm_screenshot", fake_take_vm_screenshot)

    args = argparse.Namespace(
        python="/usr/bin/python3",
        osworld_root=str(tmp_path / "osworld"),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    screenshot, reusable_overlay = vu._take_initial_vm_screenshot(
        canonical_config={"id": "task-1"},
        task_id="task-1",
        args=args,
        cua_args=["--provider_name", "apptainer", "--headless", "--env_ready_wait_seconds", "12"],
        run_dir=run_dir,
    )

    assert screenshot == run_dir / "initial_screenshot.png"
    assert reusable_overlay == str(overlay_path)
    assert killed == [(str(overlay_path), "scout_task-1")]
    assert not (run_dir / "_scout_config.json").exists()
    assert screenshot.exists()
    assert captured == {
        "osworld_root": str(tmp_path / "osworld"),
        "provider_name": "apptainer",
        "path_to_vm": str(overlay_path),
        "task_config": {"id": "task-1"},
        "output": run_dir / "initial_screenshot.png",
        "wait_seconds": 12.0,
        "headless": True,
    }


def test_take_initial_vm_screenshot_discards_failed_apptainer_overlay(monkeypatch, tmp_path):
    overlay_path = tmp_path / "failed_overlay.qcow2"
    overlay_path.write_text("overlay", encoding="utf-8")

    monkeypatch.setattr(vu, "_apptainer_create_overlay", lambda label="": str(overlay_path))
    monkeypatch.setattr(vu, "_apptainer_kill_qemu_for_overlay", lambda path, label: None)
    monkeypatch.setattr(vu, "take_vm_screenshot", lambda **_kwargs: False)

    args = argparse.Namespace(
        python="/usr/bin/python3",
        osworld_root=str(tmp_path / "osworld"),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    screenshot, reusable_overlay = vu._take_initial_vm_screenshot(
        canonical_config={"id": "task-1"},
        task_id="task-1",
        args=args,
        cua_args=["--provider_name", "apptainer"],
        run_dir=run_dir,
    )

    assert screenshot is None
    assert reusable_overlay is None
    assert not overlay_path.exists()


def test_read_vm_info_from_dir_reads_valid_json(tmp_path):
    subtask_dir = tmp_path / "subtask"
    subtask_dir.mkdir(parents=True, exist_ok=True)
    (subtask_dir / "vm_info.json").write_text(
        '{"vm_path": "/vm/test.vmx", "provider": "vmware", "server_port": 8000}',
        encoding="utf-8",
    )

    info = vu.read_vm_info_from_dir(subtask_dir, "subtask")

    assert info == {
        "vm_path": "/vm/test.vmx",
        "provider": "vmware",
        "server_port": 8000,
    }


def test_reset_vm_pool_invokes_direct_revert(monkeypatch, tmp_path):
    captured = {}

    def fake_revert_vm_pool(registry_path, snapshot_name, timeout, log, cwd):
        captured["registry_path"] = registry_path
        captured["snapshot_name"] = snapshot_name
        captured["timeout"] = timeout
        captured["log"] = log
        captured["cwd"] = cwd
        return 2, 0

    monkeypatch.setattr(vu, "revert_vm_pool", fake_revert_vm_pool)

    args = argparse.Namespace(
        python="/usr/bin/python3",
        osworld_root=str(tmp_path / "osworld"),
        result_dir=str(tmp_path / "results"),
        reset_pool_snapshot="init_state",
        reset_pool_timeout=123,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    vu.reset_vm_pool(args, run_dir=run_dir)

    assert captured["registry_path"] == tmp_path / "osworld" / vu.VMWARE_REGISTRY_FILENAME
    assert captured["snapshot_name"] == "init_state"
    assert captured["cwd"] == str(tmp_path / "osworld")
    assert captured["timeout"] == 123
