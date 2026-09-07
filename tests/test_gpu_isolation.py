import pytest

torch = pytest.importorskip("torch")

from dinogenept.cell import checkpoint, train  # noqa: E402


def test_gpu_guard_filters_only_selected_card_and_fails_closed(monkeypatch):
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.setattr(train.subprocess, "check_output", lambda *a, **kw: "GPU-a, 111\nGPU-b, 222\n")
    monkeypatch.setattr(train.os, "getpgid", lambda pid: 1 if pid in (0, 111) else 2)
    train._gpu_guard(selected_uuid="GPU-a")
    with pytest.raises(RuntimeError, match="unrelated active"):
        train._gpu_guard(selected_uuid="GPU-b")
    with pytest.raises(RuntimeError, match="unrelated active"):
        train._gpu_guard()  # DDP allocation still audits the complete two-card set.
    monkeypatch.setattr(train.subprocess, "check_output", lambda *a, **kw: "unrecognized driver output")
    with pytest.raises(RuntimeError, match="unrecognized row"):
        train._gpu_guard(selected_uuid="GPU-a")


def test_torchrun_peers_have_different_process_groups_but_verified_parent(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "fixture-run")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(train.os, "getppid", lambda: 100)
    monkeypatch.setattr(train.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(train.Path, "read_bytes", lambda path: b"/usr/bin/python\0/project/.venv/bin/torchrun\0")
    monkeypatch.setattr(train.Path, "read_text", lambda path: f"{path.parent.name} (python rank) S 100 222 222")
    monkeypatch.setattr(train.subprocess, "check_output", lambda *a, **kw: "GPU-a, 222\n")
    train._gpu_guard()
    # Matching PGID is not required, but unrelated parent / non-launcher is refused.
    monkeypatch.setattr(train.Path, "read_text", lambda path: "222 (python rank) S 999 222 222")
    with pytest.raises(RuntimeError, match="unrelated active"):
        train._gpu_guard()
    monkeypatch.setattr(train.Path, "read_text", lambda path: "222 (python rank) S 100 222 222")
    monkeypatch.setattr(train.Path, "read_bytes", lambda path: b"/bin/bash\0")
    with pytest.raises(RuntimeError, match="unrelated active"):
        train._gpu_guard()
    monkeypatch.setattr(train.Path, "read_bytes", lambda path: b"python\0-m\0torch.distributed.run\0")
    train._gpu_guard()


def test_rng_save_and_restore_never_touch_another_gpu(monkeypatch):
    saved, restored = [], []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def get(device):
        saved.append(device)
        return torch.tensor([1, 2, 3], dtype=torch.uint8)

    monkeypatch.setattr(torch.cuda, "get_rng_state", get)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: restored.append(device))
    state = checkpoint.rng_state()
    assert saved == [1] and state["cuda_device_indices"] == [1]
    checkpoint.restore_rng(state)
    assert restored == [1]
