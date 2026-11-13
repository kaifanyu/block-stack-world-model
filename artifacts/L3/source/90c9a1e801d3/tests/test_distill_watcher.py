"""Restart/placement safety for surviving independent GPU jobs."""
import importlib.util
import json
from pathlib import Path
import uuid

import pytest


def module():
    spec = importlib.util.spec_from_file_location('watch_distill', Path(__file__).resolve().parents[1]/'scripts/watch_distill.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_adopt_requires_command_and_process_identity(monkeypatch, tmp_path):
    watcher = module()
    completion = tmp_path/'completion.json'
    row = dict(pid=42, command=['python', 'train.py'], completion=str(completion), process_start_ticks=123)
    identity = dict(command=row['command'], start_ticks=123, parent=1)
    monkeypatch.setattr(watcher, 'process_identity', lambda pid: identity)
    child = watcher.AdoptedProcess(row)
    assert child.poll() is None
    identity['command'] = ['unrelated']
    with pytest.raises(RuntimeError):
        watcher.AdoptedProcess(row)
    identity['command'] = row['command']
    identity['start_ticks'] = 456
    with pytest.raises(RuntimeError):
        watcher.AdoptedProcess(row)


def test_adopt_completion_is_not_an_invented_os_exit_code(monkeypatch, tmp_path):
    watcher = module()
    path = tmp_path/'completion.json'
    row = dict(pid=42, command=['python', 'train.py'], completion=str(path))
    monkeypatch.setattr(watcher, 'process_identity', lambda pid: dict(command=row['command'], start_ticks=123, parent=1))
    child = watcher.AdoptedProcess(row)
    monkeypatch.setattr(watcher, 'process_identity', lambda pid: None)
    assert child.poll() == 1
    path.write_text(json.dumps(dict(completed=True, status='failed')))
    assert child.poll() == 1
    path.write_text(json.dumps(dict(completed=True, status='completed')))
    assert child.poll() == 0


def test_inference_gpu_reused_but_exclusive_or_external_processes_block(monkeypatch):
    watcher = module()
    state = {'infer': dict(kind='inference', gpu='2', pid=10)}
    running = {'infer': object()}
    info = dict(processes=[11], memory_used_mib=2000)
    monkeypatch.setattr(watcher, 'process_identity', lambda pid: dict(parent=10) if pid == 11 else None)
    assert watcher.available_training_gpu('2', info, state, running)
    assert not watcher.available_training_gpu('2', info, state, running, exclusive=True)
    info['processes'].append(99)
    assert not watcher.available_training_gpu('2', info, state, running)
    assert watcher.owned_process(11, {10})
    assert not watcher.owned_process(99, {10})


def test_gpu_lease_survives_colocated_training_exit():
    watcher = module()
    gpu_uuid = 'test-'+uuid.uuid4().hex
    first, second = watcher.GpuLeases(), watcher.GpuLeases()
    assert first.acquire('2', gpu_uuid, 'train')
    assert first.acquire('2', gpu_uuid, 'infer')
    first.release('2', 'train')
    assert not second.acquire('2', gpu_uuid, 'outside')
    first.release('2', 'infer')
    assert second.acquire('2', gpu_uuid, 'outside')
    second.release('2', 'outside')


def test_parallel_inference_uses_distinct_nonexclusive_training_gpus():
    watcher = module()
    state = dict(train_a=dict(kind='gpu', gpu='1', exclusive=False),
                 train_b=dict(kind='gpu', gpu='2', exclusive=True),
                 train_c=dict(kind='gpu', gpu='3', exclusive=False),
                 inference=dict(kind='inference', gpu='1'))
    running = {key: object() for key in state}
    assert watcher.inference_training_gpus(state, running) == ['3']
    assert not watcher.available_training_gpu('1', dict(processes=[], memory_used_mib=0),
                                             {'inference': state['inference']}, {'inference': object()}, exclusive=True)
