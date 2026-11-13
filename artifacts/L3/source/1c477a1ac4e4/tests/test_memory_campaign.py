"""Campaign orchestration checks; all child processes and GPU calls are mocked."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_memory_campaign.py'
SPEC = importlib.util.spec_from_file_location('memory_campaign_under_test', SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


@pytest.fixture
def args(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    campaign.write(data/'completion.json', dict(status='completed'))
    campaign.write(data/'manifest.json', dict(
        teacher_phase=dict(names=[str(i) for i in range(11)]),
        episodes=[dict(split='train')]*5000+[dict(split='val')]*100))
    panel = tmp_path/'panel.json'
    campaign.write(panel, dict(scenes=list(range(100))))
    return SimpleNamespace(out=tmp_path/'run', data=data, panel=panel,
                           steps=100000, smoke_steps=200, batch=1024, wait_gpu6=None)


@pytest.fixture
def queue(args):
    q = campaign.Campaign(args)
    yield q
    q.lock.close()


def complete_training(args, spec, stage):
    folder = args.out/'training'/spec['name']
    step = args.smoke_steps if stage == 'smoke' else args.steps
    campaign.write(folder/'completion.json', dict(status='stopped' if stage == 'smoke' else 'completed',
                                                step=step, completed=stage == 'train'))
    (folder/'training.pt').write_bytes(b'fixture')
    (folder/f'step-{step:06d}.pt').write_bytes(b'fixture')


def test_job_matrix_and_cli_preserve_controls(args):
    assert {r['gpu'] for r in campaign.RUNS} == {0, 1, 2, 4, 5, 6, 7}
    assert len({r['name'] for r in campaign.RUNS}) == 7
    for spec in campaign.RUNS:
        smoke = campaign.training_command(spec, args, smoke=True)
        full = campaign.training_command(spec, args)
        assert smoke[smoke.index('--history-blocks')+1] == str(spec['history'])
        assert smoke[smoke.index('--context')+1] == str(spec['context'])
        assert smoke[smoke.index('--phase-loss-weight')+1] == str(spec['auxiliary'])
        assert ('--phase-input' in smoke) == spec['phase']
        assert smoke[smoke.index('--stop-after')+1] == '200'
        assert '--resume' not in smoke and '--stop-after' not in full
        assert full[full.index('--resume')+1] == str(args.out/'training'/spec['name']/'training.pt')
        assert '--renderer' in campaign.evaluation_command(spec, args)
        assert campaign.evaluation_command(spec, args)[-1] == 'egl'


def test_launch_requires_smoke_and_all_checkpoint_artifacts(queue, monkeypatch):
    spec = campaign.RUNS[0]
    calls = []
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **k: calls.append((a, k)))
    with pytest.raises(RuntimeError, match='Invalid campaign transition'):
        queue.launch(spec, 'train', [])
    queue.state['runs'][spec['name']]['stage'] = 'smoke_passed'
    with pytest.raises(RuntimeError, match='missing completed smoke artifacts'):
        queue.launch(spec, 'train', [])
    complete_training(queue.args, spec, 'smoke')
    (queue.args.out/'training'/spec['name']/'training.pt').unlink()
    with pytest.raises(RuntimeError, match='missing completed smoke artifacts'):
        queue.launch(spec, 'train', [])
    assert not calls


def test_eval_requires_completed_final_checkpoint(queue, monkeypatch):
    spec = campaign.RUNS[0]
    queue.state['runs'][spec['name']]['stage'] = 'train_passed'
    complete_training(queue.args, spec, 'train')
    (queue.args.out/'training'/spec['name']/f'step-{queue.args.steps:06d}.pt').unlink()
    with pytest.raises(RuntimeError, match='missing completed train artifacts'):
        queue.launch(spec, 'eval', [])


def test_launch_records_process_and_rejects_duplicates(queue, monkeypatch):
    spec = campaign.RUNS[0]
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=4242)
    monkeypatch.setattr(campaign.subprocess, 'Popen', popen)
    monkeypatch.setattr(campaign.psutil, 'Process', lambda pid: SimpleNamespace(create_time=lambda: 123.))
    command = campaign.training_command(spec, queue.args, smoke=True)
    queue.launch(spec, 'smoke', command)
    with pytest.raises(RuntimeError, match='duplicate launch'):
        queue.launch(spec, 'smoke', command)
    assert len(calls) == 1
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '0'
    assert calls[0][1]['start_new_session'] is True
    row = campaign.read(queue.path)['runs'][spec['name']]
    assert row['processes']['smoke']['command'] == command
    assert row['processes']['smoke']['create_time'] == 123.


def test_gpu3_cannot_be_launched_even_with_invalid_spec(queue):
    spec = dict(campaign.RUNS[0], gpu=3)
    with pytest.raises(RuntimeError, match='GPU 3 is reserved'):
        queue.launch(spec, 'smoke', [])


@pytest.mark.parametrize('missing_checkpoint,bad_exit', [(True, False), (False, True)])
def test_failed_smoke_blocks_full_training(queue, missing_checkpoint, bad_exit):
    spec = campaign.RUNS[0]
    row = queue.state['runs'][spec['name']]
    row.update(stage='smoke', processes={'smoke': dict(pid=99, create_time=1.)})
    queue.handles[(spec['name'], 'smoke')] = SimpleNamespace(poll=lambda: 1 if bad_exit else 0)
    complete_training(queue.args, spec, 'smoke')
    if missing_checkpoint:
        (queue.args.out/'training'/spec['name']/'step-000200.pt').unlink()
    queue.poll(spec)
    assert row['stage'] == 'failed'
    with pytest.raises(RuntimeError, match='Review failed jobs'):
        queue.launch(spec, 'train', [])


def test_restart_adopts_live_process_and_advances_completed_smoke(queue, monkeypatch):
    spec = campaign.RUNS[0]
    row = queue.state['runs'][spec['name']]
    row.update(stage='smoke', processes={'smoke': dict(pid=99, create_time=1.)})
    queue.save()
    queue.lock.close()
    restarted = campaign.Campaign(queue.args)
    try:
        monkeypatch.setattr(campaign, 'live', lambda record: True)
        restarted.poll(spec)
        assert restarted.state['runs'][spec['name']]['stage'] == 'smoke'
        complete_training(queue.args, spec, 'smoke')
        monkeypatch.setattr(campaign, 'live', lambda record: False)
        restarted.poll(spec)
        assert restarted.state['runs'][spec['name']]['stage'] == 'smoke_passed'
        assert restarted.completed_artifacts(spec, 'smoke')
    finally:
        restarted.lock.close()


def test_running_queue_skips_occupied_gpu_and_baseline_reservation(queue, monkeypatch):
    calls = []
    queue.args.wait_gpu6 = queue.args.out/'baseline'
    monkeypatch.setattr(campaign, 'gpu_occupants', lambda: {i: [777] if i in (0, 3) else [] for i in range(8)})
    def launch(spec, stage, command):
        calls.append((spec['gpu'], stage))
        queue.state['runs'][spec['name']]['stage'] = stage
    monkeypatch.setattr(queue, 'launch', launch)
    class EndCycle(Exception):
        pass
    monkeypatch.setattr(campaign.time, 'sleep', lambda _: (_ for _ in ()).throw(EndCycle()))
    with pytest.raises(EndCycle):
        queue.run()
    assert {gpu for gpu, stage in calls} == {1, 2, 4, 5, 7}
    assert all(stage == 'smoke' for gpu, stage in calls)


def test_eval_completion_requires_actual_200_rollouts(queue):
    spec = campaign.RUNS[0]
    folder = queue.args.out/'evaluation'/spec['name']
    campaign.write(folder/'completion.json', dict(completed=True, episodes=200))
    campaign.write(folder/'results.json', dict(episodes=[{}]*199))
    assert not queue.completed_artifacts(spec, 'eval')
    campaign.write(folder/'results.json', dict(episodes=[{}]*200))
    assert queue.completed_artifacts(spec, 'eval')
