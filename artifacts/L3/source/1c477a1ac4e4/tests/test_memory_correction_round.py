"""Resource and scientific controls for the autonomous correction round."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('memory_round_under_test', SCRIPTS/'run_memory_correction_round.py')
round_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round_module)


def arguments(tmp_path):
    return SimpleNamespace(out=tmp_path/'round', campaign=tmp_path/'campaign',
                           data=tmp_path/'base', panel=tmp_path/'panel.json')


def test_paired_finetunes_share_initial_policy_and_budget(tmp_path):
    args = arguments(tmp_path)
    checkpoint = tmp_path/'initial.pt'
    for corrected in (False, True):
        cmd = round_module.train_command(args, str(corrected), checkpoint, args.data, corrected, True)
        assert cmd[cmd.index('--warm-start')+1] == str(checkpoint)
        assert cmd[cmd.index('--steps')+1] == '20000'
        assert cmd[cmd.index('--batch')+1] == '1024'
        assert cmd[cmd.index('--seed')+1] == '0'
        assert cmd[cmd.index('--correction-fraction')+1] == ('.25' if corrected else '0')
        assert cmd[cmd.index('--stop-after')+1] == '200'
        full = round_module.train_command(args, str(corrected), checkpoint, args.data, corrected, False)
        assert '--resume' in full and '--warm-start' not in full and '--stop-after' not in full
    collection = round_module.collect_command(args, checkpoint)
    assert collection[collection.index('--policy')+1] == str(checkpoint)
    assert collection[collection.index('--attempts')+1] == '100'
    assert collection[collection.index('--max-wall-seconds')+1] == '21600'


def test_failed_base_campaign_cannot_collect_or_train(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    round_module.write(args.campaign/'completion.json', dict(completed=False))
    runner = round_module.Round(args)
    calls = []
    monkeypatch.setattr(runner, 'start', lambda *a, **k: calls.append(a))
    try:
        with pytest.raises(RuntimeError, match='did not complete'):
            runner.run()
        assert not calls
    finally:
        runner.lock.close()


def test_fewer_than_eight_corrections_prevents_retraining(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    round_module.write(args.campaign/'completion.json', dict(completed=True))
    checkpoint = args.campaign/'training/L3b-c3-h16-aux/step-100000.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixture')
    runner = round_module.Round(args)
    calls = []
    def start(name, command, gpu=None):
        calls.append(name)
        round_module.write(args.out/'collection/completion.json', dict(completed=True, usable=True, accepted_episodes=7))
        return None
    monkeypatch.setattr(runner, 'start', start)
    monkeypatch.setattr(runner, 'finish', lambda _: None)
    try:
        runner.run()
        assert calls == ['collect']
        assert runner.state['status'] == 'insufficient_validated_corrections'
        assert not round_module.read(args.out/'completion.json')['completed']
    finally:
        runner.lock.close()


def test_existing_round_requires_review_before_restart(tmp_path):
    args = arguments(tmp_path)
    runner = round_module.Round(args)
    runner.lock.close()
    with pytest.raises(FileExistsError, match='inspect recorded jobs'):
        round_module.Round(args)
