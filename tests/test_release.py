"""Portable release commands and trainable world-model checkpoint contracts."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

import wm_shared
wm_shared.bootstrap()
from act_jepa_model import ActJepaConfig, ActJepaPolicy, load_act_jepa, save_act_jepa
from tdmpc_model import TDMPCConfig, TDMPCWorldModel, load_tdmpc, save_tdmpc
from scripts.download_checkpoints import verify
from scripts.evaluate_policy import command


def test_manifest_defaults_and_corruption_detection(tmp_path):
    records = json.loads((Path(__file__).resolve().parents[1] / 'checkpoints.json').read_text())['checkpoints']
    assert len(records) == 7 and len({r['path'] for r in records}) == 7
    for record in records:
        if 'evaluation' in record:
            cmd = command(record, tmp_path / record['path'], tmp_path / 'eval')
            if record['id'] == 'rs-imle':
                assert cmd[cmd.index('--flow-steps') + 1] == '1'
            if record['id'] == 'rgb-fm':
                assert cmd[cmd.index('--time-shift') + 1] == '2.0'
            if record['id'] == 'td-mpc':
                assert cmd[cmd.index('--mode') + 1] == 'policy'
    p = tmp_path / 'weights.pt'
    p.write_bytes(b'known weights')
    record = dict(size_bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    verify(p, record)
    p.write_bytes(b'wrong weights')
    with pytest.raises(ValueError, match='SHA-256'):
        verify(p, record)


def test_world_models_train_and_reload_without_external_worktree(tmp_path, monkeypatch):
    monkeypatch.setenv('BLOCKSTACK_SRC', '/missing/other/checkout')
    assert wm_shared.bootstrap() == Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    torch.manual_seed(23)
    b, c, h, history = 2, 1, 2, 2
    inputs = dict(rgb=torch.randint(0, 256, (b, c, 128, 256, 3), dtype=torch.uint8),
                  prev_actions=torch.randn(b, history, 125), prev_mask=torch.ones(b, history, dtype=torch.bool))
    actions = torch.randn(b, h, 125)
    batch = dict(inputs, actions=actions, action_mask=torch.ones(b, h, dtype=torch.bool),
                 exec_actions=actions, auxiliary=torch.randn(b, 52), future_z=torch.randn(b, h, 192),
                 future_valid=torch.ones(b, h, dtype=torch.bool), future_stage=torch.zeros(b, h, dtype=torch.long))
    act = ActJepaPolicy(ActJepaConfig(context=c, horizon=h, n_prev_blocks=history, p_hist=0,
                                     dim=32, depth=1, heads=4, mlp=64, vision_mode='raw',
                                     auxiliary_loss_weight=.1, future_offsets=(1, 2), predictor_depth=1))
    loss = act.loss_components(batch, torch.Generator().manual_seed(7))['loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert act.wm_out.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in act.image_encoder.parameters())
    act.eval()
    save_act_jepa(tmp_path / 'act.pt', act, {})
    loaded, _ = load_act_jepa(tmp_path / 'act.pt')
    noise = torch.zeros(b, 1, 125)
    torch.testing.assert_close(act.sample(inputs, steps=1, n_steps=2, noise=noise),
                               loaded.sample(inputs, steps=1, n_steps=2, noise=noise), rtol=0, atol=0)

    td = TDMPCWorldModel(TDMPCConfig(context=c, n_prev_blocks=history, p_hist=0, dim=32, heads=4,
                                    mlp=64, encoder_depth=1, latent_dim=32, mlp_dim=32,
                                    prior_depth=1, rollout=h, chunk=h, num_q=2))
    batch.update(rgb_seq=torch.cat([inputs['rgb']] * (c+h), dim=1),
                 seq_prev_actions=inputs['prev_actions'][:, None].expand(-1, h+1, -1, -1),
                 seq_prev_mask=inputs['prev_mask'][:, None].expand(-1, h+1, -1),
                 seq_valid=torch.ones(b, h, dtype=torch.bool), seq_reward=torch.ones(b, h),
                 seq_done=torch.zeros(b, h, dtype=torch.bool), seq_auxiliary=torch.randn(b, h+1, 52),
                 seq_stage=torch.zeros(b, h+1, dtype=torch.long))
    loss = td.loss_components(batch, torch.Generator().manual_seed(7))['loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in td.dynamics.parameters())
    assert all(p.grad is None for p in td.target_qs.parameters())
    td.eval()
    save_tdmpc(tmp_path / 'td.pt', td, {})
    loaded, _ = load_tdmpc(tmp_path / 'td.pt')
    with torch.no_grad():
        tokens, latent = td.encode(inputs['rgb'], inputs['prev_actions'], inputs['prev_mask'])
        other, other_latent = loaded.encode(inputs['rgb'], inputs['prev_actions'], inputs['prev_mask'])
        torch.testing.assert_close(td.sample_prior(tokens, noise, steps=2),
                                   loaded.sample_prior(other, noise, steps=2), rtol=0, atol=0)
        torch.testing.assert_close(td.plan_value(latent, actions), loaded.plan_value(other_latent, actions), rtol=0, atol=0)
