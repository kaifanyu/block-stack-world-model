import json
from pathlib import Path

import pytest
import torch

from assignment.training.training.distill_policy import DistillPolicy, load_distill_policy, rung_config
from assignment.training.training.policy_names import canonical_name, output_directory
from assignment.training.training.train_distill import parser
from scripts.run_fm_trial import command


@pytest.mark.parametrize('old,new', [('L2','privilegeFM'), ('L4','encoderFM'),
                                     ('L3a','simulatorFM-a'), ('L3b','simulatorFM')])
def test_old_checkpoint_and_new_name_identical(tmp_path, old, new):
    torch.set_num_threads(1)
    old_config = rung_config(old, dim=32, depth=1, heads=4, mlp=64, horizon=2)
    new_config = rung_config(new, dim=32, depth=1, heads=4, mlp=64, horizon=2)
    assert old_config == new_config
    model = DistillPolicy(old_config).eval()
    # Historical files omit newer optional fields and retain legacy rung metadata.
    config = dict(vars(old_config))
    for key in ['input_clip','vision_mode','auxiliary_dim','auxiliary_loss_weight','encoder_lr_scale']:
        config.pop(key, None)
    path = tmp_path/'policy.pt'
    torch.save(dict(policy_kind='distill_flow', config=config, state_dict=model.state_dict(),
                    info={'rung': old}), path)
    loaded, info = load_distill_policy(path)
    renamed = DistillPolicy(new_config).eval()
    renamed.load_state_dict(loaded.state_dict(), strict=True)
    inputs = dict(z=torch.randn(1,1,192), priv=torch.randn(1,190),
                  proprio=torch.randn(1,25), targets=torch.randn(1,25),
                  prev_actions=torch.randn(1,2,125), prev_mask=torch.ones(1,2,dtype=torch.bool))
    noise = torch.randn(1,2,125)
    torch.testing.assert_close(loaded.sample(inputs,steps=2,n_steps=2,noise=noise),
                               renamed.sample(inputs,steps=2,n_steps=2,noise=noise),rtol=0,atol=0)
    assert info['rung'] == old  # Preserve provenance rather than rewriting files.
    assert parser().parse_args(['--data','data','--rung',old,'--run-name','trial']).rung == new


def test_presets_and_output_paths():
    root = Path(__file__).resolve().parents[1]
    for name, preset in json.loads((root/'configs/fm_trials.json').read_text()).items():
        args = parser().parse_args(command(preset, 'data', name)[3:])
        assert args.rung in ('privilegeFM','encoderFM')
        assert output_directory(root/'analysis', args.rung, name).name == name
    assert output_directory('analysis','L4','control') == Path('analysis/Encoder FM Policy/control')
    with pytest.raises(ValueError):
        output_directory('analysis','L2','../escape')
