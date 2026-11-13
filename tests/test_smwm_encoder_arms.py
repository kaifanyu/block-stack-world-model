"""Gradient, optimizer, and checkpoint contracts of the encoder-only arms."""
from pathlib import Path
import tempfile

import torch

from assignment.training.training.distill_policy import DistillPolicy, rung_config, save_distill_policy, load_distill_policy
from assignment.training.training.train_distill import build_optimizer, build_scheduler, update


def small(mode):
    torch.set_num_threads(1)
    return DistillPolicy(rung_config('L4', vision_mode=mode, auxiliary_loss_weight=.1,
                                    dim=32, depth=1, heads=4, mlp=64, horizon=2))


def test_adapter_auxiliary_trains_shared_features_without_labels_at_inference():
    model = small('adapter')
    inputs = dict(z=torch.randn(2, 1, 192), auxiliary=torch.randn(2, 52))
    features = model.visual_features(inputs)
    components = model.loss_components(inputs, torch.randn(2, 2, 125), torch.ones(2, 2, dtype=torch.bool))
    components['auxiliary_loss'].backward()
    assert model.visual_adapter[0].weight.grad.abs().sum() > 0
    assert all('encoder' not in name for name, _ in model.named_parameters())
    torch.testing.assert_close(features, model.visual_features({'z': inputs['z']}))
    model.eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'policy.pt'
        save_distill_policy(path, model, {})
        restored, _ = load_distill_policy(path)
        torch.testing.assert_close(restored.visual_features(inputs), features)


def test_finetune_gradients_lr_ratio_bn_and_self_contained_checkpoint():
    model = small('smwm_finetune')
    inputs = dict(rgb=torch.randint(0, 256, (1, 1, 128, 256, 3), dtype=torch.uint8),
                  auxiliary=torch.randn(1, 52))
    mean = model.image_encoder.projector.net[1].running_mean.clone()
    components = model.loss_components(inputs, torch.randn(1, 2, 125), torch.ones(1, 2, dtype=torch.bool))
    components['auxiliary_loss'].backward()
    assert model.image_encoder.encoder.embed.weight.grad.abs().sum() > 0
    assert model.image_encoder.projector.net[0].weight.grad.abs().sum() > 0
    torch.testing.assert_close(mean, model.image_encoder.projector.net[1].running_mean)
    optimizer = build_optimizer(model, lr=.001)
    encoder_ids = {id(p) for p in model.image_encoder.parameters()}
    for group in optimizer.param_groups:
        expected = .0001 if id(group['params'][0]) in encoder_ids else .001
        assert abs(group['lr']-expected) < 1e-10
        assert all((id(p) in encoder_ids) == (expected == .0001) for p in group['params'])
    model.eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'policy.pt'
        save_distill_policy(path, model, {})
        restored, _ = load_distill_policy(path)
        with torch.no_grad():
            torch.testing.assert_close(restored.visual_features(inputs), model.visual_features(inputs))
        assert not any('predictor' in name for name, _ in restored.named_parameters())
