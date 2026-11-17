"""Exact CUDA-graph replay of a policy's fixed-shape flow sampler.

At evaluation batch sizes (1-8 requests, 12 tokens each) the flow policy is
launch-bound: every Euler step issues ~100 small kernels and the GPU idles
between them. Capturing one whole `model.sample` call (image CNN, prefix and
every Euler step) as a CUDA graph removes the per-kernel CPU cost without
changing any arithmetic: replay runs the same kernels on the same values.

One graph is captured per input signature (batch size, input shapes/dtypes,
generated blocks, flow steps). Replay copies the request tensors into that
graph's static buffers. Noise must be supplied by the caller so that no random
number generation is frozen into the graph.
"""
from __future__ import annotations

import torch


class CudaGraphSampler:
    def __init__(self, model, device, autocast_dtype=torch.bfloat16, warmup=3):
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('CUDA-graph sampling requires a CUDA device')
        if device.index is None:
            # Tensors always report an indexed device; 'cuda' alone would never match.
            device = torch.device('cuda', torch.cuda.current_device())
        if model.training:
            raise ValueError('CUDA-graph sampling requires an eval-mode model')
        if any(p.requires_grad for p in model.parameters()):
            # Autocast caches casts of grad-requiring weights; a cached cast
            # captured inside a graph would go stale after a weight update.
            raise ValueError('CUDA-graph sampling requires frozen parameters')
        self.model, self.device, self.autocast_dtype, self.warmup = model, device, autocast_dtype, warmup
        self.graphs = {}

    @staticmethod
    def signature(inputs, noise, steps, n_steps, options):
        tensors = tuple(sorted((name, tuple(value.shape), str(value.dtype)) for name, value in inputs.items()))
        return (int(steps), int(n_steps), tuple(noise.shape), str(noise.dtype), tensors, tuple(sorted(options.items())))

    def _autocast(self):
        return torch.autocast('cuda', self.autocast_dtype, enabled=self.autocast_dtype is not None,
                              cache_enabled=False)

    def _capture(self, inputs, noise, steps, n_steps, options):
        static_inputs = {name: value.clone() for name, value in inputs.items()}
        static_noise = noise.clone()

        def run():
            return self.model.sample(static_inputs, steps=steps, n_steps=n_steps, noise=static_noise, **options)

        # Warm up on a side stream so lazy initialization (cuBLAS handles,
        # autotuning) happens before capture, as the CUDA-graph docs require.
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), self._autocast():
            for _ in range(self.warmup):
                run()
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), self._autocast():
            output = run()
        return graph, static_inputs, static_noise, output

    @torch.no_grad()
    def sample(self, inputs, steps, n_steps, noise, **options):
        """options are extra keyword arguments of model.sample, e.g. solver/time_shift."""
        if noise is None:
            raise ValueError('CUDA-graph sampling needs caller-supplied noise')
        for value in (*inputs.values(), noise):
            if value.device != self.device:
                raise ValueError('All sampler inputs must already be on the graph device')
        key = self.signature(inputs, noise, steps, n_steps, options)
        if key not in self.graphs:
            self.graphs[key] = self._capture(inputs, noise, steps, n_steps, options)
        graph, static_inputs, static_noise, output = self.graphs[key]
        for name, value in inputs.items():
            static_inputs[name].copy_(value)
        static_noise.copy_(noise)
        graph.replay()
        # The static output is overwritten by the next replay of this graph.
        return output.clone()

    def stats(self):
        return dict(captured_graphs=len(self.graphs),
                    batch_sizes=sorted({key[2][0] for key in self.graphs}))
