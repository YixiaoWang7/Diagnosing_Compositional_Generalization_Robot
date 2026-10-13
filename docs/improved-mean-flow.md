# Improved MeanFlow

Select the shared-head iMF objective in an existing Transformer policy configuration:

```json
"algorithm": {
  "type": "improved_mean_flow",
  "num_inference_steps": 1,
  "logit_normal_loc": -0.4,
  "logit_normal_scale": 1.0,
  "data_proportion": 0.5,
  "norm_p": 1.0,
  "norm_eps": 0.01
}
```

Training, checkpoint recovery and `load_policy` use the same public APIs as FM.
`InferenceConfig(num_inference_steps=2)` selects two iMF jumps without changing
the checkpoint's algorithm. Untagged algorithm configurations still select FM;
new FM configurations can explicitly set `"type": "flow_matching"`.

## Objective and architecture

This implements Algorithm 1 from [Improved Mean Flows](https://arxiv.org/abs/2512.02012),
using its boundary prediction instead of an auxiliary velocity head. For clean
actions `a`, Gaussian noise `e`, and `0 <= r <= t <= 1`:

```text
z = (1-t)*a + t*e
v = u(z, t, t)
u, derivative = jvp(u, (z,t,r), (v,1,0))
V = u + (t-r)*stop_gradient(derivative)
error = sum((V - (e-a))**2)  # per action sequence
loss = mean(error / stop_gradient((error + norm_eps)**norm_p))
```

The JVP tangent is the predicted instantaneous velocity, which distinguishes iMF
from original MeanFlow. The interval `t-r` remains inside the differentiated
function. Half the batch uses `r=t` by default to train the boundary prediction.
With adaptive weighting, the scalar loss is typically close to one; compare
task performance, not its numerical magnitude against FM MSE.

The existing Transformer receives a separate interval MLP, added to time
conditioning before AdaLN. State remains exclusive to AdaLN; visual/query context
uses cross-attention. Shared layers retain their seeded initialization and the
interval MLP's final layer starts at zero. Convolutional heads are not supported.

Only iMF loss evaluation uses FP32 math attention for forward-mode AD. Observation
encoding retains the training AMP setting and is computed once; gradients still
reach trainable encoders. Boundary prediction and JVP reuse the same dropout draw.
Validation works under `torch.no_grad()`. Inference uses ordinary attention and
needs no derivatives or training data.

Sampling uses `z = z - (t-r)*u(z,t,r)` from noise at time 1 to actions at time 0.
FM retains its original opposite time convention. DINO preprocessing and encoding
costs remain part of full-policy latency even with one action-field evaluation.

The starting time/loss settings follow the
[official JAX implementation](https://github.com/Lyy-iiis/imeanflow/blob/main/imf.py).
Classifier-free guidance, its conditioning tokens, and the auxiliary output head
are outside this implementation. No additional dependencies are required.
