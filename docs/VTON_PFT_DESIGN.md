# Mask-Constrained Patch Forcing for Virtual Try-On

This document describes the VTON implementation in this repository. It focuses on the time distribution, latent construction, model architecture, conditioning paths, losses, inference procedure, and the reasons the design should transfer a pretrained PFT-XL model without leaking the paired ground-truth garment.

## 1. Task definition

The training sample contains:

| Symbol | Batch key | Meaning |
|---|---|---|
| \(I^*\) | `image` | Complete paired ground-truth person image |
| \(I_p\) | `person` | Person image used to construct the agnostic input |
| \(I_a\) | `person_agnostic` | Person with the original garment removed |
| \(I_g\) | `garment` | In-shop garment image |
| \(M\) | `agnostic_mask` | Region allowed to change; 1 means editable |
| \(M_g\) | `garment_mask` | Foreground mask for garment attention tokens |

Paired training normally has \(I_p=I^*\). This is safe only if the complete person is not used as model context. In this implementation, the full image is used as the supervised target and final RGB compositing reference. The context supplied to PFT is produced exclusively from the expanded agnostic image.

At inference, \(I_p\) can contain a different old garment. The agnostic mask removes it before VAE encoding, while \(I_g\) provides the desired garment.

## 2. End-to-end data flow

```text
Paired target I* -------------------------> frozen VAE encoder ---> target latent z*

Person Ip ---> supplied agnostic mask M ---> token dilation
          ---> expanded RGB mask M+ ---> remove garment before VAE
          ---> frozen VAE encoder -------------------------------> agnostic latent za

Garment Ig ---> offline frozen DINOv2-small ---------------------> garment feature map fg
Garment mask Mg -------------------------------------------------> garment token key mask

Noise epsilon + z* + za + token times --------------------------> evolving latent xt

[xt, za, interpolated mask] ---> zero-initialized input extension
fg + Mg ----------------------> garment cross-attention
per-token time ---------------> per-token AdaLN modulation
                                     |
                              pretrained PFT-XL
                                     |
                              velocity + uncertainty
                                     |
                              final VAE latent
                                     |
                              frozen VAE decoder
                                     |
original person RGB + feathered edit mask ---> exact outside-mask composite
```

The full person latent is never used as clean context. This matters because masking a latent after encoding is insufficient: the VAE encoder has a spatial receptive field, so garment texture can already have spread into latent cells outside the nominal mask.

## 3. Mask representations

One mask representation is not suitable for every operation. The implementation derives three masks from the supplied agnostic mask.

### 3.1 Token schedule mask

The input mask is max-pooled to the PFT token grid. A token becomes editable if any source-mask pixel contributes to it:

\[
M_T = \operatorname{MaxPoolToTokenGrid}(M).
\]

Max pooling is deliberate. Bilinear or average resizing could turn a small missed garment region into a low fractional value and then classify the token as preserved.

The token mask is morphologically dilated:

\[
M_T^+ = \operatorname{Dilate}(M_T, r).
\]

Max-pooling first rounds the supplied mask outward to complete PFT tokens. The edit/time path then applies one additional token of dilation without random jitter. The person-conditioning path uses the undilated token envelope, preventing the edit expansion from erasing nearby identity evidence.

The split-mask configuration keeps the RGB agnostic condition masked only by the original removal mask. A separate edit mask receives one token of dilation and controls sampled times, noise, loss, sampler updates, garment-attention queries, and final compositing. The encoded person condition is cleared only under the undilated token mask, while the fixed flow context is cleared under the expanded edit mask. Identity evidence in the expansion ring therefore remains available as conditioning without being copied into the evolving state.

For 512-by-384 training, the VAE produces a 64-by-48 latent and PFT uses a 32-by-24 token grid. PFT-XL's pretrained convolutional patch projection transfers directly. Its frozen 16-by-16 positional embedding is bicubically interpolated to the rectangular grid, so rectangular training fine-tunes from the square pretrained model without changing checkpoint parameter shapes.

With the current SD VAE downsampling factor of 8 and PFT patch size 2, one PFT token covers approximately \(16\times16\) input pixels.

### 3.2 Hard latent update mask

Each token value is repeated over its \(2\times2\) latent cells:

\[
M_L = \operatorname{Repeat}_{2\times2}(M_T^+).
\]

This binary mask controls:

- which latent cells contain noise or target interpolation;
- which cells receive sampler updates;
- which cells contribute to the main flow loss;
- which agnostic latent cells are forced to zero;
- which cells are clamped back to agnostic context after every inference step.

### 3.3 Soft conditioning mask

A separate mask is area-resized to latent resolution and combined with a softened version of the expanded hard mask. This one-channel map is supplied to the network as a condition. It tells the model where the boundary lies without being responsible for the safety-critical update decision.

### 3.4 RGB compositing mask

After VAE decoding, the expanded mask is upsampled and feathered inward. The final output is

\[
I_{out}=\alpha I_{gen}+(1-\alpha)I_p.
\]

The support of \(\alpha\) is restricted to the expanded editable region. Therefore pixels outside that region are copied exactly from the original person image, even if the VAE decoder changes nearby pixels.

## 4. Preventing paired-data leakage

The following order is important.

1. Encode the target image to obtain its latent shape and the supervised target \(z^*=E(I^*)\).
2. Create the expanded mask from the supplied agnostic mask.
3. Upsample that expanded mask back to image resolution.
4. Apply it to the already agnostic RGB image:

   \[
   I_a^+=I_a\odot(1-M^+).
   \]

5. Encode only this expanded agnostic image:

   \[
   z_a=E(I_a^+).
   \]

6. Zero the editable cells again after encoding:

   \[
   z_a\leftarrow z_a\odot(1-M_L).
   \]

This gives two defenses:

- removal happens before the VAE, preventing encoder receptive-field leakage from the original garment;
- editable latent cells are cleared again after the VAE.

The target image must still be encoded because supervised flow matching needs a target. Target information is used only inside the editable region and only at the noise level described by the corresponding timestep. Seeing progressively cleaner target signal as \(t\to1\) is normal diffusion/flow training, not label leakage.

## 5. Rectified-flow convention

This repository uses the convention

\[
t=0 \quad\text{means noise},\qquad t=1 \quad\text{means clean data}.
\]

For target latent \(z^*\) and Gaussian noise \(\epsilon\), the straight interpolation is

\[
z(t)=t z^*+(1-t)\epsilon.
\]

Its target velocity is constant:

\[
u=\frac{d z(t)}{dt}=z^*-\epsilon.
\]

PFT differs from ordinary rectified flow because every spatial token can have its own timestep.

## 6. Patchwise time sampling

### 6.1 Global ceiling

The configured `LogitNormalTruncatedGaussian` first samples a per-image time ceiling:

\[
\bar t=\sigma(\mu+s\eta),\qquad \eta\sim\mathcal N(0,1),
\]

where the current configuration uses

\[
\mu=0.7,\qquad s=1.0.
\]

This logit-normal distribution gives a broad range of global noise levels while favoring useful intermediate and cleaner states.

### 6.2 Token times below the ceiling

For each token, define

\[
\delta=\min(\bar t/2,\,0.6),
\]

then sample

\[
t_i=\bar t-\delta|\eta_i|,qquad \eta_i\sim\mathcal N(0,1).
\]

If a rare draw produces \(t_i<0\), it is replaced by a uniform sample in \([0,\bar t]\).

Consequently, editable tokens have heterogeneous times but no editable token is cleaner than the sampled ceiling \(\bar t\). This is the LTG principle used by PFT to prevent the network from routinely seeing unrealistically clean generated patches during training.

### 6.3 Garment-forcing samples

With probability 0.3 per training example, every editable token time is overridden to zero. Its editable state is therefore pure noise, forcing prediction to use the person context and DINO garment condition instead of reading partially clean paired-target pixels from the flow state. The remaining 70% of examples retain heterogeneous PFT token times.

### 6.4 Known context versus generated tokens

After sampling, token times are overridden according to the edit mask:

\[
t_i^{eff}=
\begin{cases}
t_i,&M_{T,i}^+=1,\\
1,&M_{T,i}^+=0.
\end{cases}
\]

Outside-mask tokens are intentionally fixed at \(t=1\) because they represent known agnostic context, not generated content. This does not violate the train-inference matching goal: inference also starts with these same context tokens clean and fixed.

## 7. Training latent construction

The evolving latent given to PFT is

\[
x_t=
M_L\odot\left[t^{eff}z^*+(1-t^{eff})\epsilon\right]
+(1-M_L)\odot z_a.
\]

This equation has two spatial regimes:

### Editable garment region

\[
x_t=t_i z^*+(1-t_i)\epsilon.
\]

Each garment token has its own time. Difficult structures such as logos, folds, boundaries, and hands crossing clothing can therefore be trained under different local noise levels.

### Preserved context region

\[
x_t=z_a,\qquad t_i^{eff}=1.
\]

The context is clean but garment-agnostic. It provides identity, pose boundaries, background, hair, and other preserved evidence without exposing the paired garment.

## 8. PFT-XL backbone

The native pretrained model operates on:

- 4 latent channels;
- a \(32\times32\) SD-VAE latent for a \(256\times256\) image;
- a \(2\times2\) latent patch size;
- a \(16\times16=256\)-token sequence;
- hidden dimension 1152;
- 28 transformer blocks;
- 16 attention heads;
- a four-channel velocity output plus one uncertainty channel.

Every block contains self-attention and an MLP controlled by per-token AdaLN modulation. Because the timestep embedding has shape \((B,N,D)\), a clean context token and a noisy garment token can coexist in the same self-attention sequence while receiving different normalization parameters.

## 9. Person conditioning

The original PFT patch embedder accepts four state channels. The VTON model expands it to nine:

\[
[x_t\;(4),\;z_a\;(4),\;m_{cond}\;(1)].
\]

The original four-channel projection weights are copied from PFT-XL. The five new input-channel weights are initialized to exactly zero.

At initialization:

\[
\operatorname{Embed}_{VTON}([x_t,z_a,m])
=\operatorname{Embed}_{PFT}(x_t).
\]

Thus adding person conditioning does not disturb the pretrained denoiser on the first optimization step. Training gradually learns how agnostic appearance and mask location should modify token representations.

The agnostic latent is supplied twice for different purposes:

- as fixed clean context outside the edit region in \(x_t\);
- as an explicit condition channel, allowing every editable query to access spatial person information through transformer self-attention.

Inside the hard editable region, the explicit agnostic latent is zero. Pose and identity cues therefore come from legitimate surrounding context rather than remnants of the old garment.

## 10. Garment conditioning

Garment images are encoded once, offline, with frozen DINOv2-small at 448 by 336. The resulting 32-by-24 FP16 feature map is cached on disk. Training loads only this map; DINO itself and its activations are never placed on the training GPU. A learned projection maps the 384-channel DINO features into the PFT hidden dimension. Garment background tokens are suppressed using the resized garment foreground mask \(M_g\).

Every fourth transformer block contains garment cross-attention:

\[
Q=W_Q h_{person},\qquad
K=W_K h_g,\qquad
V=W_V h_g,
\]

\[
CA(h,z_g)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}+B_{M_g}\right)V.
\]

The cross-attention residual is multiplied by the editable query mask, so garment features are injected primarily into tokens that are allowed to change.

Cross-attention is inserted after self-attention and before the MLP in blocks 4, 8, 12, 16, 20, 24, and 28.

The cross-attention output projection is initialized to zero, while the DINO projection has a small nonzero Xavier initialization. Therefore garment cross-attention leaves pretrained PFT unchanged initially, but its output projection receives gradients immediately.

The interpolated garment mask does not replace garment features. It only determines which DINO garment tokens are valid keys and values.

## 11. Class and garment-free conditioning

The ImageNet class embedding from PFT-XL is retained. When no VTON category label is supplied, the model uses the pretrained null-class index 1000.

During training, the garment latent and garment mask are independently dropped for approximately 10% of samples. This teaches a garment-unconditional branch.

At guided inference, the conditional velocity is

\[
v_{cfg}=v_{null}+w(v_{garment}-v_{null}),
\]

where \(w\) is `cfg_scale`. The agnostic person remains present in both branches; only the garment condition is removed from the null branch. This makes guidance strengthen garment identity without discarding person identity or pose context.

## 12. Training losses

### 12.1 Editable-region flow loss

The main loss is evaluated only in the expanded editable region:

\[
\mathcal L_{flow}
=\frac{\sum M_L\odot\|v_\theta(x_t,t,c)-u\|^2}
{C\sum M_L+\varepsilon}.
\]

This focuses capacity on the region the sampler will actually update.

### 12.2 Outside velocity regularization

Although outside cells are clamped and receive zero sampler step size, a small regularizer encourages their predicted velocity to be zero:

\[
\mathcal L_{outside}
=\operatorname{mean}_{1-M_L}\|v_\theta\|^2.
\]

Its default weight is 0.01. This discourages unstable irrelevant predictions without dominating the pretrained flow objective.

### 12.3 Uncertainty loss

PFT predicts one log-variance channel \(\ell_\theta\). It defines

\[
\sigma_\theta^2=\exp(\ell_\theta).
\]

The target velocity is scored under a diagonal Gaussian whose mean is the detached predicted velocity:

\[
\mathcal L_\sigma
=\operatorname{NLL}
\left(u;\operatorname{stopgrad}(v_\theta),\sigma_\theta^2\right).
\]

This loss is also restricted to the editable region and has default weight 0.01. Detaching the mean prevents the uncertainty objective from changing velocity merely to make variance prediction easier.

### 12.4 Total loss

\[
\mathcal L
=\mathcal L_{flow}
+0.01\mathcal L_{outside}
+0.01\mathcal L_\sigma.
\]

## 13. Standard inference

Inference begins with noise only in the editable region:

\[
x_0=M_L\odot\epsilon+(1-M_L)\odot z_a.
\]

Token times begin at

\[
t_i=
\begin{cases}
0,&M_{T,i}^+=1,\\
1,&M_{T,i}^+=0.
\end{cases}
\]

For uniform Euler sampling, every editable token advances by the same \(\Delta t\):

\[
x\leftarrow x+M_L\odot\Delta t\,v_\theta(x,t,c).
\]

After every update, the outside region is clamped:

\[
x\leftarrow M_L\odot x+(1-M_L)\odot z_a.
\]

Clamping makes preservation an explicit sampler invariant rather than a behavior the model must learn.

## 14. Uncertainty-adaptive inference

Adaptive sampling should be enabled only after the uncertainty head has been fine-tuned on VTON.

At each outer step:

1. Predict velocity and log variance.
2. Average log variance over each \(2\times2\) latent token.
3. Rank only editable tokens.
4. Select the highest-uncertainty fraction, normally 30%.
5. Advance easy tokens by the complete outer step.
6. Advance uncertain tokens through several smaller inner steps, recomputing their velocity after each step.

For `inner_steps = k`, difficult tokens use step size

\[
\Delta t_{inner}=\Delta t/k.
\]

All editable tokens reach the same outer endpoint, but uncertain tokens receive \(k\) local evaluations. Preserved tokens remain fixed at \(t=1\) and are excluded from uncertainty ranking.

This design tests the main VTON-specific Patch Forcing hypothesis: fine garment structures should receive more computation while easy cloth areas and known person context provide progressively cleaner evidence.

## 15. Why this training setup should work

### 15.1 It matches inference states

Training and inference both contain clean agnostic context outside the edit mask and noisy/generated content inside it. The network is not trained with clean paired garment pixels in locations that will contain old-garment context during inference.

### 15.2 It prevents the simplest shortcut

The old or paired garment is removed before VAE encoding, the edit envelope is dilated, and editable agnostic latent cells are zeroed again. The easiest path to low loss is therefore to use the supplied garment condition rather than copy the worn garment.

### 15.3 It retains the pretrained image prior

All new person channels and garment cross-attention outputs are zero-initialized. Before fine-tuning, the VTON model is numerically equal to the original PFT model for the same state, time, and class inputs. This avoids destroying the pretrained natural-image and patchwise-flow behavior at initialization.

### 15.4 Known regions provide useful context

Self-attention can use clean face, hair, body boundary, pose, and background tokens to resolve noisy garment tokens. This is precisely the setting where patchwise heterogeneous time conditioning is useful.

### 15.5 Conditions have distinct responsibilities

- \(x_t\) represents the current generated state.
- \(z_a\) represents person identity and spatial context without the old garment.
- the soft mask identifies the editable boundary.
- VAE garment tokens preserve local color, texture, boundaries, and fine appearance.
- DINO garment tokens provide complementary semantic structure.
- \(M_g\) removes irrelevant garment background keys.
- per-token time tells the model how reliable each spatial token currently is.

Keeping these roles separate makes it harder for the model to confuse a control signal with appearance content.

### 15.6 The objective matches sampler authority

The main loss is concentrated where the sampler is allowed to update. Outside preservation is guaranteed by clamping and RGB compositing rather than relying solely on learned reconstruction.

## 16. Low-memory fine-tuning

The 16 GB smoke configuration uses:

- batch size 1;
- four-step gradient accumulation;
- gradient checkpointing across transformer blocks;
- adapter, input projection, garment cross-attention, and final-layer training;
- frozen remaining PFT backbone parameters;
- no EMA model copy;
- no FID network;
- eight-step validation sampling.

This mode is intended to validate data flow, checkpoint transfer, gradients, loss reduction, and paired/unpaired visual outputs. It is not a substitute for the later full-data ablation between adapter-only and broader attention fine-tuning.

## 17. Important limitations

1. PFT-XL was pretrained on a square grid. The implementation supports 512-by-384 fine-tuning through positional interpolation, but this remains resolution transfer rather than native rectangular pretraining.
2. One token equals roughly 16 input pixels. This handles small parsing errors but not large category topology changes by itself.
3. The implementation has no DensePose or explicit pose encoder. It relies on agnostic spatial context and visible body boundaries.
4. Joint VAE and DINO conditioning combines photometric and semantic evidence, but exact readable text is still not guaranteed without a dedicated text/detail path or loss.
5. Adaptive uncertainty is meaningful only after the head has been trained on the VTON distribution.
6. Unpaired validation has no pixel-aligned ground truth. It should be judged qualitatively or with garment/person-specific metrics, not SSIM against the input person.

## 18. Required invariants and debugging checks

The implementation should maintain these invariants:

- With new condition weights at zero, VTON PFT output equals pretrained PFT output.
- Outside \(M_L\), training state equals the agnostic latent, never the full-person latent.
- Outside token time is exactly 1.
- Outside latent values do not change during either standard or adaptive sampling.
- Outside RGB pixels equal the source person after final composition.
- Garment cross-attention gradients reach its zero-initialized output projection.
- Person-condition gradients reach the five added input channels.
- Uncertainty ranking considers editable tokens only.

The tests in `tests/test_vton.py` cover the principal architectural and preservation invariants.

## 19. Code map

| Component | File |
|---|---|
| VTON PFT architecture and conditioning | `patch_flow/models/pf_transformer_vton.py` |
| Patchwise time construction and samplers | `patch_flow/flow_vton.py` |
| Mask conversion and RGB composition | `patch_flow/vton_utils.py` |
| Masked losses and VAE preprocessing | `patch_flow/trainer_vton.py` |
| Paired/unpaired VITON-HD data | `patch_flow/vton_data.py` |
| Full training configuration | `configs/experiment/viton-pft-xl.yaml` |
| 16 GB smoke configuration | `configs/experiment/viton-pft-xl-smoke16gb.yaml` |
| Single-example inference | `scripts/vton_sample.py` |

## 20. Minimal training sequence

1. Run `setup.sh` or provide compatible PFT-XL and SD-VAE checkpoints.
2. Generate or transfer the agnostic masks.
3. Run the 32-sample smoke split and smoke training.
4. Confirm that the paired result reconstructs the held-out garment and the unpaired result follows the new garment.
5. Confirm that background, face, and other outside-mask pixels remain exact after composition.
6. Train adapter-only on the full paired dataset.
7. Compare against broader self-attention/AdaLN fine-tuning.
8. Fine-tune and calibrate the uncertainty head.
9. Compare uniform and adaptive sampling at matched network-evaluation budgets.

## 21. Training steps and periodic validation previews

Training length, validation frequency, checkpoint frequency, and preview frequency are separate settings.

### Optimizer-step settings

The main settings are under `train_params`:

```yaml
train_params:
  max_steps: 50000
  accumulate_grad_batches: 4
  val_check_interval: 1000
  limit_val_batches: 1
  log_every_n_steps: 1
```

- `max_steps` is the number of optimizer updates before training stops.
- `accumulate_grad_batches` is the number of mini-batches used for one optimizer update.
- `val_check_interval` runs validation after this many optimizer updates.
- `limit_val_batches` controls how many validation batches are evaluated at each validation event.
- `log_every_n_steps` controls scalar loss logging.

For batch size 1 and `accumulate_grad_batches: 4`:

\[
50\text{ optimizer steps}\times4\text{ mini-batches}=200\text{ training samples seen},
\]

before accounting for repeated examples or multiple devices.

The approximate number of samples processed is

\[
N_{samples}=N_{steps}\times B_{device}\times N_{devices}\times N_{accumulation}.
\]

### Preview settings

VTON preview settings are under `trainer.params`:

```yaml
trainer:
  params:
    save_validation_previews: true
    preview_every_n_validations: 1
    sample_kwargs:
      num_steps: 8
      adaptive: false
```

- `save_validation_previews` enables PNG output.
- `preview_every_n_validations: 1` saves at every validation event; use 2 to save every second event.
- `sample_kwargs.num_steps` is the number of flow-sampling steps used to generate each preview. It does not change training length.
- Adaptive sampling should remain disabled for early smoke training.

Preview sheets are written to

```text
logs/<experiment>/<date>/<run>/previews/stepXXXXXX.png
logs/<experiment>/<date>/<run>/previews/latest.png
```

Each row is one validation example. Columns are ordered as target, source person, expanded agnostic person, effective edit mask, garment, and try-on output. In the smoke split, the first row is paired and the second row is unpaired.

### Smoke-test example

The supplied 16 GB configuration uses:

```yaml
train_params:
  max_steps: 1000
  accumulate_grad_batches: 4
  val_check_interval: 5
  limit_val_batches: 1

checkpoint_params:
  every_n_train_steps: 500
```

It therefore produces previews at optimizer steps 5, 10, 15, and so on through step 1000, and checkpoints at steps 500 and 1000.

The smoke dataset contains only 32 training pairs, so the training loop reshuffles and repeats the finite dataloader across epochs until `max_steps` is reached. With batch size 1 and gradient accumulation 4, one pass supplies 8 optimizer updates; 1000 updates therefore require multiple passes over the smoke split.

Settings can be changed without editing YAML:

```bash
python train.py experiment=viton-pft-xl-smoke16gb \
  train_params.max_steps=100 \
  train_params.val_check_interval=5 \
  checkpoint_params.every_n_train_steps=20 \
  trainer.params.preview_every_n_validations=1
```

For full training, a reasonable initial schedule is:

```bash
python train.py experiment=viton-pft-xl \
  train_params.max_steps=50000 \
  train_params.val_check_interval=1000 \
  checkpoint_params.every_n_train_steps=5000
```

Validation sampling is much more expensive than a training update. Very small `val_check_interval` values are useful for a smoke test but should be increased for full training.
