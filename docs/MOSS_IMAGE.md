# MOSS-Image-SFT

Native text-to-image generation and adapter training for [MOSS-Image-SFT](https://www.modelscope.cn/openmoss/MOSS-Image-SFT.git) at commit `797fa8e678b4f0a766271bf5103e6cd7a2faa999`. The figures here come from retained records in [`docs/verification`](verification), which `internal/evaluation` `TestMOSSRetainedEvidence` holds to what was measured.

## Generating pictures

Register the model directory and choose it for image generation. A request takes:

| Control | Default | Notes |
| --- | --- | --- |
| prompt | required | Sent as typed, up to 4,096 tokens |
| prompt mode | text | `layout` validates a JSON layout before any work |
| width, height | 1024 | Positive multiples of 16 |
| steps | 50 | The schedule may drop steps that land on one training grid point |
| guidance scale | 3 | 1 removes the negative branch |
| negative prompt | empty | |
| seed | 42 | The browser refuses a seed it cannot carry exactly |

A control outside this list is refused, never ignored.

**Pictures match upstream.** Four whole generations at 50 steps and guidance 3 were compared with the pinned upstream pipeline at the conditioning, every step and the final pixels: English, Chinese and layout prompts, square, landscape and portrait, a negative prompt and a second seed. All 1,664 compared boundaries pass. The worst difference in any compared value is 3.9e-6, and each case's repeat gave the same PNG. See [`moss-image-parity.json`](verification/moss-image-parity.json).

**Quality.** On TIFA v1.0, 64 held-out captions drawn at 1024 square with the default request, a vision-language model answered 415 questions about the pictures: 0.911 ±0.014, judged by Qwen3.8-27B. This is the scorecard's held-out reading for image generation, and it measures whether a picture shows what its caption says. It is a reading of the native path: the parity above compares four generations with upstream, and upstream was not run on these questions.

**Cost.** On one NVIDIA GeForce RTX 4090 D, on one protocol, with the OS file cache warm for both ([`moss-image-resources.json`](verification/moss-image-resources.json)):

| Picture | Native | Upstream |
| --- | --- | --- |
| 1024 square, first | 101.5 s, 13.0 GiB | 52.5 s, 27.6 GiB |
| 1024 square, new seed | 95.1 s, 13.0 GiB | 52.4 s, 27.6 GiB |
| 1024 square, new prompt | 101.0 s, 13.0 GiB | 52.4 s, 27.6 GiB |
| 1024 by 768 | 72.2 s, 13.0 GiB | 37.1 s, 27.1 GiB |
| 2048 square | 552.8 s, 17.9 GiB | not measured |

Native takes about 1.9 times upstream's time and about 0.47 times its peak device memory. Nothing here is a claim about a smaller card.

Since then, each BF16 rounding of an elementwise result folds into the kernel that computes it, and the first picture's denoising takes 76.4 s against 94.2 s, the same picture bit for bit. A step then takes 1.449 s against upstream's 1.036 s, and its kernels account for its time: the BF16 products, the elementwise results stored as rounded F32 values where upstream stores BF16, the attention, and the conversions and copies around the products ([`moss-denoiser-wall.json`](verification/moss-denoiser-wall.json)). The rest of the gap is tracked as `time-and-memory/moss-situ`.

## Training adapters

```bash
go run ./cmd/train -route moss -model <model directory> -pictures <manifest.tsv> -size 256 -rank 16 -alpha 16 -steps 400 -out <checkpoint> -adapter <adapter directory> -served <model directory>
```

The manifest holds one picture, a tab and its caption per line, relative to the manifest. A run prepares each picture and caption once, trains low-rank adapters on the six projections of each of the transformer's joint blocks, and refuses to publish if its loss on a fixed path does not descend. `-adapter` writes a PEFT adapter. `-served` writes a model directory with the adapter merged, sharing the base's other components, which serves through the same image capability as the base under its own identity. `-heldout` with `-report` judges pictures the run never trains on, on fixed paths, before and after it.

A real run is recorded in [`moss-image-training-run.json`](verification/moss-image-training-run.json): rank 16 for 400 steps on 288 captioned VidGen frames at 256 square. Held-out flow-matching loss on 32 frames of other videos fell from 0.7405 to 0.7326, about one percent, better on all 32. It peaked at 19.2 GiB and took about 1.8 s a step. One rank-16 step on the real checkpoint took 3.9 s at 512 square and 14.7 s at 1024 square, at 26.1 GiB.

The trained pictures keep the base's colour and contrast and read somewhat more like the video frames they trained on. That is one person's look at six pairs: nothing here shows they are better. A falling loss proved nothing by itself: an early run, fed wrongly scaled pictures, showed a far larger fall and worse pictures.

## Not supported

- Image editing, image-to-image and video.
- Positive-branch skip-layer guidance: the negative branch's fixed skip of layer 22 is part of the pipeline, and the optional control is refused.
- Automatic prompt enhancement.
- Training the text encoder, the VAE or the timestep path.
- Loading an adapter at serving time: a trained model is served merged.
- The MOSS-Image Base checkpoint.

## Sources

The numerical oracle is `pipeline_moss_image.py` at the commit above, SHA-256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`, run under torch 2.12.0+cu126, transformers 5.14.1, diffusers 0.39.0 and accelerate 1.14.0, under its Apache-2.0 licence. Code packaging is [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image/tree/c1355e6b1986dfe8fba3c0455783a05899212342). Parity targets the pinned released pipeline: BF16 text encoder and denoiser, F32 noise, guidance, Euler and VAE, with shift 5. Deterministic import means the same pinned bytes and policy give the same content identities, tensor mappings and recipe, not GPU bit identity across hardware or library versions.
