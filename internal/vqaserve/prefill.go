package vqaserve

import (
	"context"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/patchtower"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

// PrefillContext is the shared prompt state of the device prefill and the
// full pipeline. The serving path builds it from processor-derived inputs
// (NewPrefillContext); the parity harness builds it from its goldens, which
// is why the fields are open.
type PrefillContext struct {
	Cfg                routedlm.Config
	Src                *safetensors.Source
	Spec               patchtower.Spec
	BlockLast          []float32
	Merger             patchtower.MergerWeights
	GridT              int
	GridH              int
	GridW              int
	ImageRows          int
	InputIDs           []int
	ImageMaskPositions []int
	Mask               []int
	PromptLen          int
	Prefill            []float32 // host prefill embeds [tokens, H]
	Segments           [][2]int
	Blocks             []routedlm.PrefillAttnBlock
	MaskText           []float32 // [tokens] 1 at text rows
	MaskVis            []float32 // [tokens] 1 at vision rows
}

// NewPrefillContext builds the prompt state from processor-derived inputs
// (input ids, image-mask positions, image grid): the golden-free serving
// path. The image-feature block_last and the host prefill embeds are
// produced on-device by the pipeline, so they stay nil here; imageRows
// derives from the grid and is cross-checked against the image-token count.
func NewPrefillContext(modelDir string, inputIDs, imageMaskPositions []int, gridT, gridH, gridW int) (*PrefillContext, error) {
	spec, err := patchtower.LoadSpec(modelDir)
	if err != nil {
		return nil, err
	}
	cfg, err := routedlm.LoadConfig(modelDir, Binding)
	if err != nil {
		return nil, err
	}
	src, err := safetensors.OpenSource(modelDir)
	if err != nil {
		return nil, err
	}
	merger, err := patchtower.LoadMergerWeights(src, spec)
	if err != nil {
		src.Close()
		return nil, err
	}
	if gridH%spec.MergeSize != 0 || gridW%spec.MergeSize != 0 {
		src.Close()
		return nil, fmt.Errorf("prefill context: grid [%d,%d] not divisible by merge=%d", gridH, gridW, spec.MergeSize)
	}
	imageRows := gridT * (gridH / spec.MergeSize) * (gridW / spec.MergeSize)
	if imageRows != len(imageMaskPositions) {
		src.Close()
		return nil, fmt.Errorf("prefill context: imageRows=%d != image-token count=%d", imageRows, len(imageMaskPositions))
	}
	promptLen := len(inputIDs)
	mask := routedlm.ModalityMask(promptLen, imageMaskPositions)
	context := &PrefillContext{
		Cfg: cfg, Src: src, Spec: spec, Merger: merger,
		GridT: gridT, GridH: gridH, GridW: gridW, ImageRows: imageRows,
		InputIDs: inputIDs, ImageMaskPositions: imageMaskPositions,
		Mask: mask, PromptLen: promptLen,
	}
	context.BindMask()
	return context, nil
}

// BindMask derives the visual segments, the prefill attention blocks and the
// text and vision row masks from Mask and PromptLen, for a context built
// either way.
func (pc *PrefillContext) BindMask() {
	pc.Segments = routedlm.VisualSegments(pc.Mask)
	pc.Blocks = routedlm.PrefillAttnBlocks(pc.Segments, pc.PromptLen)
	pc.MaskText = make([]float32, pc.PromptLen)
	pc.MaskVis = make([]float32, pc.PromptLen)
	for i, m := range pc.Mask {
		if m == 0 {
			pc.MaskText[i] = 1
		} else {
			pc.MaskVis[i] = 1
		}
	}
}

// BindPrefillBranch binds each weight of one branch of a layer to its input of the prefill graph by BindFeed.
func BindPrefillBranch(ctx context.Context, allocations *device.AllocationSet, inputs *executor.DeviceInputs, nodes routedlm.DevicePrefillBranch,
	inputNorm []float32, q, k, v, o routedlm.BF16Matrix, qNorm, kNorm, postNorm []float32, gate, up, down routedlm.BF16Matrix) error {
	f32 := func(node *tensor.Tensor, val []float32) error {
		return BindFeed(ctx, allocations, inputs, node, driver.Bytes(val))
	}
	bf16 := func(node *tensor.Tensor, m routedlm.BF16Matrix) error {
		return BindFeed(ctx, allocations, inputs, node, driver.Bytes(m.Data))
	}
	return FirstError(
		f32(nodes.InputNorm, inputNorm),
		bf16(nodes.Q, q), bf16(nodes.K, k), bf16(nodes.V, v), bf16(nodes.O, o),
		f32(nodes.QNorm, qNorm), f32(nodes.KNorm, kNorm), f32(nodes.PostNorm, postNorm),
		bf16(nodes.Gate, gate), bf16(nodes.Up, up), bf16(nodes.Down, down),
	)
}

// BindLayerBranches binds both branches of one layer to the prefill graph's inputs by BindPrefillBranch.
func BindLayerBranches(ctx context.Context, allocations *device.AllocationSet, inputs *executor.DeviceInputs, text, vision routedlm.DevicePrefillBranch, w routedlm.LayerWeights) error {
	if err := BindPrefillBranch(ctx, allocations, inputs, text,
		w.InputNorm.Text, w.QKV.QText, w.QKV.KText, w.QKV.VText, w.QKV.OText,
		w.QKV.QNorm[0], w.QKV.KNorm[0], w.Output.PostText, w.Output.GateText, w.Output.UpText, w.Output.DownText); err != nil {
		return err
	}
	return BindPrefillBranch(ctx, allocations, inputs, vision,
		w.InputNorm.Vision, w.QKV.QVision, w.QKV.KVision, w.QKV.VVision, w.QKV.OVision,
		w.QKV.QNorm[1], w.QKV.KNorm[1], w.Output.PostVision, w.Output.GateVision, w.Output.UpVision, w.Output.DownVision)
}
