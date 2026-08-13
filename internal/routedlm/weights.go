package routedlm

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"overgo/internal/safetensors"
)

// InputNormWeights: per-branch pre-attention RMSNorm scales.
type InputNormWeights struct {
	Text, Vision []float32
}

// QKVWeights: per-branch attention projections + per-branch per-head QK norm
// scales (binding sections concatenated to head_dim; a shared unforked
// section yields identical branch entries).
type QKVWeights struct {
	QText, KText, VText, OText         BF16Matrix
	QVision, KVision, VVision, OVision BF16Matrix
	QNorm, KNorm                       [2][]float32
}

// OutputWeights: per-branch post-attention norm + SiLU MLP.
type OutputWeights struct {
	PostText, PostVision             []float32
	GateText, UpText, DownText       BF16Matrix
	GateVision, UpVision, DownVision BF16Matrix
}

type LayerWeights struct {
	InputNorm InputNormWeights
	QKV       QKVWeights
	Output    OutputWeights
}

// BranchLayerWeights is one routed branch. Generation streams only the image
// branch; loading the unused text branch doubles checkpoint traffic.
type BranchLayerWeights struct {
	InputNorm, QNorm, KNorm, PostNorm []float32
	Q, K, V, O, Gate, Up, Down        BF16Matrix
}

// LoadBranchLayerWeights loads one layer branch through the shared binding.
func LoadBranchLayerWeights(src *safetensors.Source, cfg Config, b BranchBinding, layer, branch int) (BranchLayerWeights, error) {
	if err := b.validate(); err != nil {
		return BranchLayerWeights{}, err
	}
	if layer < 0 || layer >= cfg.NumHiddenLayers || branch < 0 || branch > 1 {
		return BranchLayerWeights{}, fmt.Errorf("routed lm branch layer: layer=%d branch=%d", layer, branch)
	}
	h, f := cfg.HiddenSize, cfg.IntermediateSize
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	var w BranchLayerWeights
	var err error
	mat := func(dst *BF16Matrix, suffix string, in, out int) {
		if err == nil {
			*dst, err = materializeBF16(src, b.LayerTensorName(layer, branch, suffix), in, out)
		}
	}
	vec := func(dst *[]float32, suffix string, dim int) {
		if err == nil {
			*dst, err = materializeVectorF32(src, b.LayerTensorName(layer, branch, suffix), dim)
		}
	}
	sections := func(dst *[]float32, suffixes []string) {
		if err != nil {
			return
		}
		names := make([]string, len(suffixes))
		for index, suffix := range suffixes {
			names[index] = b.LayerTensorName(layer, branch, suffix)
		}
		*dst, err = materializeNormSectionsF32(src, names, cfg.HeadDim)
	}
	vec(&w.InputNorm, "input_layernorm.weight", h)
	mat(&w.Q, "self_attn.q_proj.weight", h, qOut)
	mat(&w.K, "self_attn.k_proj.weight", h, kvOut)
	mat(&w.V, "self_attn.v_proj.weight", h, kvOut)
	mat(&w.O, "self_attn.o_proj.weight", qOut, h)
	sections(&w.QNorm, b.QNormSections)
	sections(&w.KNorm, b.KNormSections)
	vec(&w.PostNorm, "post_attention_layernorm.weight", h)
	mat(&w.Gate, "mlp.gate_proj.weight", h, f)
	mat(&w.Up, "mlp.up_proj.weight", h, f)
	mat(&w.Down, "mlp.down_proj.weight", f, h)
	if err != nil {
		return BranchLayerWeights{}, fmt.Errorf("routed lm branch layer %d/%d: %w", layer, branch, err)
	}
	return w, nil
}

func materializeBF16(src *safetensors.Source, name string, in, out int) (BF16Matrix, error) {
	t, ok := src.Tensors[name]
	if !ok {
		return BF16Matrix{}, fmt.Errorf("routed lm: missing tensor %s", name)
	}
	if t.DType != "BF16" {
		return BF16Matrix{}, fmt.Errorf("routed lm: tensor %s dtype %s, want BF16", name, t.DType)
	}
	if int(t.Elements()) != in*out {
		return BF16Matrix{}, fmt.Errorf("routed lm: tensor %s elements %d != %dx%d (shape %v)", name, t.Elements(), out, in, t.Shape)
	}
	raw := make([]byte, in*out*2)
	if _, err := t.ReadAt(raw, 0); err != nil {
		return BF16Matrix{}, fmt.Errorf("routed lm: %s read: %w", name, err)
	}
	data := make([]uint16, in*out)
	for i := range data {
		data[i] = binary.LittleEndian.Uint16(raw[i*2:])
	}
	return BF16Matrix{Data: data, In: in, Out: out}, nil
}

func materializeVectorF32(src *safetensors.Source, name string, dim int) ([]float32, error) {
	t, ok := src.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("routed lm: missing tensor %s", name)
	}
	if int(t.Elements()) != dim {
		return nil, fmt.Errorf("routed lm: tensor %s elements %d != %d", name, t.Elements(), dim)
	}
	reader, err := safetensors.F32Reader(t)
	if err != nil {
		return nil, fmt.Errorf("routed lm: %s: %w", name, err)
	}
	raw := make([]byte, dim*4)
	if _, err := io.ReadFull(reader, raw); err != nil {
		return nil, fmt.Errorf("routed lm: %s read: %w", name, err)
	}
	out := make([]float32, dim)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}

// materializeNormSectionsF32: 1-D norm sections concatenated in order;
// per-section lengths come from tensor shapes and must sum to total.
func materializeNormSectionsF32(src *safetensors.Source, names []string, total int) ([]float32, error) {
	out := make([]float32, 0, total)
	for _, name := range names {
		t, ok := src.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("routed lm: missing tensor %s", name)
		}
		if len(t.Shape) != 1 || t.Shape[0] <= 0 {
			return nil, fmt.Errorf("routed lm: norm section %s shape %v, want 1-D", name, t.Shape)
		}
		section, err := materializeVectorF32(src, name, int(t.Shape[0]))
		if err != nil {
			return nil, err
		}
		out = append(out, section...)
	}
	if len(out) != total {
		return nil, fmt.Errorf("routed lm: norm sections %v elements %d != %d", names, len(out), total)
	}
	return out, nil
}

// LoadLayerWeights: one layer's dual-branch weight sets (matrices stay bf16);
// names resolved through the branch binding.
func LoadLayerWeights(src *safetensors.Source, cfg Config, b BranchBinding, layer int) (LayerWeights, error) {
	if err := b.validate(); err != nil {
		return LayerWeights{}, err
	}
	if layer < 0 || layer >= cfg.NumHiddenLayers {
		return LayerWeights{}, fmt.Errorf("routed lm layer: layer=%d outside [0,%d)", layer, cfg.NumHiddenLayers)
	}
	h, f := cfg.HiddenSize, cfg.IntermediateSize
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	var w LayerWeights
	var err error
	mat := func(dst *BF16Matrix, branch int, suffix string, in, out int) {
		if err != nil {
			return
		}
		*dst, err = materializeBF16(src, b.LayerTensorName(layer, branch, suffix), in, out)
	}
	vec := func(dst *[]float32, branch int, suffix string, dim int) {
		if err != nil {
			return
		}
		*dst, err = materializeVectorF32(src, b.LayerTensorName(layer, branch, suffix), dim)
	}
	sections := func(dst *[]float32, branch int, suffixes []string) {
		if err != nil {
			return
		}
		names := make([]string, len(suffixes))
		for i, suffix := range suffixes {
			names[i] = b.LayerTensorName(layer, branch, suffix)
		}
		*dst, err = materializeNormSectionsF32(src, names, cfg.HeadDim)
	}
	vec(&w.InputNorm.Text, 0, "input_layernorm.weight", h)
	vec(&w.InputNorm.Vision, 1, "input_layernorm.weight", h)
	mat(&w.QKV.QText, 0, "self_attn.q_proj.weight", h, qOut)
	mat(&w.QKV.KText, 0, "self_attn.k_proj.weight", h, kvOut)
	mat(&w.QKV.VText, 0, "self_attn.v_proj.weight", h, kvOut)
	mat(&w.QKV.OText, 0, "self_attn.o_proj.weight", qOut, h)
	mat(&w.QKV.QVision, 1, "self_attn.q_proj.weight", h, qOut)
	mat(&w.QKV.KVision, 1, "self_attn.k_proj.weight", h, kvOut)
	mat(&w.QKV.VVision, 1, "self_attn.v_proj.weight", h, kvOut)
	mat(&w.QKV.OVision, 1, "self_attn.o_proj.weight", qOut, h)
	for branch := 0; branch < 2; branch++ {
		sections(&w.QKV.QNorm[branch], branch, b.QNormSections)
		sections(&w.QKV.KNorm[branch], branch, b.KNormSections)
	}
	vec(&w.Output.PostText, 0, "post_attention_layernorm.weight", h)
	vec(&w.Output.PostVision, 1, "post_attention_layernorm.weight", h)
	mat(&w.Output.GateText, 0, "mlp.gate_proj.weight", h, f)
	mat(&w.Output.UpText, 0, "mlp.up_proj.weight", h, f)
	mat(&w.Output.DownText, 0, "mlp.down_proj.weight", f, h)
	mat(&w.Output.GateVision, 1, "mlp.gate_proj.weight", h, f)
	mat(&w.Output.UpVision, 1, "mlp.up_proj.weight", h, f)
	mat(&w.Output.DownVision, 1, "mlp.down_proj.weight", f, h)
	if err != nil {
		return LayerWeights{}, fmt.Errorf("routed lm layer %d: %w", layer, err)
	}
	return w, nil
}

// ReadTensorRowsF32: selected rows of a 2D BF16/F32 tensor as f32.
func ReadTensorRowsF32(t safetensors.Tensor, rowWidth int, rows []int) ([]float32, error) {
	if len(t.Shape) != 2 || int(t.Shape[1]) != rowWidth {
		return nil, fmt.Errorf("routed lm rows: tensor %s shape %v, want [*,%d]", t.Name, t.Shape, rowWidth)
	}
	elemBytes := 0
	switch t.DType {
	case "BF16":
		elemBytes = 2
	case "F32":
		elemBytes = 4
	default:
		return nil, fmt.Errorf("routed lm rows: tensor %s dtype %s unsupported", t.Name, t.DType)
	}
	rowBytes := rowWidth * elemBytes
	out := make([]float32, len(rows)*rowWidth)
	raw := make([]byte, rowBytes)
	for i, row := range rows {
		if row < 0 || row >= int(t.Shape[0]) {
			return nil, fmt.Errorf("routed lm rows: row %d outside [0,%d)", row, t.Shape[0])
		}
		if _, err := t.ReadAt(raw, int64(row)*int64(rowBytes)); err != nil {
			return nil, fmt.Errorf("routed lm rows: %s row %d: %w", t.Name, row, err)
		}
		dst := out[i*rowWidth : (i+1)*rowWidth]
		if elemBytes == 2 {
			for c := range dst {
				dst[c] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[c*2:])) << 16)
			}
		} else {
			for c := range dst {
				dst[c] = math.Float32frombits(binary.LittleEndian.Uint32(raw[c*4:]))
			}
		}
	}
	return out, nil
}

// EmbeddingRows: token embedding rows by id.
func EmbeddingRows(src *safetensors.Source, cfg Config, b BranchBinding, tokenIDs []int) ([]float32, error) {
	t, ok := src.Tensors[b.EmbedName]
	if !ok {
		return nil, fmt.Errorf("routed lm embed: missing %s", b.EmbedName)
	}
	if len(t.Shape) != 2 || int(t.Shape[0]) != cfg.VocabSize || int(t.Shape[1]) != cfg.HiddenSize {
		return nil, fmt.Errorf("routed lm embed shape %v, want [%d,%d]", t.Shape, cfg.VocabSize, cfg.HiddenSize)
	}
	return ReadTensorRowsF32(t, cfg.HiddenSize, tokenIDs)
}

// TerminalWeights: per-branch final norms (f32; equal binding names load the
// same scales) + the head tensor handle (streamed).
type TerminalWeights struct {
	FinalNorm [2][]float32
	Head      safetensors.Tensor
}

// LoadTerminalWeights: per-branch final norms + head (lm_head, else tied
// embeddings).
func LoadTerminalWeights(src *safetensors.Source, cfg Config, b BranchBinding) (TerminalWeights, error) {
	if err := b.validate(); err != nil {
		return TerminalWeights{}, err
	}
	var out TerminalWeights
	for branch, name := range b.FinalNormName {
		if branch > 0 && name == b.FinalNormName[0] {
			out.FinalNorm[branch] = out.FinalNorm[0]
			continue
		}
		norm, err := materializeVectorF32(src, name, cfg.HiddenSize)
		if err != nil {
			return TerminalWeights{}, err
		}
		out.FinalNorm[branch] = norm
	}
	head, ok := src.Tensors[b.LMHeadName]
	if !ok {
		head, ok = src.Tensors[b.EmbedName]
	}
	if !ok {
		return TerminalWeights{}, fmt.Errorf("routed lm terminal: missing %s or %s", b.LMHeadName, b.EmbedName)
	}
	if len(head.Shape) != 2 || int(head.Shape[0]) != cfg.VocabSize || int(head.Shape[1]) != cfg.HiddenSize {
		return TerminalWeights{}, fmt.Errorf("routed lm terminal head shape %v, want [%d,%d]", head.Shape, cfg.VocabSize, cfg.HiddenSize)
	}
	out.Head = head
	return out, nil
}
