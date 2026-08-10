package routedlm

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"overgo/internal/safetensors"
)

// Checkpoint tensor names.
const (
	EmbedTokensName = "model.language_model.model.embed_tokens.weight"
	FinalNormName   = "model.language_model.model.norm.weight"
	LMHeadName      = "model.language_model.lm_head.weight"
	layerPrefixFmt  = "model.language_model.model.layers.%d."
)

// InputNormWeights: per-branch pre-attention RMSNorm scales.
type InputNormWeights struct {
	Text, Vision []float32
}

// QKVWeights: per-branch attention projections + shared per-head QK norms.
type QKVWeights struct {
	QText, KText, VText, OText         BF16Matrix
	QVision, KVision, VVision, OVision BF16Matrix
	QNorm, KNorm                       []float32
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

// LoadLayerWeights: one layer's dual-branch weight sets (matrices stay bf16).
func LoadLayerWeights(src *safetensors.Source, cfg Config, layer int) (LayerWeights, error) {
	if layer < 0 || layer >= cfg.NumHiddenLayers {
		return LayerWeights{}, fmt.Errorf("routed lm layer: layer=%d outside [0,%d)", layer, cfg.NumHiddenLayers)
	}
	prefix := fmt.Sprintf(layerPrefixFmt, layer)
	h, f := cfg.HiddenSize, cfg.IntermediateSize
	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim
	var w LayerWeights
	var err error
	mat := func(dst *BF16Matrix, name string, in, out int) {
		if err != nil {
			return
		}
		*dst, err = materializeBF16(src, prefix+name, in, out)
	}
	vec := func(dst *[]float32, name string, dim int) {
		if err != nil {
			return
		}
		*dst, err = materializeVectorF32(src, prefix+name, dim)
	}
	vec(&w.InputNorm.Text, "input_layernorm.weight", h)
	vec(&w.InputNorm.Vision, "input_layernorm_v.weight", h)
	mat(&w.QKV.QText, "self_attn.q_proj.weight", h, qOut)
	mat(&w.QKV.KText, "self_attn.k_proj.weight", h, kvOut)
	mat(&w.QKV.VText, "self_attn.v_proj.weight", h, kvOut)
	mat(&w.QKV.OText, "self_attn.o_proj.weight", qOut, h)
	mat(&w.QKV.QVision, "self_attn.q_proj_v.weight", h, qOut)
	mat(&w.QKV.KVision, "self_attn.k_proj_v.weight", h, kvOut)
	mat(&w.QKV.VVision, "self_attn.v_proj_v.weight", h, kvOut)
	mat(&w.QKV.OVision, "self_attn.o_proj_v.weight", qOut, h)
	vec(&w.QKV.QNorm, "self_attn.query_layernorm.weight", cfg.HeadDim)
	vec(&w.QKV.KNorm, "self_attn.key_layernorm.weight", cfg.HeadDim)
	vec(&w.Output.PostText, "post_attention_layernorm.weight", h)
	vec(&w.Output.PostVision, "post_attention_layernorm_v.weight", h)
	mat(&w.Output.GateText, "mlp.gate_proj.weight", h, f)
	mat(&w.Output.UpText, "mlp.up_proj.weight", h, f)
	mat(&w.Output.DownText, "mlp.down_proj.weight", f, h)
	mat(&w.Output.GateVision, "mlp_v.gate_proj.weight", h, f)
	mat(&w.Output.UpVision, "mlp_v.up_proj.weight", h, f)
	mat(&w.Output.DownVision, "mlp_v.down_proj.weight", f, h)
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
func EmbeddingRows(src *safetensors.Source, cfg Config, tokenIDs []int) ([]float32, error) {
	t, ok := src.Tensors[EmbedTokensName]
	if !ok {
		return nil, fmt.Errorf("routed lm embed: missing %s", EmbedTokensName)
	}
	if len(t.Shape) != 2 || int(t.Shape[0]) != cfg.VocabSize || int(t.Shape[1]) != cfg.HiddenSize {
		return nil, fmt.Errorf("routed lm embed shape %v, want [%d,%d]", t.Shape, cfg.VocabSize, cfg.HiddenSize)
	}
	return ReadTensorRowsF32(t, cfg.HiddenSize, tokenIDs)
}

// TerminalWeights: final norm (f32) + the head tensor handle (streamed).
type TerminalWeights struct {
	FinalNorm []float32
	Head      safetensors.Tensor
}

// LoadTerminalWeights: final norm + head (lm_head, else tied embeddings).
func LoadTerminalWeights(src *safetensors.Source, cfg Config) (TerminalWeights, error) {
	norm, err := materializeVectorF32(src, FinalNormName, cfg.HiddenSize)
	if err != nil {
		return TerminalWeights{}, err
	}
	head, ok := src.Tensors[LMHeadName]
	if !ok {
		head, ok = src.Tensors[EmbedTokensName]
	}
	if !ok {
		return TerminalWeights{}, fmt.Errorf("routed lm terminal: missing %s or %s", LMHeadName, EmbedTokensName)
	}
	if len(head.Shape) != 2 || int(head.Shape[0]) != cfg.VocabSize || int(head.Shape[1]) != cfg.HiddenSize {
		return TerminalWeights{}, fmt.Errorf("routed lm terminal head shape %v, want [%d,%d]", head.Shape, cfg.VocabSize, cfg.HiddenSize)
	}
	return TerminalWeights{FinalNorm: norm, Head: head}, nil
}
