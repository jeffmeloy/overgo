// Package latentvideo: text-to-video generation stages, neutral and
// config-driven (no model family named in code). This file owns semantic
// text conditioning: unigram tokenizer (HF tokenizer.json) -> streamed
// relative-position encoder -> Linear/GELU/Linear projection -> fixed-length
// context. Ported behavior (adaptive AcquireSemanticTextConditioning host
// path), written against overgo's shared primitives.
package latentvideo

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/processmeasure"
	"overgo/internal/representation"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tokenizer"
)

// TextConditioningSpec: artifact locations plus the two policy facts the
// checkpoint cannot carry (relative-attention max distance and norm epsilon
// come from the encoder's published config).
type TextConditioningSpec struct {
	TokenizerDir        string // HF unigram tokenizer.json + special_tokens_map.json
	EncoderCheckpoint   string // PyTorch ZIP checkpoint (.pth)
	ProjectionDir       string // safetensors dir carrying text_embedding.{0,2}
	SequenceLength      int    // fixed context rows (text_len)
	RelativeMaxDistance int
	NormEps             float64
	// Common is a prompt every request of a model carries, and Kept the store
	// its encoder rows are kept in from one process to the next, under
	// KeptUnder, the identity of the code that computes them. With no store
	// or no identity the rows are kept by the process alone, as any prompt's.
	Common    string
	Kept      artifact.Repository
	KeptUnder string
}

// textConditioningResult: the projected SequenceLength x Dim context.
type textConditioningResult struct {
	Context               []float32
	Rows, Dim             int
	TokenCount            int
	Encoder               encoderStats
	ProjectionSourceBytes int64
}

// projectionTensorNames: the Linear-GELU-Linear projection contract.
var projectionTensorNames = [4]string{
	"text_embedding.0.weight", "text_embedding.0.bias", "text_embedding.2.weight", "text_embedding.2.bias",
}

type projectionWeights struct {
	Linear0W, Linear0B []float32
	Linear2W, Linear2B []float32
}

// TextConditioning tokenizes prompt, streams the encoder, projects, and pads
// to SequenceLength rows (pad rows repeat the first inactive projected row,
// the reference's padding convention).
func TextConditioning(ctx context.Context, spec TextConditioningSpec, prompt string) (textConditioningResult, error) {
	if err := ctx.Err(); err != nil {
		return textConditioningResult{}, err
	}
	weights, sourceBytes, err := loadProjectionWeights(spec.ProjectionDir)
	if err != nil {
		return textConditioningResult{}, err
	}
	return textConditioningWithWeights(ctx, spec, prompt, weights, sourceBytes)
}

func textConditioningWithWeights(ctx context.Context, spec TextConditioningSpec, prompt string, weights projectionWeights, sourceBytes int64) (textConditioningResult, error) {
	conditioned, err := textConditioningsWithWeights(ctx, spec, weights, sourceBytes, prompt)
	if err != nil {
		return textConditioningResult{}, err
	}
	return conditioned[0], nil
}

// textConditioningsWithWeights conditions several prompts of one request
// together: the encoder's rows of each (encodePrompts), the ones not kept
// from an earlier request encoded in one pass over the checkpoint, each
// projected and padded. Encoder is the measures of that pass, zero for a
// request whose prompts were all kept.
func textConditioningsWithWeights(ctx context.Context, spec TextConditioningSpec, weights projectionWeights, sourceBytes int64, prompts ...string) ([]textConditioningResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	defer processmeasure.Stage(ctx, "prompt tokenized and projected")()
	encoded, textDim, encoderStats, err := encodePrompts(ctx, spec, prompts...)
	if err != nil {
		return nil, err
	}
	projection, err := representation.CompileTwoLayerProjectionF32(textDim, weights.Linear0W, weights.Linear0B, weights.Linear2W, weights.Linear2B)
	if err != nil {
		return nil, fmt.Errorf("text conditioning: %w", err)
	}
	dim := projection.OutputWidth
	conditioned := make([]textConditioningResult, len(prompts))
	for index, rows := range encoded {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		tokenCount := len(rows) / textDim
		context, err := projectCompactTextConditioning(rows, tokenCount, spec.SequenceLength, textDim, dim, weights, true)
		if err != nil {
			return nil, err
		}
		conditioned[index] = textConditioningResult{
			Context: context, Rows: spec.SequenceLength, Dim: dim,
			TokenCount: tokenCount, Encoder: encoderStats, ProjectionSourceBytes: sourceBytes,
		}
	}
	return conditioned, ctx.Err()
}

func loadProjectionWeights(dir string) (projectionWeights, int64, error) {
	var w projectionWeights
	source, err := safetensors.OpenSource(dir)
	if err != nil {
		return w, 0, err
	}
	defer source.Close()
	var sourceBytes int64
	values := [4][]float32{}
	for index, name := range projectionTensorNames {
		tensor, ok := source.Tensors[name]
		if !ok {
			return w, 0, fmt.Errorf("text conditioning projection: missing tensor %s", name)
		}
		decoded, err := safetensors.ReadF32(tensor)
		if err != nil {
			return w, 0, fmt.Errorf("text conditioning projection %s: %w", name, err)
		}
		values[index] = decoded
		sourceBytes += tensor.Size()
	}
	w.Linear0W, w.Linear0B, w.Linear2W, w.Linear2B = values[0], values[1], values[2], values[3]
	return w, sourceBytes, nil
}

// projectCompactTextConditioning: Linear(dim<-textDim)+GELU+Linear(dim<-dim)
// over the active rows plus ONE extra pad-source row (the projection of a
// zero encoder row), then pad-broadcast to textLen rows. bf16 rounds after
// each op, mirroring the reference.
func projectCompactTextConditioning(context []float32, contextTokens, textLen, textDim, dim int, w projectionWeights, bf16 bool) ([]float32, error) {
	projection, err := representation.CompileTwoLayerProjectionF32(textDim, w.Linear0W, w.Linear0B, w.Linear2W, w.Linear2B)
	if err != nil || !checked.Equal(projection.OutputWidth, dim) {
		return nil, fmt.Errorf("compact text conditioning: invalid projection weights: %w", err)
	}
	return representation.ProjectPaddedF32(context, contextTokens, textLen, projection, bf16)
}

// fixedUnigramTokenizer: fixed-length encode policy (truncate to length-1,
// append EOS, pad; mask marks active rows) over the shared unigram table.
type fixedUnigramTokenizer struct {
	table  *tokenizer.Unigram
	length int
	padID  int
	endID  int
}

func loadFixedUnigramTokenizer(dir string, length int) (*fixedUnigramTokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, tokenizerRoles))
	if err != nil {
		return nil, err
	}
	var roles struct {
		Pad string `json:"pad_token"`
		End string `json:"eos_token"`
		UNK string `json:"unk_token"`
	}
	if err := json.Unmarshal(raw, &roles); err != nil {
		return nil, err
	}
	if !checked.PositiveInts(length) || !checked.NonzeroAll(roles.Pad, roles.End, roles.UNK) {
		return nil, fmt.Errorf("fixed unigram tokenizer: incomplete sequence policy")
	}
	table, artifactUnkID, byteFallback, err := tokenizer.LoadHFUnigramJSON(filepath.Join(dir, tokenizerTable))
	if err != nil {
		return nil, err
	}
	if byteFallback {
		return nil, fmt.Errorf("fixed unigram tokenizer: artifact enables byte fallback outside its sequence policy")
	}
	roleID := func(role, literal string) (int, error) {
		id, ok := table.PieceID(literal)
		if !ok {
			return 0, fmt.Errorf("fixed unigram tokenizer: %s token %q is absent", role, literal)
		}
		return id, nil
	}
	padID, err := roleID("pad", roles.Pad)
	if err != nil {
		return nil, err
	}
	endID, err := roleID("end", roles.End)
	if err != nil {
		return nil, err
	}
	unkID, err := roleID("unknown", roles.UNK)
	if err != nil {
		return nil, err
	}
	if !checked.Equal(unkID, artifactUnkID) {
		return nil, fmt.Errorf("fixed unigram tokenizer: unknown token id=%d, artifact unk_id=%d", unkID, artifactUnkID)
	}
	return &fixedUnigramTokenizer{table: table, length: length, padID: padID, endID: endID}, nil
}

func (t *fixedUnigramTokenizer) EncodeWithMask(text string) ([]int, []int, error) {
	if t == nil || t.table == nil {
		return nil, nil, fmt.Errorf("fixed unigram tokenizer is nil")
	}
	text = tokenizer.NormalizeEscapedWidthWhitespace(text)
	var pieces []int
	var err error
	if checked.Nonzero(text) {
		if pieces, err = t.table.Encode(text); err != nil {
			return nil, nil, err
		}
	}
	pieces = pieces[:min(len(pieces), t.length-tensor.SingletonExtent)]
	ids, mask := make([]int, t.length), make([]int, t.length)
	n := copy(ids, pieces)
	for i := range n {
		mask[i] = tensor.SingletonExtent
	}
	ids[n], mask[n] = t.endID, tensor.SingletonExtent
	for i := n + tensor.SingletonExtent; i < len(ids); i++ {
		ids[i] = t.padID
	}
	return ids, mask, nil
}
