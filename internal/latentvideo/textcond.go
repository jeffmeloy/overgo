// Package latentvideo: text-to-video generation stages, neutral and
// config-driven (no model family named in code). This file owns semantic
// text conditioning: unigram tokenizer (HF tokenizer.json) -> streamed
// relative-position encoder -> Linear/GELU/Linear projection -> fixed-length
// context. Ported behavior (adaptive AcquireSemanticTextConditioning host
// path), written against overgo's shared primitives.
package latentvideo

import (
	"encoding/json"
	"fmt"
	"html"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/hostmath"
	"overgo/internal/pytorchzip"
	"overgo/internal/safetensors"
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
}

// TextConditioningResult: the projected SequenceLength x Dim context.
type TextConditioningResult struct {
	Context               []float32
	Rows, Dim             int
	TokenCount            int
	Encoder               EncoderStats
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
func TextConditioning(spec TextConditioningSpec, prompt string) (TextConditioningResult, error) {
	var out TextConditioningResult
	if spec.SequenceLength <= 0 {
		return out, fmt.Errorf("text conditioning: positive sequence length required")
	}
	tok, err := loadFixedUnigramTokenizer(spec.TokenizerDir, spec.SequenceLength)
	if err != nil {
		return out, err
	}
	metas, err := pytorchzip.ReadTensorMetadata(spec.EncoderCheckpoint)
	if err != nil {
		return out, err
	}
	plan, err := CompileEncoderPlan(metas, EncoderConfig{RelativeMaxDistance: spec.RelativeMaxDistance, NormEps: spec.NormEps})
	if err != nil {
		return out, err
	}
	if !plan.OK {
		return out, fmt.Errorf("text conditioning encoder contract: missing=%v unexpected=%v dtype=%v shape=%v", plan.Missing, plan.Unexpected, plan.DTypeMismatches, plan.ShapeMismatches)
	}
	weights, sourceBytes, err := loadProjectionWeights(spec.ProjectionDir)
	if err != nil {
		return out, err
	}
	textDim := plan.Config.Dim
	if len(weights.Linear0W)%textDim != 0 {
		return out, fmt.Errorf("text conditioning: projection input %d incompatible with encoder width %d", len(weights.Linear0W), textDim)
	}
	dim := len(weights.Linear0W) / textDim
	if len(weights.Linear0B) != dim || len(weights.Linear2W) != dim*dim || len(weights.Linear2B) != dim {
		return out, fmt.Errorf("text conditioning: inconsistent projection shapes")
	}
	ids, mask, err := tok.EncodeWithMask(prompt)
	if err != nil {
		return out, err
	}
	tokenCount := 0
	for _, v := range mask {
		if v != 0 {
			tokenCount++
		}
	}
	if tokenCount <= 0 || tokenCount > len(ids) {
		return out, fmt.Errorf("text conditioning: bad token count %d", tokenCount)
	}
	encoded, encoderStats, err := EncodeTokensStreamed(spec.EncoderCheckpoint, plan, ids, mask)
	if err != nil {
		return out, err
	}
	encoded = encoded[:tokenCount*textDim]
	context, err := projectCompactTextConditioning(encoded, tokenCount, spec.SequenceLength, textDim, dim, weights, true)
	if err != nil {
		return out, err
	}
	return TextConditioningResult{
		Context: context, Rows: spec.SequenceLength, Dim: dim,
		TokenCount: tokenCount, Encoder: encoderStats, ProjectionSourceBytes: sourceBytes,
	}, nil
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
		reader, err := safetensors.F32Reader(tensor)
		if err != nil {
			return w, 0, fmt.Errorf("text conditioning projection %s: %w", name, err)
		}
		elements := tensor.Elements()
		raw := make([]byte, elements*4)
		if _, err := io.ReadFull(reader, raw); err != nil {
			return w, 0, fmt.Errorf("text conditioning projection %s payload: %w", name, err)
		}
		decoded := make([]float32, elements)
		for i := range decoded {
			bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
			decoded[i] = math.Float32frombits(bits)
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
	if contextTokens < 0 || contextTokens > textLen {
		return nil, fmt.Errorf("compact text conditioning: context tokens=%d outside [0,%d]", contextTokens, textLen)
	}
	if len(context) != contextTokens*textDim {
		return nil, fmt.Errorf("compact text conditioning: context len=%d, want %d", len(context), contextTokens*textDim)
	}
	if len(w.Linear0W) != dim*textDim || len(w.Linear0B) != dim || len(w.Linear2W) != dim*dim || len(w.Linear2B) != dim {
		return nil, fmt.Errorf("compact text conditioning: invalid projection weights")
	}
	activeRows := contextTokens
	if activeRows < textLen {
		activeRows++
	}
	hidden := make([]float32, activeRows*dim)
	hostmath.ParallelRangeF64(activeRows, textDim*dim, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			hrow := hidden[r*dim : (r+1)*dim]
			for o := 0; o < dim; o++ {
				acc := float64(w.Linear0B[o])
				if r < contextTokens {
					xrow := context[r*textDim : (r+1)*textDim]
					wrow := w.Linear0W[o*textDim : (o+1)*textDim]
					for i := 0; i < textDim; i++ {
						acc += float64(xrow[i]) * float64(wrow[i])
					}
				}
				hrow[o] = float32(acc)
			}
		}
	})
	roundBF16If(bf16, hidden)
	hostmath.GELUTanhInPlace(hidden)
	roundBF16If(bf16, hidden)
	projected := make([]float32, activeRows*dim)
	hostmath.ParallelRangeF64(activeRows, dim*dim, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			xr, orow := hidden[r*dim:(r+1)*dim], projected[r*dim:(r+1)*dim]
			for o := 0; o < dim; o++ {
				wr := w.Linear2W[o*dim : (o+1)*dim]
				var acc float64
				for i, value := range xr {
					acc += float64(value) * float64(wr[i])
				}
				acc += float64(w.Linear2B[o])
				orow[o] = float32(acc)
			}
		}
	})
	roundBF16If(bf16, projected)
	out := make([]float32, textLen*dim)
	copy(out, projected[:contextTokens*dim])
	if contextTokens < textLen {
		pad := projected[contextTokens*dim : (contextTokens+1)*dim]
		for r := contextTokens; r < textLen; r++ {
			copy(out[r*dim:(r+1)*dim], pad)
		}
	}
	return out, nil
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
	raw, err := os.ReadFile(filepath.Join(dir, "special_tokens_map.json"))
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
	if length <= 0 || roles.Pad == "" || roles.End == "" || roles.UNK == "" {
		return nil, fmt.Errorf("fixed unigram tokenizer: incomplete sequence policy")
	}
	table, artifactUnkID, byteFallback, err := tokenizer.LoadHFUnigramJSON(filepath.Join(dir, "tokenizer.json"))
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
	if unkID != artifactUnkID {
		return nil, fmt.Errorf("fixed unigram tokenizer: unknown token id=%d, artifact unk_id=%d", unkID, artifactUnkID)
	}
	return &fixedUnigramTokenizer{table: table, length: length, padID: padID, endID: endID}, nil
}

func (t *fixedUnigramTokenizer) EncodeWithMask(text string) ([]int, []int, error) {
	if t == nil || t.table == nil {
		return nil, nil, fmt.Errorf("fixed unigram tokenizer is nil")
	}
	text = normalizeEscapedWidthWhitespace(text)
	var pieces []int
	var err error
	if text != "" {
		if pieces, err = t.table.Encode(text); err != nil {
			return nil, nil, err
		}
	}
	pieces = pieces[:min(len(pieces), t.length-1)]
	ids, mask := make([]int, t.length), make([]int, t.length)
	n := copy(ids, pieces)
	for i := 0; i < n; i++ {
		mask[i] = 1
	}
	ids[n], mask[n] = t.endID, 1
	for i := n + 1; i < len(ids); i++ {
		ids[i] = t.padID
	}
	return ids, mask, nil
}

// normalizeEscapedWidthWhitespace: HTML-unescape twice, fold fullwidth ASCII
// and ideographic space, collapse whitespace runs (reference normalization).
func normalizeEscapedWidthWhitespace(text string) string {
	text = html.UnescapeString(html.UnescapeString(text))
	text = strings.Map(func(r rune) rune {
		switch {
		case r >= 0xFF01 && r <= 0xFF5E:
			return r - 0xFEE0
		case r == 0x3000:
			return ' '
		default:
			return r
		}
	}, text)
	return strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
}
