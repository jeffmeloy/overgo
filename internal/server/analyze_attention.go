package server

import (
	"context"
	"errors"
	"math"
	"net/http"

	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

// AttentionCaptureAPI: one cacheless forward that records, for a single layer,
// the scaled per-head query and the post-RoPE key exactly as handed to the
// attention op, so the host can recompute softmax(scale·Q·Kᵀ) per head without
// touching the CUDA attention kernel. Implemented by the inference Runner.
type AttentionCaptureAPI interface {
	ExtractAttention(ctx context.Context, tokenIDs []tokenizer.TokenID, layer int32) (inference.AttentionCapture, error)
}

// analyzeAttentionMaxPositions bounds captured tokens; attention weights are
// O(heads·positions²), so this is a smaller cap than the hidden-state tab. It is
// surfaced to the caller, not a hidden choice.
const analyzeAttentionMaxPositions = 48

type analyzeAttentionRequest struct {
	Prompt       string `json:"prompt"`
	Layer        *int   `json:"layer"`
	MaxPositions int    `json:"max_positions"`
}

type analyzeAttentionToken struct {
	ID    int32  `json:"id"`
	Text  string `json:"text"`
	Index int    `json:"index"`
}

type analyzeAttentionResponse struct {
	Layer              int                     `json:"layer"`
	Block              int                     `json:"block_count"`
	Positions          int                     `json:"positions"`
	RequestedPositions int                     `json:"requested_positions"`
	MaxPositions       int                     `json:"max_positions"`
	Truncated          bool                    `json:"truncated"`
	Heads              int                     `json:"heads"`
	KVHeads            int                     `json:"kv_heads"`
	HeadDim            int                     `json:"head_dim"`
	GroupSize          int                     `json:"group_size"`
	Scale              float64                 `json:"scale"`
	Tokens             []analyzeAttentionToken `json:"tokens"`
	// Weights[head][queryPos][keyPos]: the causal softmax attention weight;
	// entries above the diagonal are zero. Exact per the captured tensors.
	Weights [][][]float64 `json:"weights"`
}

// analyzeAttention: capture one layer's query/key over the prompt and report the
// exact per-head causal attention weights softmax(scale·Q·Kᵀ). This recomputes
// the attention distribution on the host from tensors captured at the op
// boundary, so it faithfully reproduces the kernel's pre-softmax logits (any
// query pre-scaling is already folded into the captured query and scale). No
// distribution or shape assumption is made: the weights are the model's own
// normalized attention, reported verbatim.
func (h *Handler) analyzeAttention(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	capture, ok := h.generator.(AttentionCaptureAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "attention capture is unavailable")
		return
	}
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "tokenization is unavailable")
		return
	}
	var body analyzeAttentionRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}

	tokens, err := tokenizerAPI.TokenizeText(body.Prompt, true, true)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	limit := body.MaxPositions
	if limit <= 0 || limit > analyzeAttentionMaxPositions {
		limit = analyzeAttentionMaxPositions
	}
	requestedPositions := len(tokens)
	truncated := requestedPositions > limit
	if truncated {
		tokens = tokens[:limit]
	}
	if len(tokens) < 2 {
		writeInvalidRequest(response, errors.New("attention needs at least two tokens"))
		return
	}

	blockCount := 0
	if properties, ok := h.generator.(ModelPropertiesAPI); ok {
		blockCount = int(properties.ModelProperties().BlockCount)
	}
	layer := blockCount / 2
	if body.Layer != nil {
		layer = *body.Layer
	}
	if blockCount > 0 && (layer < 0 || layer >= blockCount) {
		writeInvalidRequest(response, errors.New("layer is out of range"))
		return
	}

	attention, err := capture.ExtractAttention(request.Context(), tokens, int32(layer))
	if err != nil {
		writeError(response, http.StatusInternalServerError, "capture_failed", err.Error())
		return
	}
	weights, err := attentionWeights(attention, len(tokens))
	if err != nil {
		writeError(response, http.StatusInternalServerError, "capture_failed", err.Error())
		return
	}

	tokenList := make([]analyzeAttentionToken, len(tokens))
	pieceAPI, hasPieces := h.generator.(TokenPieceAPI)
	for index, id := range tokens {
		text := ""
		if hasPieces {
			if piece, pieceErr := pieceAPI.TokenPiece(id); pieceErr == nil {
				text = piece
			}
		}
		tokenList[index] = analyzeAttentionToken{ID: int32(id), Text: text, Index: index}
	}

	group := 1
	if attention.KVHeads > 0 {
		group = attention.Heads / attention.KVHeads
	}
	writeJSON(response, http.StatusOK, analyzeAttentionResponse{
		Layer:              layer,
		Block:              blockCount,
		Positions:          len(tokens),
		RequestedPositions: requestedPositions,
		MaxPositions:       limit,
		Truncated:          truncated,
		Heads:              attention.Heads,
		KVHeads:            attention.KVHeads,
		HeadDim:            attention.HeadDim,
		GroupSize:          group,
		Scale:              float64(attention.Scale),
		Tokens:             tokenList,
		Weights:            weights,
	})
}

// attentionWeights: recompute causal softmax(scale·Q·Kᵀ) per head from the
// captured query/key tensors. Both are row-major with the last dim outermost, so
// element [dim d, head h, token t] lives at Data[d + h*headDim + t*headDim*heads]
// (queries use heads, keys use kvHeads). GQA maps query head h to key head
// h/groupSize. Softmax uses the standard max-shift for numerical stability.
func attentionWeights(capture inference.AttentionCapture, tokens int) ([][][]float64, error) {
	heads, kvHeads, headDim := capture.Heads, capture.KVHeads, capture.HeadDim
	if heads <= 0 || kvHeads <= 0 || headDim <= 0 || tokens <= 0 {
		return nil, errors.New("attention capture geometry is invalid")
	}
	query := capture.Query.Data
	key := capture.Key.Data
	if len(query) != headDim*heads*tokens || len(key) != headDim*kvHeads*tokens {
		return nil, errors.New("attention capture data does not match geometry")
	}
	group := heads / kvHeads
	if group <= 0 {
		group = 1
	}
	scale := float64(capture.Scale)

	weights := make([][][]float64, heads)
	logits := make([]float64, tokens)
	for head := 0; head < heads; head++ {
		kvHead := head / group
		matrix := make([][]float64, tokens)
		for queryPos := 0; queryPos < tokens; queryPos++ {
			row := make([]float64, tokens)
			// Causal: query at position i attends to keys 0..i only.
			maxLogit := math.Inf(-1)
			for keyPos := 0; keyPos <= queryPos; keyPos++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					q := float64(query[d+head*headDim+queryPos*headDim*heads])
					k := float64(key[d+kvHead*headDim+keyPos*headDim*kvHeads])
					dot += q * k
				}
				logit := scale * dot
				logits[keyPos] = logit
				if logit > maxLogit {
					maxLogit = logit
				}
			}
			sum := 0.0
			for keyPos := 0; keyPos <= queryPos; keyPos++ {
				e := math.Exp(logits[keyPos] - maxLogit)
				row[keyPos] = e
				sum += e
			}
			if sum > 0 {
				for keyPos := 0; keyPos <= queryPos; keyPos++ {
					row[keyPos] /= sum
				}
			}
			matrix[queryPos] = row
		}
		weights[head] = matrix
	}
	return weights, nil
}
