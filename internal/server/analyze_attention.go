package server

import (
	"context"
	"errors"
	"math"
	"net/http"
	"slices"

	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

// AttentionCaptureAPI exposes exact host-replay attention boundaries.
type AttentionCaptureAPI interface {
	AttentionCaptureLayers() []int32
	ExtractAttention(ctx context.Context, tokenIDs []tokenizer.TokenID, layer int32) (inference.AttentionCapture, error)
}

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
	// Weights[head][query][key]; upper triangle is zero.
	Weights [][][]float64 `json:"weights"`
}

// analyzeAttention replays one supported causal attention layer on host.
func (h *Handler) analyzeAttention(response http.ResponseWriter, request *http.Request) {
	capture, ok := h.generator.(AttentionCaptureAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "attention capture is unavailable")
		return
	}
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "tokenization is unavailable")
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
	if limit <= 0 {
		limit = len(tokens)
	}
	if owned := h.config.Analysis.StatePositions; owned > 0 {
		limit = min(limit, owned)
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

	layers := capture.AttentionCaptureLayers()
	if len(layers) == 0 {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "exact attention capture is unavailable")
		return
	}
	blockCount := 0
	if properties, ok := h.generator.(ModelPropertiesAPI); ok {
		blockCount = int(properties.ModelProperties().BlockCount)
	}
	layer := int(layers[len(layers)/2])
	if body.Layer != nil {
		layer = *body.Layer
	}
	if !slices.Contains(layers, int32(layer)) {
		writeInvalidRequest(response, errors.New("layer does not support exact attention replay"))
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

// attentionWeights replays causal softmax(scale*Q*K^T), including GQA mapping.
func attentionWeights(capture inference.AttentionCapture, tokens int) ([][][]float64, error) {
	heads, kvHeads, headDim := capture.Heads, capture.KVHeads, capture.HeadDim
	if heads <= 0 || kvHeads <= 0 || heads%kvHeads != 0 || headDim <= 0 || tokens <= 0 ||
		capture.Scale <= 0 || math.IsNaN(float64(capture.Scale)) || math.IsInf(float64(capture.Scale), 0) {
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
	for head := range heads {
		kvHead := head / group
		matrix := make([][]float64, tokens)
		for queryPos := range tokens {
			row := make([]float64, tokens)
			// Causal: query at position i attends to keys 0..i only.
			maxLogit := math.Inf(-1)
			for keyPos := 0; keyPos <= queryPos; keyPos++ {
				dot := 0.0
				for d := range headDim {
					q := float64(query[d+head*headDim+queryPos*headDim*heads])
					k := float64(key[d+kvHead*headDim+keyPos*headDim*kvHeads])
					if math.IsNaN(q) || math.IsInf(q, 0) || math.IsNaN(k) || math.IsInf(k, 0) {
						return nil, errors.New("attention capture contains non-finite values")
					}
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
