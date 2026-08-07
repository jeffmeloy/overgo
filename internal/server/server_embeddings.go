package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"

	"overgo/internal/inference"
	"overgo/internal/media"
)

type embeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
}

type embeddingItem struct {
	Object         string `json:"object"`
	Embedding      any    `json:"embedding"`
	Index          int    `json:"index"`
	EncodingFormat string `json:"encoding_format,omitempty"`
}

type embeddingResponse struct {
	Object string          `json:"object"`
	Data   []embeddingItem `json:"data"`
	Model  string          `json:"model"`
	Usage  embeddingUsage  `json:"usage"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type rerankRequest struct {
	Model      string   `json:"model"`
	Query      *string  `json:"query"`
	Documents  []string `json:"documents"`
	Texts      []string `json:"texts"`
	TopN       *int     `json:"top_n"`
	ReturnText bool     `json:"return_text"`
}

type rerankItem struct {
	Index          int      `json:"index"`
	RelevanceScore *float32 `json:"relevance_score,omitempty"`
	Score          *float32 `json:"score,omitempty"`
	Text           *string  `json:"text,omitempty"`
}

type rerankResponse struct {
	Model   string         `json:"model"`
	Object  string         `json:"object"`
	Usage   embeddingUsage `json:"usage"`
	Results []rerankItem   `json:"results"`
}

func (h *Handler) rerank(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	ranker, ok := h.generator.(Ranker)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "reranking is unavailable")
		return
	}
	if capability, ok := h.generator.(RankCapability); ok && !capability.SupportsRank() {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "reranking is unavailable")
		return
	}
	var body rerankRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if !h.requireModel(response, body.Model) {
		return
	}
	if body.Query == nil {
		writeInvalidRequestMessage(response, "query must be provided")
		return
	}
	tei := body.Texts != nil
	documents := body.Documents
	if documents == nil {
		documents = body.Texts
	}
	if len(documents) == 0 {
		writeInvalidRequestMessage(response, "documents must be a non-empty string array")
		return
	}
	if len(documents) > h.config.MaxEmbeddingInputs {
		writeInvalidRequestMessage(
			response, fmt.Sprintf("rerank document count exceeds %d", h.config.MaxEmbeddingInputs),
		)
		return
	}
	topN := len(documents)
	if body.TopN != nil {
		if *body.TopN < 0 {
			writeInvalidRequestMessage(response, "top_n must be non-negative")
			return
		}
		topN = min(*body.TopN, len(documents))
	}
	slotID, acquired := h.acquireRequestSlot(response, -1)
	if !acquired {
		return
	}
	defer h.releaseSlot(slotID)
	items := make([]rerankItem, len(documents))
	usage := embeddingUsage{}
	for index, document := range documents {
		result, err := ranker.RankPair(request.Context(), *body.Query, document)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		if len(result.Scores) == 0 {
			writeGenerationError(response, errors.New("server: rerank result has no scores"))
			return
		}
		score := result.Scores[0]
		item := rerankItem{Index: index}
		if tei {
			item.Score = &score
			if body.ReturnText {
				text := document
				item.Text = &text
			}
		} else {
			item.RelevanceScore = &score
		}
		items[index] = item
		usage.PromptTokens += result.Tokens
		usage.TotalTokens += result.Tokens
	}
	sort.SliceStable(items, func(left, right int) bool {
		leftScore, rightScore := items[left].RelevanceScore, items[right].RelevanceScore
		if tei {
			leftScore, rightScore = items[left].Score, items[right].Score
		}
		return *leftScore > *rightScore
	})
	items = items[:topN]
	if tei {
		writeJSON(response, http.StatusOK, items)
		return
	}
	writeJSON(response, http.StatusOK, rerankResponse{
		Model: h.config.ModelID, Object: "list", Usage: usage, Results: items,
	})
}

func (h *Handler) embeddings(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	embedder, ok := h.generator.(Embedder)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "embeddings are unavailable")
		return
	}
	var body embeddingRequest
	if !h.decodeJSONWithLimit(response, request, &body, maxRequestBytes) {
		return
	}
	if !h.requireModel(response, body.Model) {
		return
	}
	if body.EncodingFormat != "" &&
		body.EncodingFormat != "float" &&
		body.EncodingFormat != "base64" {
		writeInvalidRequestMessage(response, "encoding_format must be float or base64")
		return
	}
	inputs, err := h.parseNativePrompts(request.Context(), body.Input)
	if err != nil {
		writeInvalidRequestMessage(response, "embedding input: "+err.Error())
		return
	}
	if len(inputs) > h.config.MaxEmbeddingInputs {
		writeInvalidRequestMessage(
			response, fmt.Sprintf("embedding input count exceeds %d", h.config.MaxEmbeddingInputs),
		)
		return
	}
	slotID, acquired := h.acquireRequestSlot(response, -1)
	if !acquired {
		return
	}
	defer h.releaseSlot(slotID)
	result := embeddingResponse{
		Object: "list",
		Data:   make([]embeddingItem, len(inputs)),
		Model:  h.config.ModelID,
	}
	for index, input := range inputs {
		vector, tokens, embedErr := h.embedPrompt(request.Context(), embedder, input)
		if embedErr != nil {
			writeGenerationError(response, embedErr)
			return
		}
		var encoded any = vector
		format := ""
		if body.EncodingFormat == "base64" {
			encoded = encodeFloat32Base64(vector)
			format = "base64"
		}
		result.Data[index] = embeddingItem{
			Object:         "embedding",
			Embedding:      encoded,
			Index:          index,
			EncodingFormat: format,
		}
		result.Usage.PromptTokens += tokens
		result.Usage.TotalTokens += tokens
	}
	writeJSON(response, http.StatusOK, result)
}

type nativeEmbeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	Content        json.RawMessage `json:"content"`
	EmbdNormalize  *int            `json:"embd_normalize"`
	EncodingFormat string          `json:"encoding_format"`
	Pooling        string          `json:"pooling"`
}

type nativeEmbeddingItem struct {
	Index     int         `json:"index"`
	Embedding [][]float32 `json:"embedding"`
}

func (h *Handler) nativeEmbeddings(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	embedder, ok := h.generator.(Embedder)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "embeddings are unavailable")
		return
	}
	var body nativeEmbeddingRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if !h.requireModel(response, body.Model) {
		return
	}
	if body.EncodingFormat != "" && body.EncodingFormat != "float" {
		writeInvalidRequestMessage(response, "encoding_format must be float for native embeddings")
		return
	}
	pooling := inference.EmbeddingPooling(body.Pooling)
	if pooling == "" {
		pooling = inference.EmbeddingPoolingMean
	}
	if pooling != inference.EmbeddingPoolingMean &&
		pooling != inference.EmbeddingPoolingLast &&
		pooling != inference.EmbeddingPoolingNone {
		writeInvalidRequestMessage(response, "pooling must be mean, last, or none")
		return
	}
	normalize := 2
	if body.EmbdNormalize != nil {
		normalize = *body.EmbdNormalize
	}
	raw := body.Input
	if len(raw) == 0 {
		raw = body.Content
	}
	inputs, err := h.parseNativePrompts(request.Context(), raw)
	if err != nil {
		writeInvalidRequestMessage(response, "embedding input: "+err.Error())
		return
	}
	if len(inputs) > h.config.MaxEmbeddingInputs {
		writeInvalidRequestMessage(
			response, fmt.Sprintf("embedding input count exceeds %d", h.config.MaxEmbeddingInputs),
		)
		return
	}
	slotID, acquired := h.acquireRequestSlot(response, -1)
	if !acquired {
		return
	}
	defer h.releaseSlot(slotID)
	result := make([]nativeEmbeddingItem, len(inputs))
	for index, input := range inputs {
		embedded, embedErr := h.embedPromptAdvanced(
			request.Context(),
			embedder,
			input,
			inference.EmbeddingOptions{Pooling: pooling, Normalize: normalize},
		)
		if embedErr != nil {
			writeGenerationError(response, embedErr)
			return
		}
		result[index] = nativeEmbeddingItem{
			Index:     index,
			Embedding: embedded.Vectors,
		}
	}
	writeJSON(response, http.StatusOK, result)
}

func (h *Handler) embedPromptAdvanced(
	ctx context.Context,
	embedder Embedder,
	prompt nativePrompt,
	options inference.EmbeddingOptions,
) (inference.EmbeddingResult, error) {
	if advanced, ok := h.generator.(AdvancedEmbedder); ok {
		if prompt.TokenIDs == nil {
			return advanced.EmbedAdvanced(ctx, prompt.Text, options)
		}
		return advanced.EmbedTokensAdvanced(ctx, prompt.TokenIDs, options)
	}
	if options.Pooling != inference.EmbeddingPoolingMean || options.Normalize != 2 {
		return inference.EmbeddingResult{}, errors.New(
			"generator does not expose advanced embedding modes",
		)
	}
	vector, tokens, err := h.embedPrompt(ctx, embedder, prompt)
	if err != nil {
		return inference.EmbeddingResult{}, err
	}
	return inference.EmbeddingResult{Vectors: [][]float32{vector}, Tokens: tokens}, nil
}

func encodeFloat32Base64(values []float32) string {
	return base64.StdEncoding.EncodeToString(media.EncodeFloat32LE(values))
}

func (h *Handler) embedPrompt(
	ctx context.Context,
	embedder Embedder,
	prompt nativePrompt,
) ([]float32, int, error) {
	if prompt.TokenIDs == nil {
		return embedder.Embed(ctx, prompt.Text)
	}
	tokenEmbedder, ok := h.generator.(TokenEmbedder)
	if !ok {
		return nil, 0, errors.New("server: exact-token embeddings are unavailable")
	}
	return tokenEmbedder.EmbedTokens(ctx, prompt.TokenIDs)
}
