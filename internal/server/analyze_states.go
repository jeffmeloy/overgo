package server

import (
	"context"
	"errors"
	"net/http"

	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// HiddenStateCaptureAPI: one cacheless forward that returns the residual-stream
// vector entering each requested layer, per token, as host data laid out
// token-major then layer-order then embedding. Implemented by the inference
// Runner (ExtractLayerInputs); no KV cache is retained.
type HiddenStateCaptureAPI interface {
	ExtractLayerInputs(ctx context.Context, tokenIDs []tokenizer.TokenID, layerIDs []int32) (reference.Value, error)
}

type analyzeStatesRequest struct {
	Prompt       string `json:"prompt"`
	Layer        *int   `json:"layer"`
	Metric       string `json:"metric"`
	MaxPositions int    `json:"max_positions"`
	K            int    `json:"k"`
}

type analyzeStatesToken struct {
	ID    int32  `json:"id"`
	Text  string `json:"text"`
	Index int    `json:"index"`
}

type analyzeStatesResponse struct {
	Layer               int                  `json:"layer"`
	Block               int                  `json:"block_count"`
	Positions           int                  `json:"positions"`
	RequestedPositions  int                  `json:"requested_positions"`
	MaxPositions        int                  `json:"max_positions"`
	Truncated           bool                 `json:"truncated"`
	Width               int                  `json:"width"`
	Metric              string               `json:"metric"`
	K                   int                  `json:"k"`
	Tokens              []analyzeStatesToken `json:"tokens"`
	Distance            [][]float64          `json:"distance"`
	Neighbors           [][]int              `json:"neighbors"`
	Layout              [][2]float64         `json:"layout"`
	Stress              float64              `json:"stress"`
	LayoutIterations    int                  `json:"layout_iterations"`
	LayoutMaxIterations int                  `json:"layout_max_iterations"`
	LayoutTolerance     float64              `json:"layout_tolerance"`
}

// analyzeStates: capture the residual-stream vectors at one layer over the
// prompt tokens, then report their distribution-free structure — exact
// cosine/Euclidean distance matrices, a kNN neighbor graph, and a rank-only
// non-metric-MDS layout. Nothing here fits a model or assumes a shape.
func (h *Handler) analyzeStates(response http.ResponseWriter, request *http.Request) {
	if h.config.Analysis == (AnalysisPolicy{}) {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "state analysis policy is unavailable")
		return
	}
	capture, ok := h.generator.(HiddenStateCaptureAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "hidden-state capture is unavailable")
		return
	}
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "tokenization is unavailable")
		return
	}
	var body analyzeStatesRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}

	metric := metricSpearman
	if body.Metric != "" {
		metric = dissimilarityMetric(body.Metric)
		if !validMetric(metric) {
			writeInvalidRequest(response, errors.New("metric must be spearman, cosine, or euclidean"))
			return
		}
	}

	tokens, err := tokenizerAPI.TokenizeText(body.Prompt, true, true)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	limit := body.MaxPositions
	if limit <= 0 || limit > h.config.Analysis.StatePositions {
		limit = h.config.Analysis.StatePositions
	}
	requestedPositions := len(tokens)
	truncated := requestedPositions > limit
	if truncated {
		tokens = tokens[:limit]
	}
	if len(tokens) < 2 {
		writeInvalidRequest(response, errors.New("hidden-state structure needs at least two tokens"))
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

	value, err := capture.ExtractLayerInputs(request.Context(), tokens, []int32{int32(layer)})
	if err != nil {
		writeError(response, http.StatusInternalServerError, "capture_failed", err.Error())
		return
	}
	vectors, width, err := reshapeSingleLayer(value, len(tokens))
	if err != nil {
		writeError(response, http.StatusInternalServerError, "capture_failed", err.Error())
		return
	}

	k := body.K
	if k <= 0 {
		k = defaultNeighborCount(len(tokens))
	}
	distance := dissimilarityMatrix(vectors, metric)
	neighbors := kNNAdjacency(distance, k)
	layout, stress, iterations := nonMetricMDS(distance, h.config.Analysis.MDSIterations, h.config.Analysis.MDSTolerance)

	tokenList := make([]analyzeStatesToken, len(tokens))
	pieceAPI, hasPieces := h.generator.(TokenPieceAPI)
	for index, id := range tokens {
		text := ""
		if hasPieces {
			if piece, pieceErr := pieceAPI.TokenPiece(id); pieceErr == nil {
				text = piece
			}
		}
		tokenList[index] = analyzeStatesToken{ID: int32(id), Text: text, Index: index}
	}

	writeJSON(response, http.StatusOK, analyzeStatesResponse{
		Layer:               layer,
		Block:               blockCount,
		Positions:           len(tokens),
		RequestedPositions:  requestedPositions,
		MaxPositions:        limit,
		Truncated:           truncated,
		Width:               width,
		Metric:              string(metric),
		K:                   k,
		Tokens:              tokenList,
		Distance:            distance,
		Neighbors:           neighbors,
		Layout:              layout,
		Stress:              stress,
		LayoutIterations:    iterations,
		LayoutMaxIterations: h.config.Analysis.MDSIterations,
		LayoutTolerance:     h.config.Analysis.MDSTolerance,
	})
}

// reshapeSingleLayer: split a single-layer ExtractLayerInputs result (Data laid
// out token-major, one embedding-width vector per token) into per-token vectors.
func reshapeSingleLayer(value reference.Value, tokens int) ([][]float32, int, error) {
	if tokens <= 0 || value.Shape.Rank != 2 {
		return nil, 0, errors.New("invalid capture shape")
	}
	width := int(value.Shape.Dims[0])
	if width <= 0 || len(value.Data) != width*tokens {
		return nil, 0, errors.New("capture data does not match shape")
	}
	vectors := make([][]float32, tokens)
	for token := range tokens {
		vectors[token] = value.Data[token*width : (token+1)*width]
	}
	return vectors, width, nil
}
