package server

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// statesGenerator: a fakeGenerator that returns a fixed token list and known
// per-token hidden-state vectors, so the /analyze/states structure computation
// is exercised without a real model.
type statesGenerator struct {
	fakeGenerator
	tokens  []tokenizer.TokenID
	vectors []float32 // token-major, width floats per token
	width   int
}

func (g *statesGenerator) TokenizeText(string, bool, bool) ([]tokenizer.TokenID, error) {
	return g.tokens, nil
}

func (g *statesGenerator) ExtractLayerInputs(_ context.Context, tokenIDs []tokenizer.TokenID, layerIDs []int32) (reference.Value, error) {
	// Return only the (possibly truncated) tokens the handler asked for.
	n := len(tokenIDs)
	return reference.Value{
		Shape: tensor.MustShape(uint64(g.width), uint64(n)),
		Data:  g.vectors[:g.width*n],
	}, nil
}

func threeTokenStates() *statesGenerator {
	// token0 == token1 (distance 0); token2 orthogonal.
	return &statesGenerator{
		tokens: []tokenizer.TokenID{10, 11, 12},
		width:  4,
		vectors: []float32{
			1, 0, 0, 0,
			1, 0, 0, 0,
			0, 0, 0, 1,
		},
	}
}

func TestAnalyzeStatesReportsStructure(t *testing.T) {
	handler := newTestHandler(t, threeTokenStates())
	// Request cosine explicitly for closed-form distances (0 and 1).
	response := serveTestRequest(handler, http.MethodPost, "/analyze/states", `{"prompt":"anything","layer":5,"metric":"cosine","k":2}`)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result analyzeStatesResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if result.Positions != 3 || result.Width != 4 || result.Layer != 5 || result.Metric != "cosine" {
		t.Fatalf("positions/width/layer/metric = %d/%d/%d/%s", result.Positions, result.Width, result.Layer, result.Metric)
	}
	if math.Abs(result.Distance[0][1]) > 1e-6 || math.Abs(result.Distance[0][2]-1) > 1e-6 {
		t.Fatalf("distance = %v", result.Distance)
	}
	if len(result.Neighbors[0]) == 0 || result.Neighbors[0][0] != 1 {
		t.Fatalf("neighbors[0] = %v", result.Neighbors[0])
	}
	if len(result.Layout) != 3 || result.Stress < 0 || result.Stress > 1.0001 || result.LayoutIterations <= 0 {
		t.Fatalf("layout=%d stress=%v iters=%d", len(result.Layout), result.Stress, result.LayoutIterations)
	}
	if len(result.Tokens) != 3 || result.Tokens[0].Text == "" {
		t.Fatalf("tokens = %+v", result.Tokens)
	}
}

// The default metric is the assumption-light one (Spearman rank correlation),
// not a silently-privileged Euclidean/cosine geometry.
func TestAnalyzeStatesDefaultsToSpearman(t *testing.T) {
	handler := newTestHandler(t, threeTokenStates())
	response := serveTestRequest(handler, http.MethodPost, "/analyze/states", `{"prompt":"anything"}`)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
	var result analyzeStatesResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if result.Metric != "spearman" {
		t.Fatalf("default metric = %q, want spearman", result.Metric)
	}
}

func TestAnalyzeStatesSurfacesTruncation(t *testing.T) {
	// Five tokens, but only three positions permitted → truncation is reported,
	// not silent.
	gen := &statesGenerator{
		tokens:  []tokenizer.TokenID{1, 2, 3, 4, 5},
		width:   2,
		vectors: []float32{0, 0, 1, 0, 2, 0, 3, 0, 4, 0},
	}
	handler := newTestHandler(t, gen)
	response := serveTestRequest(handler, http.MethodPost, "/analyze/states", `{"prompt":"x","max_positions":3}`)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result analyzeStatesResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !result.Truncated || result.Positions != 3 || result.RequestedPositions != 5 || result.MaxPositions != 3 {
		t.Fatalf("truncation not surfaced: %+v", result)
	}
	if len(result.Distance) != 3 {
		t.Fatalf("distance size = %d, want 3", len(result.Distance))
	}
}

func TestAnalyzeStatesRejectsInvalidMetric(t *testing.T) {
	handler := newTestHandler(t, threeTokenStates())
	response := serveTestRequest(handler, http.MethodPost, "/analyze/states", `{"prompt":"x","metric":"banana"}`)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", response.Code)
	}
}

func TestAnalyzeStatesUnsupportedWithoutCapture(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodPost, "/analyze/states", `{"prompt":"x"}`)
	if response.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501", response.Code)
	}
}

func TestAnalyzeStatesRejectsShortPrompt(t *testing.T) {
	gen := &statesGenerator{tokens: []tokenizer.TokenID{7}, width: 2, vectors: []float32{1, 0}}
	handler := newTestHandler(t, gen)
	response := serveTestRequest(handler, http.MethodPost, "/analyze/states", `{"prompt":"x"}`)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", response.Code)
	}
}

func TestAnalyzeStatesRejectsNonPost(t *testing.T) {
	handler := newTestHandler(t, threeTokenStates())
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/analyze/states", nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", response.Code)
	}
}
