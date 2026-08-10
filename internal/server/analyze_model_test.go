package server

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"overgo/internal/inference"
)

func TestAnalyzeModelReportsMeasuredStatistics(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/analyze/model", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result analyzeModelResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v body=%s", err, response.Body.String())
	}

	// Facts pass through from ModelProperties unchanged.
	if result.Model.Architecture != "qwen3" || result.Model.Parameters != 4000000000 {
		t.Fatalf("facts = %+v", result.Model)
	}

	// Derived values are exact ratios of the facts (embedding 2560, heads 32,
	// KV heads 8, blocks 36, params 4e9, size 4.2e9) — no fitting, no rounding
	// of a guessed quantity.
	if result.Derived.HeadDim == nil || *result.Derived.HeadDim != 80 {
		t.Fatalf("head_dim = %v, want 80", result.Derived.HeadDim)
	}
	if result.Derived.KVGroupSize == nil || *result.Derived.KVGroupSize != 4 {
		t.Fatalf("kv_group_size = %v, want 4", result.Derived.KVGroupSize)
	}
	if !result.Derived.GroupedQuery {
		t.Fatal("grouped_query = false, want true (32 != 8)")
	}
	if result.Derived.ParamsPerBlock == nil || *result.Derived.ParamsPerBlock != 4000000000/36 {
		t.Fatalf("params_per_block = %v", result.Derived.ParamsPerBlock)
	}
	if result.Derived.AvgBitsPerWeight == nil || math.Abs(*result.Derived.AvgBitsPerWeight-8.4) > 1e-9 {
		t.Fatalf("avg_bits_per_weight = %v, want 8.4", result.Derived.AvgBitsPerWeight)
	}
	if !slices.Contains(result.Capabilities, "rerank") {
		t.Fatalf("capabilities = %v, want rerank present", result.Capabilities)
	}
}

// deriveModelStatistics must omit ratios whose divisor is zero rather than
// fabricate a value — the distribution-free principle forbids guessing.
func TestAnalyzeModelOmitsRatiosWithZeroDivisor(t *testing.T) {
	handler := newTestHandler(t, &zeroDivisorGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/analyze/model", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	for _, absent := range []string{"head_dim", "kv_group_size", "params_per_block", "avg_bits_per_weight"} {
		if strings.Contains(body, absent) {
			t.Fatalf("expected %q omitted, body=%s", absent, body)
		}
	}
	var result analyzeModelResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if result.Derived.GroupedQuery {
		t.Fatal("grouped_query should be false when KV head count is zero")
	}
}

func TestAnalyzeModelRejectsNonGet(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/analyze/model", nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", response.Code)
	}
}

// zeroDivisorGenerator: a model whose head/block/parameter counts are all zero,
// so every derived ratio must be omitted.
type zeroDivisorGenerator struct {
	fakeGenerator
}

func (g *zeroDivisorGenerator) ModelProperties() inference.ModelProperties {
	return inference.ModelProperties{}
}
