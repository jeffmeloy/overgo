package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	server "overgo/internal/server"
	"overgo/internal/servingtest"
	"overgo/internal/testevidence"
)

// TestAnalyzeAttentionEndToEnd exercises the whole /analyze/attention path
// against a real dense Qwen2 model served by a real Runner: HTTP request →
// engine attention capture → host softmax recompute → JSON. It asserts the
// weights are causal (upper triangle zero) and each query row is a probability
// distribution (sums to 1). Gated on OVERGO_QWEN2_MODEL (a dense GGUF); skipped
// otherwise and under -short.
func TestAnalyzeAttentionEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	modelPath := os.Getenv("OVERGO_QWEN2_MODEL")
	if modelPath == "" {
		t.Skip("OVERGO_QWEN2_MODEL is not set")
	}
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		modelPath, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithProgram(&loaded, inference.OpenOptions{DeviceOrdinal: 0})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()

	handler, err := server.New(server.Config{ModelID: "qwen2-attn", MaxTokens: 128}, runner)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	httpServer := httptest.NewServer(handler)
	defer httpServer.Close()

	body, _ := json.Marshal(map[string]any{"prompt": "The quick brown fox jumps", "layer": 6})
	request, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, httpServer.URL+"/analyze/attention", bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", response.StatusCode)
	}

	var payload struct {
		Heads     int           `json:"heads"`
		KVHeads   int           `json:"kv_heads"`
		Positions int           `json:"positions"`
		GroupSize int           `json:"group_size"`
		Scale     float64       `json:"scale"`
		Weights   [][][]float64 `json:"weights"`
	}
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	if payload.Heads <= 0 || payload.Positions < 2 || len(payload.Weights) != payload.Heads {
		t.Fatalf("bad payload: heads=%d positions=%d weights=%d", payload.Heads, payload.Positions, len(payload.Weights))
	}
	const eps = 1e-4
	for head := 0; head < payload.Heads; head++ {
		matrix := payload.Weights[head]
		if len(matrix) != payload.Positions {
			t.Fatalf("head %d has %d rows, want %d", head, len(matrix), payload.Positions)
		}
		for i := 0; i < payload.Positions; i++ {
			sum := 0.0
			for j := 0; j < payload.Positions; j++ {
				if j > i && matrix[i][j] != 0 {
					t.Fatalf("head %d cell (%d,%d) = %g, want 0 (causal)", head, i, j, matrix[i][j])
				}
				if matrix[i][j] < 0 {
					t.Fatalf("head %d cell (%d,%d) = %g, want >= 0", head, i, j, matrix[i][j])
				}
				sum += matrix[i][j]
			}
			if sum < 1-eps || sum > 1+eps {
				t.Fatalf("head %d row %d sums to %g, want 1", head, i, sum)
			}
		}
	}
	t.Logf("heads=%d kvHeads=%d group=%d positions=%d scale=%g — all rows causal and normalized",
		payload.Heads, payload.KVHeads, payload.GroupSize, payload.Positions, payload.Scale)
}
