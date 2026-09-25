package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

// TestNativeCompletionUsesThePump holds the native completion engine to the
// shared generation pump: a hosted provider tokenizes at its end, so the
// counts the native response reports are the provider's accounting, as every
// other text route reports them, not the relay's id arithmetic.
func TestNativeCompletionUsesThePump(t *testing.T) {
	// Serial: a helper it calls sets the process environment.
	generator, environment, _ := newRelayGenerator(t, []string{"Hello"})
	policy, supported, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !supported {
		t.Fatalf("inference runtime policy = %v supported=%v", err, supported)
	}
	handler, err := New(Config{
		RuntimePolicy: policy, ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature,
		DefaultTopP: testFullTopP, Analysis: testAnalysisPolicy, Environment: environment,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	response := serveTestRequest(handler, http.MethodPost, "/completion", `{"prompt":"hi","n_predict":4}`)
	var completion nativeCompletionResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &completion) != nil {
		t.Fatalf("native completion status=%d body=%s", response.Code, response.Body.String())
	}
	if completion.Content != "Hello" || completion.TokensEvaluated != 7 || completion.TokensPredicted != 1 ||
		completion.Timings.PromptN != 7 || completion.Timings.PredictedN != 1 {
		t.Fatalf("native counts are not the provider's: %s", response.Body.String())
	}
}
