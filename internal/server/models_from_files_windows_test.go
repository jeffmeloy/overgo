//go:build windows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestModelsRunFromTheirOwnFiles closes SQA-283 on the device: a GGUF and a safetensors directory of prototype-known
// architectures, llama and Qwen 3.5, with nothing recorded of them, answer a chat through the template their own files
// declare, and the response and the record of the attempt both say they are not verified.
func TestModelsRunFromTheirOwnFiles(t *testing.T) {
	// Serial: device fixtures share the one card.
	cudatest.Require(t)
	const (
		template        = `{% for message in messages %}a{{ message['content'] }}{% endfor %}`
		directoryTokens = 32
	)
	directory := testutil.Qwen35Directory(t)
	testutil.WriteTextFile(t, directory, "chat_template.jinja", template)
	// The fixture's tokens spell nothing; these are as many, each one letter, so it can read a prompt.
	vocabulary := map[string]int{}
	for token := range directoryTokens {
		vocabulary[string(rune('['+token))] = token
	}
	tokenizer := testcheck.Must(json.Marshal(map[string]any{"model": map[string]any{"merges": []string{}, "vocab": vocabulary}}))(t)
	testutil.WriteTextFile(t, directory, "tokenizer.json", string(tokenizer))
	for name, path := range map[string]string{
		"a GGUF":                  testutil.HermeticLlamaGGUFWith(t, 64, testutil.GGUFScalar("tokenizer.chat_template", gguf.ValueTypeString, template)),
		"a safetensors directory": directory,
	} {
		t.Run(name, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "store")
			created := testcheck.Must(overgodb.Create(root))(t)
			testcheck.NoError(t, created.Close())
			runner := testcheck.Must(modelcli.OpenRunner(t.Context(), root, path, inference.OpenOptions{}))(t)
			defer runner.Close()
			store := overgodbtest.Open(t, root)
			handler := testcheck.Must(New(Config{
				ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
				Analysis: testAnalysisPolicy, Repository: store,
			}, runner))(t)
			t.Cleanup(func() { handler.Close() })
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
				`{"model":"`+testModelID+`","messages":[{"role":"user","content":"b"}],"max_tokens":2,"temperature":0}`)))
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
			var answer chatResponse
			testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &answer))
			state := string(modelrecipe.StateNotVerified)
			if answer.Verification != state || response.Header().Get(modelVerificationHeader) != state {
				t.Fatalf("answer reads %q, header %q", answer.Verification, response.Header().Get(modelVerificationHeader))
			}
			attempts := 0
			if _, err := overgodb.VisitDecodedDocuments(t.Context(), store, overgodb.DocumentQuery{
				Contracts: []artifact.DocumentContract{{
					Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
				}},
				Order: overgodb.DocumentNewestFirst,
			}, runrecord.ParseServingObservation, func(_ overgodb.DocumentView, observation runrecord.ServingObservation) error {
				if observation.Task == recipe.TaskInference && observation.Verification == state {
					attempts++
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if attempts == 0 {
				t.Fatal("the record of the attempt does not say the model is not verified")
			}
		})
	}
}
