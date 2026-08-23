//go:build modeltest

package seq2seq

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testutil"
	"overgo/internal/textgeneration"
)

func TestRealTextGeneration(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	generator, err := LoadGenerator(filepath.Join(roots.Models, "needle"))
	if err != nil {
		t.Fatalf("Needle artifact unavailable: %v", err)
	}
	prompt := `What's the weather in San Francisco? <tools> [{"name":"get_weather","parameters":{"location":"string"}}]`
	output, err := generateThroughRecipe(t, generator, textgeneration.Request{Text: prompt, MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	payload := strings.TrimSpace(strings.TrimPrefix(output, "<tool_call>"))
	var calls []struct {
		Name      string `json:"name"`
		Arguments struct {
			Location string `json:"location"`
		} `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(payload), &calls); err != nil {
		t.Fatalf("real tool-call output is not JSON: %q: %v", output, err)
	}
	if len(calls) == 0 || calls[0].Name != "get_weather" || calls[0].Arguments.Location != "San Francisco" {
		t.Fatalf("real tool-call output is not grounded: %q", output)
	}
	t.Logf("real 26M tool-call generation: %q", output)
}
