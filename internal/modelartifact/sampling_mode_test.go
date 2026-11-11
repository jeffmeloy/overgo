package modelartifact

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestAModelIsServedTheSamplingOfItsMode holds the record's half of loop-proof/sampling-by-mode through the declaration
// file a model's record is read from: Qwen3.8's page recommends one set thinking and another with thinking off, and a
// declaration of both is recorded as the set to think with and the thinking-off set, each as written; one set is
// recorded alone, as before. The policy's half and the server's are the tests of this name in internal/modelrecipe and
// internal/server.
func TestAModelIsServedTheSamplingOfItsMode(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	recorded := func(declaration string) map[string]GenerationSampling {
		t.Helper()
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, ModelCardSamplingFile), []byte(declaration), 0o600))
		_, generation, _, err := ReadModelConfigComponents(directory)
		if err != nil || generation == nil {
			t.Fatalf("the declaration %s was not recorded: %v", declaration, err)
		}
		var sets map[string]GenerationSampling
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(json.Marshal(generation))(t), &sets))
		return sets
	}
	two := recorded(`[
		{"source": "https://huggingface.co/Qwen/Qwen3.8-27B", "mode": "thinking", "do_sample": true,
		 "temperature": 1.0, "top_p": 0.95, "top_k": 20},
		{"source": "https://huggingface.co/Qwen/Qwen3.8-27B", "mode": "thinking-off", "do_sample": true,
		 "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
	]`)
	if thinking := two["sampling"]; thinking.Temperature != 1 || thinking.TopP != 0.95 || thinking.TopK != 20 || thinking.PresencePenalty != 0 {
		t.Fatalf("the set to think with was recorded as %+v", thinking)
	}
	if off, ok := two["thinking_off_sampling"]; !ok || off.Temperature != 0.7 || off.TopP != 0.8 || off.PresencePenalty != 1.5 {
		t.Fatalf("the thinking-off set was recorded as %+v (present %t); want temperature 0.7, top-p 0.8, presence penalty 1.5", off, ok)
	}
	one := recorded(`{"source": "https://example.com/card", "do_sample": true, "temperature": 0.6}`)
	if _, apart := one["thinking_off_sampling"]; apart || one["sampling"].Temperature != 0.6 {
		t.Fatalf("one set was recorded as %+v; want it alone, serving both modes", one)
	}
}
