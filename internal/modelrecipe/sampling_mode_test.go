package modelrecipe

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
)

// TestAModelIsServedTheSamplingOfItsMode holds a model's sampling per mode (loop-proof/sampling-by-mode): a model card
// declaration naming a set to think with and a set with thinking off is read into the model's record as two sets, the
// runtime policy carries both under one identity, and a request is served the first thinking and the second with
// thinking off. A model declaring one set is served it in both modes, as before, and a declaration with only a
// thinking-off set is refused. The server's half, a request's mode choosing its set and its own values standing, is
// the test of this name in internal/server.
func TestAModelIsServedTheSamplingOfItsMode(t *testing.T) {
	t.Parallel()
	catalog, found, err := CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("catalog policy = %v, %v", found, err)
	}
	directory := t.TempDir()
	declare := func(declaration string) (*modelartifact.GenerationEssentials, error) {
		t.Helper()
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, modelartifact.ModelCardSamplingFile), []byte(declaration), 0o600))
		_, generation, _, err := modelartifact.ReadModelConfigComponents(directory)
		return generation, err
	}
	// Qwen3.8's page recommends one set thinking and another with thinking off.
	two := testcheck.Must(declare(`[
		{"source": "https://huggingface.co/Qwen/Qwen3.8-27B", "mode": "thinking", "do_sample": true,
		 "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0},
		{"source": "https://huggingface.co/Qwen/Qwen3.8-27B", "mode": "thinking-off", "do_sample": true,
		 "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 1.5}
	]`))(t)
	if two.Sampling == nil || two.ThinkingOffSampling == nil {
		t.Fatalf("a declaration of two sets was recorded as %+v and %+v; want one for each mode", two.Sampling, two.ThinkingOffSampling)
	}
	policy := testcheck.Must(withDeclaredSampling(catalog, two.Sampling, two.ThinkingOffSampling))(t)
	testcheck.NoError(t, policy.ValidateIdentity())
	if thinking := policy.SamplingFor(true); thinking.Temperature != 1 || thinking.TopP != 0.95 || thinking.TopK != 20 || thinking.PresencePenalty != 0 {
		t.Fatalf("a thinking request was served %+v; want temperature 1, top-p 0.95, top-k 20 and no penalty", thinking)
	}
	if off := policy.SamplingFor(false); off.Temperature != 0.7 || off.TopP != 0.8 || off.TopK != 20 || off.PresencePenalty != 1.5 {
		t.Fatalf("a request with thinking off was served %+v; want temperature 0.7, top-p 0.8, top-k 20 and presence penalty 1.5", off)
	}

	one := testcheck.Must(declare(`{"source": "https://example.com/card", "do_sample": true, "temperature": 0.6}`))(t)
	single := testcheck.Must(withDeclaredSampling(catalog, one.Sampling, one.ThinkingOffSampling))(t)
	if served := single.SamplingFor(false); served.Temperature != 0.6 || !reflect.DeepEqual(served, single.SamplingFor(true)) {
		t.Fatalf("a model of one set was served %+v with thinking off and %+v thinking; want its one set in both", served, single.SamplingFor(true))
	}
	if _, err := declare(`[{"source": "https://example.com/card", "mode": "thinking-off", "do_sample": true, "temperature": 0.7}]`); err == nil {
		t.Fatal("a declaration with only a thinking-off set was accepted; want it refused, having no set to think with")
	}
}
