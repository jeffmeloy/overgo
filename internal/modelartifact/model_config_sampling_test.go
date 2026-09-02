package modelartifact

import (
	"os"
	"path/filepath"
	"testing"
)

// The checkpoint's generation_config.json sampling is extracted with its
// origin; a model card declaration stands over it and must name its
// source; a config without sampling fields declares no sampling.
func TestModelConfigExtractsDeclaredSampling(t *testing.T) {
	directory := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("generation_config.json", `{"bos_token_id":2,"eos_token_id":[1,106],"do_sample":true,"temperature":1.0,"top_k":64,"top_p":0.95}`)
	_, generation, sources, err := ReadModelConfigComponents(directory)
	if err != nil {
		t.Fatal(err)
	}
	if generation == nil || generation.Sampling == nil || generation.Sampling.Origin != "generation_config.json" ||
		!generation.Sampling.DoSample || generation.Sampling.Temperature != 1.0 ||
		generation.Sampling.TopK != 64 || generation.Sampling.TopP != 0.95 || generation.Sampling.MinP != 0 {
		t.Fatalf("sampling = %+v", generation.Sampling)
	}
	if len(sources) != 1 {
		t.Fatalf("sources = %+v", sources)
	}

	write(ModelCardSamplingFile, `{"source":"https://example.test/card","mode":"instruct","do_sample":true,"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5}`)
	_, generation, sources, err = ReadModelConfigComponents(directory)
	if err != nil {
		t.Fatal(err)
	}
	if generation.Sampling.Origin != "https://example.test/card" || generation.Sampling.Mode != "instruct" ||
		generation.Sampling.Temperature != 0.7 || generation.Sampling.TopP != 0.8 || generation.Sampling.TopK != 20 ||
		generation.Sampling.PresencePenalty != 1.5 {
		t.Fatalf("card sampling = %+v", generation.Sampling)
	}
	if len(sources) != 2 {
		t.Fatalf("sources = %+v", sources)
	}

	write(ModelCardSamplingFile, `{"mode":"instruct","temperature":0.7}`)
	if _, _, _, err := ReadModelConfigComponents(directory); err == nil {
		t.Fatal("card declaration without a source was accepted")
	}

	plain := t.TempDir()
	if err := os.WriteFile(filepath.Join(plain, "generation_config.json"), []byte(`{"eos_token_id":151643}`), 0o644); err != nil {
		t.Fatal(err)
	}
	_, generation, _, err = ReadModelConfigComponents(plain)
	if err != nil || generation == nil || generation.Sampling != nil {
		t.Fatalf("config without sampling fields = %+v, %v", generation, err)
	}
}
