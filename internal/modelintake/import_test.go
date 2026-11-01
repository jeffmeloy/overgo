package modelintake

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/hfgguf"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

func importFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	config := `{"model_type":"llama","architectures":["LlamaForCausalLM"],"max_position_embeddings":16,"hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"rope_theta":10000,"rms_norm_eps":0.000001,"vocab_size":32}`
	if err := os.WriteFile(filepath.Join(root, "config.json"), []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	shapes := map[string][]int{
		"model.embed_tokens.weight": {32, 8}, "model.norm.weight": {8},
		"model.layers.0.input_layernorm.weight": {8}, "model.layers.0.post_attention_layernorm.weight": {8},
		"model.layers.0.self_attn.q_proj.weight": {8, 8}, "model.layers.0.self_attn.k_proj.weight": {4, 8},
		"model.layers.0.self_attn.v_proj.weight": {4, 8}, "model.layers.0.self_attn.o_proj.weight": {8, 8},
		"model.layers.0.mlp.gate_proj.weight": {16, 8}, "model.layers.0.mlp.up_proj.weight": {16, 8}, "model.layers.0.mlp.down_proj.weight": {8, 16},
	}
	values := map[string][]float32{}
	for name, shape := range shapes {
		n := 1
		for _, d := range shape {
			n *= d
		}
		values[name] = make([]float32, n)
	}
	if err := safetensors.Save(filepath.Join(root, "model.safetensors"), values, shapes, nil); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestDeterministicComponentImportDescription(t *testing.T) {
	firstRoot, secondRoot := importFixture(t), importFixture(t)
	first, err := InspectImport(t.Context(), firstRoot, hfgguf.LanguageHidden)
	if err != nil {
		t.Fatal(err)
	}
	second, err := InspectImport(t.Context(), secondRoot, hfgguf.LanguageHidden)
	if err != nil {
		t.Fatal(err)
	}
	if first.Content.Descriptor.ID != second.Content.Descriptor.ID || !bytes.Equal(first.Content.Data, second.Content.Data) {
		t.Fatal("relocation changes description identity")
	}
	var body importDescription
	if err := json.Unmarshal(first.Content.Data, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Components) != 1 || body.Components[0].Unsupported != "" || len(body.Components[0].Bindings) != 11 || len(body.Components[0].Operations) == 0 {
		t.Fatalf("incomplete component description: %+v", body)
	}
	if !slices.Contains(body.Components[0].Operations, "get_rows") {
		t.Fatal("embedding operation is absent")
	}
	if bytes.Contains(first.Content.Data, []byte(firstRoot)) {
		t.Fatal("identity contains local root")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := RegisterImport(t.Context(), store, first); err != nil {
		t.Fatal(err)
	}
	before, _ := store.Head()
	if err := RegisterImport(t.Context(), store, first); err != nil {
		t.Fatal(err)
	}
	after, _ := store.Head()
	if before != after {
		t.Fatal("identical inspection changed store head")
	}
	if err := RegisterImport(t.Context(), store, second); err != nil {
		t.Fatalf("relocated import: %v", err)
	}
	parents, err := store.Parents(t.Context(), *body.Components[0].Profile)
	if err != nil || len(parents) == 0 {
		t.Fatalf("profile derivation was not retained: %v", err)
	}
	changed, err := InspectImport(t.Context(), firstRoot, hfgguf.LanguageTokens)
	if err != nil {
		t.Fatal(err)
	}
	if changed.Content.Descriptor.ID == first.Content.Descriptor.ID {
		t.Fatal("output contract change reused identity")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := InspectImport(ctx, firstRoot, hfgguf.LanguageHidden); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel = %v", err)
	}
	if err := os.WriteFile(filepath.Join(secondRoot, "config.json"), []byte(`{"model_type":"unknown","architectures":["RemoteModel"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	unknown, err := InspectImport(t.Context(), secondRoot, hfgguf.LanguageHidden)
	if err != nil {
		t.Fatal(err)
	}
	body = importDescription{}
	if err := json.Unmarshal(unknown.Content.Data, &body); err != nil {
		t.Fatal(err)
	}
	if body.Components[0].Unsupported == "" || len(body.Components[0].Bindings) != 0 {
		t.Fatal("unknown architecture gained executable mappings")
	}
	if unknown.Content.Descriptor.ID == first.Content.Descriptor.ID {
		t.Fatal("changed source reused identity")
	}
	tampered := first
	tampered.Inventory = unknown.Inventory
	if err := RegisterImport(t.Context(), store, tampered); err == nil {
		t.Fatal("inspection published against a different source identity")
	}
	t.Run("shared-profile-pipeline", func(t *testing.T) {
		pipeline := t.TempDir()
		for _, name := range []string{"encoder_a", "encoder_b"} {
			if err := os.CopyFS(filepath.Join(pipeline, name), os.DirFS(firstRoot)); err != nil {
				t.Fatal(err)
			}
		}
		index := `{"_class_name":"FixturePipeline","encoder_b":["transformers","LlamaForCausalLM"],"encoder_a":["transformers","LlamaForCausalLM"]}`
		if err := os.WriteFile(filepath.Join(pipeline, "model_index.json"), []byte(index), 0600); err != nil {
			t.Fatal(err)
		}
		inspection, err := InspectImport(t.Context(), pipeline, hfgguf.LanguageHidden)
		if err != nil {
			t.Fatal(err)
		}
		if err := RegisterImport(t.Context(), store, inspection); err != nil {
			t.Fatal(err)
		}
		results := make(chan ImportInspection, 3)
		failures := make(chan error, 3)
		for range 3 {
			go func() {
				result, err := InspectImport(t.Context(), pipeline, hfgguf.LanguageHidden)
				results <- result
				failures <- err
			}()
		}
		for range 3 {
			result := <-results
			if err := <-failures; err != nil {
				t.Fatal(err)
			}
			if result.Content.Descriptor.ID != inspection.Content.Descriptor.ID {
				t.Fatal("concurrent inspection changed identity")
			}
		}
	})
	t.Run("declared-policies", func(t *testing.T) {
		pipeline := t.TempDir()
		if err := os.CopyFS(filepath.Join(pipeline, "encoder"), os.DirFS(firstRoot)); err != nil {
			t.Fatal(err)
		}
		files := map[string]string{
			"model_index.json":                `{"_class_name":"RemotePipeline","encoder":["transformers","LlamaForCausalLM"],"scheduler":["remote_module","RemoteScheduler"],"tokenizer":["transformers","Qwen2Tokenizer"]}`,
			"scheduler/scheduler_config.json": `{"_class_name":"RemoteScheduler","shift":5,"num_train_timesteps":1000}`,
			"tokenizer/tokenizer_config.json": `{"tokenizer_class":"Qwen2Tokenizer","model_max_length":4096,"chat_template":"{{ messages }}"}`,
		}
		for name, content := range files {
			path := filepath.Join(pipeline, filepath.FromSlash(name))
			if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(content), 0600); err != nil {
				t.Fatal(err)
			}
		}
		inspection, err := InspectImport(t.Context(), pipeline, "")
		if err != nil {
			t.Fatal(err)
		}
		var body importDescription
		if err := json.Unmarshal(inspection.Content.Data, &body); err != nil {
			t.Fatal(err)
		}
		if len(body.Components) != 3 {
			t.Fatal("missing policy components")
		}
		for _, component := range body.Components {
			if component.Declaration.Name == "encoder" {
				continue
			}
			if component.Unsupported == "" || len(component.Policies) != 1 || len(component.Policies[0].Parameters) == 0 {
				t.Fatalf("policy source lost or acquired execution authority: %+v", component)
			}
		}
		if string(body.Components[1].Policies[0].Parameters["shift"]) != "5" {
			t.Fatal("scheduler declaration lost")
		}
		if err := os.WriteFile(filepath.Join(pipeline, "scheduler", "scheduler_config.json"), []byte(`{"_class_name":"RemoteScheduler","shift":3,"num_train_timesteps":1000}`), 0600); err != nil {
			t.Fatal(err)
		}
		changed, err := InspectImport(t.Context(), pipeline, "")
		if err != nil {
			t.Fatal(err)
		}
		if changed.Content.Descriptor.ID == inspection.Content.Descriptor.ID {
			t.Fatal("scheduler policy change reused identity")
		}
	})
	t.Run("GGUF", func(t *testing.T) {
		path := testutil.QuantizedDenseGGUF(t, 1)
		original, err := InspectImport(t.Context(), path, hfgguf.LanguageTokens)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		copyPath := filepath.Join(t.TempDir(), "relocated.gguf")
		if err := os.WriteFile(copyPath, encoded, 0600); err != nil {
			t.Fatal(err)
		}
		relocated, err := InspectImport(t.Context(), copyPath, hfgguf.LanguageTokens)
		if err != nil {
			t.Fatal(err)
		}
		if original.Content.Descriptor.ID != relocated.Content.Descriptor.ID {
			t.Fatal("GGUF relocation changed description")
		}
		var body importDescription
		if err := json.Unmarshal(original.Content.Data, &body); err != nil {
			t.Fatal(err)
		}
		if len(body.Components) != 1 || body.Components[0].Unsupported != "" {
			t.Fatalf("GGUF language contract = %+v", body.Components)
		}
		if len(body.Components[0].Policies) != 1 || string(body.Components[0].Policies[0].Parameters["tokenizer.ggml.model"]) != `"llama"` {
			t.Fatal("GGUF tokenizer policy was not retained")
		}
		for _, binding := range body.Components[0].Bindings {
			if binding.Source != binding.Target || binding.SourceBytes != binding.TargetBytes || len(binding.Transforms) != 0 {
				t.Fatalf("GGUF acquired an undeclared conversion: %+v", binding)
			}
		}
	})
}
