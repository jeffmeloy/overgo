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
	"overgo/internal/overgodbtest"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func importFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	config := `{"model_type":"llama","architectures":["LlamaForCausalLM"],"max_position_embeddings":16,"hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"rope_theta":10000,"rms_norm_eps":0.000001,"vocab_size":32}`
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "config.json"), []byte(config), 0600))
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
	testcheck.NoError(t, safetensors.Save(filepath.Join(root, "model.safetensors"), values, shapes, nil))
	return root
}

func TestDeterministicComponentImportDescription(t *testing.T) {
	firstRoot, secondRoot := importFixture(t), importFixture(t)
	first := testcheck.Must(InspectImport(t.Context(), firstRoot, hfgguf.LanguageHidden))(t)
	second := testcheck.Must(InspectImport(t.Context(), secondRoot, hfgguf.LanguageHidden))(t)
	if first.Content.Descriptor.ID != second.Content.Descriptor.ID || !bytes.Equal(first.Content.Data, second.Content.Data) {
		t.Fatal("relocation changes description identity")
	}
	var body importDescription
	testcheck.NoError(t, json.Unmarshal(first.Content.Data, &body))
	if len(body.Components) != 1 || body.Components[0].Unsupported != "" || len(body.Components[0].Bindings) != 11 || len(body.Components[0].Operations) == 0 {
		t.Fatalf("incomplete component description: %+v", body)
	}
	if !slices.Contains(body.Components[0].Operations, "get_rows") {
		t.Fatal("embedding operation is absent")
	}
	if bytes.Contains(first.Content.Data, []byte(firstRoot)) {
		t.Fatal("identity contains local root")
	}
	store := overgodbtest.Open(t, t.TempDir())
	testcheck.NoError(t, RegisterImport(t.Context(), store, first))
	before, _ := store.Head()
	testcheck.NoError(t, RegisterImport(t.Context(), store, first))
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
	changed := testcheck.Must(InspectImport(t.Context(), firstRoot, hfgguf.LanguageTokens))(t)
	if changed.Content.Descriptor.ID == first.Content.Descriptor.ID {
		t.Fatal("output contract change reused identity")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := InspectImport(ctx, firstRoot, hfgguf.LanguageHidden); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel = %v", err)
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(secondRoot, "config.json"), []byte(`{"model_type":"unknown","architectures":["RemoteModel"]}`), 0600))
	unknown := testcheck.Must(InspectImport(t.Context(), secondRoot, hfgguf.LanguageHidden))(t)
	body = importDescription{}
	testcheck.NoError(t, json.Unmarshal(unknown.Content.Data, &body))
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
			testcheck.NoError(t, os.CopyFS(filepath.Join(pipeline, name), os.DirFS(firstRoot)))
		}
		index := `{"_class_name":"FixturePipeline","encoder_b":["transformers","LlamaForCausalLM"],"encoder_a":["transformers","LlamaForCausalLM"]}`
		testcheck.NoError(t, os.WriteFile(filepath.Join(pipeline, "model_index.json"), []byte(index), 0600))
		inspection := testcheck.Must(InspectImport(t.Context(), pipeline, hfgguf.LanguageHidden))(t)
		testcheck.NoError(t, RegisterImport(t.Context(), store, inspection))
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
		testcheck.NoError(t, os.CopyFS(filepath.Join(pipeline, "encoder"), os.DirFS(firstRoot)))
		files := map[string]string{
			"model_index.json":                `{"_class_name":"RemotePipeline","encoder":["transformers","LlamaForCausalLM"],"scheduler":["remote_module","RemoteScheduler"],"tokenizer":["transformers","Qwen2Tokenizer"]}`,
			"scheduler/scheduler_config.json": `{"_class_name":"RemoteScheduler","shift":5,"num_train_timesteps":1000}`,
			"tokenizer/tokenizer_config.json": `{"tokenizer_class":"Qwen2Tokenizer","model_max_length":4096,"chat_template":"{{ messages }}"}`,
		}
		for name, content := range files {
			path := filepath.Join(pipeline, filepath.FromSlash(name))
			testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0700))
			testcheck.NoError(t, os.WriteFile(path, []byte(content), 0600))
		}
		inspection := testcheck.Must(InspectImport(t.Context(), pipeline, ""))(t)
		var body importDescription
		testcheck.NoError(t, json.Unmarshal(inspection.Content.Data, &body))
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
		testcheck.NoError(t, os.WriteFile(filepath.Join(pipeline, "scheduler", "scheduler_config.json"), []byte(`{"_class_name":"RemoteScheduler","shift":3,"num_train_timesteps":1000}`), 0600))
		changed := testcheck.Must(InspectImport(t.Context(), pipeline, ""))(t)
		if changed.Content.Descriptor.ID == inspection.Content.Descriptor.ID {
			t.Fatal("scheduler policy change reused identity")
		}
	})
	t.Run("GGUF", func(t *testing.T) {
		path := testutil.QuantizedDenseGGUF(t, 1)
		original := testcheck.Must(InspectImport(t.Context(), path, hfgguf.LanguageTokens))(t)
		encoded := testcheck.Must(os.ReadFile(path))(t)
		copyPath := filepath.Join(t.TempDir(), "relocated.gguf")
		testcheck.NoError(t, os.WriteFile(copyPath, encoded, 0600))
		relocated := testcheck.Must(InspectImport(t.Context(), copyPath, hfgguf.LanguageTokens))(t)
		if original.Content.Descriptor.ID != relocated.Content.Descriptor.ID {
			t.Fatal("GGUF relocation changed description")
		}
		var body importDescription
		testcheck.NoError(t, json.Unmarshal(original.Content.Data, &body))
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
