package modelartifact

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

func pipelineFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"model_index.json":                `{"_class_name":"MOSSImagePipeline","transformer":["pipeline_moss_image","MOSSImageTransformer"],"text_encoder":["transformers","Qwen3_5Model"],"scheduler":["pipeline_moss_image","MOSSImageScheduler"],"tokenizer":["transformers","Qwen2Tokenizer"],"optional":[null,null]}`,
		"pipeline_moss_image.py":          "raise RuntimeError('inventory must never execute source')",
		"transformer/config.json":         `{"_class_name":"MOSSImageTransformer","dim":1}`,
		"text_encoder/config.json":        `{"architectures":["Qwen3_5Model"],"hidden_size":1}`,
		"scheduler/scheduler_config.json": `{"_class_name":"MOSSImageScheduler","shift":5}`,
		"tokenizer/tokenizer_config.json": `{"tokenizer_class":"Qwen2Tokenizer"}`,
		"tokenizer/vocab.json":            `{"token":0}`,
		"tokenizer/merges.txt":            "#version: 0.2\n",
		"README.md":                       "fixture documentation",
	}
	for name, body := range files {
		pipelineWrite(t, root, name, body)
	}
	writeSafetensor(t, filepath.Join(root, "transformer", "diffusion_pytorch_model.safetensors"), "weight")
	writeSafetensor(t, filepath.Join(root, "text_encoder", "model.safetensors"), "weight")
	return root
}

func pipelineWrite(t *testing.T, root, name, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestPipelineInventoryDeterministic(t *testing.T) {
	firstRoot, secondRoot := pipelineFixture(t), pipelineFixture(t)
	first, err := FromHFPath(firstRoot)
	if err != nil {
		t.Fatal(err)
	}
	second, err := FromHFPath(secondRoot)
	if err != nil {
		t.Fatal(err)
	}
	if first.Manifest.ID != second.Manifest.ID || first.TensorInventory.ID != second.TensorInventory.ID {
		t.Fatal("relocation or creation order changes inventory identity")
	}
	if len(first.TensorInventory.Tensors) != 2 {
		t.Fatalf("tensor facts: %+v", first.TensorInventory.Tensors)
	}
	for _, name := range []string{"text_encoder/model.safetensors/weight", "transformer/diffusion_pytorch_model.safetensors/weight"} {
		fact, ok := first.TensorInventory.Tensor(name)
		if !ok || fact.Storage != "u8" || fact.Bytes != 1 {
			t.Fatalf("tensor %s = %+v, found=%v", name, fact, ok)
		}
	}
	roles := map[string]artifact.ComponentRole{}
	for _, component := range first.Manifest.Components {
		roles[component.Name] = component.Role
	}
	for _, component := range first.Manifest.Components {
		if component.Name != "pipeline_moss_image.py" {
			continue
		}
		for _, descriptor := range first.Components {
			if descriptor.ID == component.Artifact && descriptor.MediaType != "text/x-python" {
				t.Fatalf("source metadata conflicts with retained Python reference: %s", descriptor.MediaType)
			}
		}
	}
	for name, want := range map[string]artifact.ComponentRole{"model_index.json": artifact.ComponentConfig, "tokenizer/vocab.json": artifact.ComponentVocabulary, "tokenizer/merges.txt": artifact.ComponentMerges, "pipeline_moss_image.py": artifact.ComponentCompanion, "scheduler/scheduler_config.json": artifact.ComponentConfig} {
		if roles[name] != want {
			t.Fatalf("%s role = %v, want %v", name, roles[name], want)
		}
	}
	pipelineWrite(t, secondRoot, "scheduler/scheduler_config.json", `{"_class_name":"MOSSImageScheduler","shift":1}`)
	changed, err := FromHFPath(secondRoot)
	if err != nil {
		t.Fatal(err)
	}
	if changed.Manifest.ID == first.Manifest.ID {
		t.Fatal("scheduler change retained source identity")
	}
}

func TestPipelineInventoryRefusesIncompleteOrEscapingComponents(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*testing.T, string)
	}{
		{"missing weights", func(t *testing.T, root string) {
			if err := os.Remove(filepath.Join(root, "transformer", "diffusion_pytorch_model.safetensors")); err != nil {
				t.Fatal(err)
			}
		}},
		{"missing directory", func(t *testing.T, root string) {
			pipelineWrite(t, root, "model_index.json", `{"_class_name":"P","missing":["module","Model"]}`)
		}},
		{"escape", func(t *testing.T, root string) {
			pipelineWrite(t, root, "model_index.json", `{"_class_name":"P","../outside":["module","Model"]}`)
		}},
		{"half absent", func(t *testing.T, root string) {
			pipelineWrite(t, root, "model_index.json", `{"_class_name":"P","transformer":[null,"Model"]}`)
		}},
		{"wrong arity", func(t *testing.T, root string) {
			pipelineWrite(t, root, "model_index.json", `{"_class_name":"P","transformer":["module","Model","extra"]}`)
		}},
		{"corrupt shard", func(t *testing.T, root string) {
			pipelineWrite(t, root, "transformer/diffusion_pytorch_model.safetensors", "bad")
		}},
		{"missing indexed tensor", func(t *testing.T, root string) {
			pipelineWrite(t, root, "transformer/diffusion_pytorch_model.safetensors.index.json", `{"weight_map":{"missing":"diffusion_pytorch_model.safetensors"}}`)
		}},
		{"duplicate tensor", func(t *testing.T, root string) {
			writeSafetensor(t, filepath.Join(root, "transformer", "second.safetensors"), "weight")
			pipelineWrite(t, root, "transformer/diffusion_pytorch_model.safetensors.index.json", `{"weight_map":{"weight":"diffusion_pytorch_model.safetensors","other":"second.safetensors"}}`)
		}},
		{"invalid config", func(t *testing.T, root string) { pipelineWrite(t, root, "transformer/config.json", "{} {}") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := pipelineFixture(t)
			tc.change(t, root)
			if _, err := FromHFPath(root); err == nil {
				t.Fatal("invalid pipeline accepted")
			}
		})
	}
}

func TestPipelineInventoryRetainsPolicyParameters(t *testing.T) {
	root := pipelineFixture(t)
	raw, err := os.ReadFile(filepath.Join(root, "model_index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var index map[string]any
	if err := json.Unmarshal(raw, &index); err != nil {
		t.Fatal(err)
	}
	index["_class_name"] = "Krea2Pipeline"
	index["transformer"] = []string{"diffusers", "Krea2Transformer2DModel"}
	index["text_encoder"] = []string{"transformers", "Qwen3VLModel"}
	index["scheduler"] = []string{"diffusers", "FlowMatchEulerDiscreteScheduler"}
	index["text_encoder_select_layers"] = []int{2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}
	index["is_distilled"] = true
	index["patch_size"] = 2
	encoded, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	pipelineWrite(t, root, "model_index.json", string(encoded))
	inventory, err := FromHFPath(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.TensorInventory.Tensors) != 2 {
		t.Fatal("policy metadata changed component enumeration")
	}
	index["text_encoder_select_layers"] = []int{2, 5}
	encoded, err = json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	pipelineWrite(t, root, "model_index.json", string(encoded))
	changed, err := FromHFPath(root)
	if err != nil {
		t.Fatal(err)
	}
	if changed.Manifest.ID == inventory.Manifest.ID {
		t.Fatal("layer selection was excluded from model identity")
	}
}
