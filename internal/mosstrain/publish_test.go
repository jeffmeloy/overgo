package mosstrain

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/peft"
	"overgo/internal/safetensors"
)

// TestPublishServedHoldsOneShardIndex publishes a sharded base, as the real
// checkpoint is. The merge writes its own index, so the base's beside it would
// leave a directory that refuses to open; the other components stay the base's.
func TestPublishServedHoldsOneShardIndex(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	base, served := filepath.Join(root, "base"), filepath.Join(root, "served")
	transformer := filepath.Join(base, transformerDirectory)
	if err := os.MkdirAll(filepath.Join(base, "vae"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(transformer, 0o755); err != nil {
		t.Fatal(err)
	}
	const module, last = "layers.0.proj.q", "final_linear_video.weight"
	weight := module + ".weight"
	first, second := "diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"
	if err := safetensors.Save(filepath.Join(transformer, first), map[string][]float32{weight: {1, 2, 3, 4}}, map[string][]int{weight: {2, 2}}, nil); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(transformer, second), map[string][]float32{last: {5}}, map[string][]int{last: {1}}, nil); err != nil {
		t.Fatal(err)
	}
	index, err := json.Marshal(map[string]any{"metadata": map[string]any{"total_size": 20}, "weight_map": map[string]string{weight: first, last: second}})
	if err != nil {
		t.Fatal(err)
	}
	for path, data := range map[string][]byte{
		filepath.Join(transformer, "diffusion_pytorch_model.safetensors.index.json"): index,
		filepath.Join(transformer, "config.json"):                                    []byte("{}"),
		filepath.Join(base, "vae", "config.json"):                                    []byte("{}"),
	} {
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	adapter := peft.Adapter{
		Config:  peft.Config{BaseModel: "base/model", Rank: 1, Alpha: 1},
		Modules: map[string]peft.Module{module: {Rank: 1, In: 2, Out: 2, Alpha: 1, Down: []float32{1, 0}, Up: []float32{1, 0}}},
	}
	if err := PublishServed(base, served, adapter); err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(filepath.Join(served, transformerDirectory))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	merged, err := safetensors.ReadF32(source.Tensors[weight])
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(merged, []float32{2, 2, 3, 4}) {
		t.Fatalf("merged weight %v", merged)
	}
	if _, err := os.Stat(filepath.Join(served, "vae", "config.json")); err != nil {
		t.Fatal(err)
	}
}

// TestAServedTargetIsRefusedBeforeTraining: a target that exists is refused and left as it was, and an absent one passes
// with its parent made and nothing left behind, so PublishServed then writes it. A base with nothing in it has nothing to
// serve from.
func TestAServedTargetIsRefusedBeforeTraining(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	base := filepath.Join(root, "base")
	if err := os.MkdirAll(filepath.Join(base, "vae"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(base, "vae", "config.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	existing := filepath.Join(root, "existing")
	if err := os.MkdirAll(existing, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := CheckServedTarget(base, existing); err == nil || !strings.Contains(err.Error(), "exists") {
		t.Fatalf("a target that exists = %v, want it refused", err)
	}
	target := filepath.Join(root, "new", "served")
	if err := CheckServedTarget(base, target); err != nil {
		t.Fatalf("an absent target = %v", err)
	}
	if _, err := os.Lstat(target); !os.IsNotExist(err) {
		t.Fatalf("the check left its probe at %s: %v", target, err)
	}
	if err := CheckServedTarget(base, target); err != nil {
		t.Fatalf("the check refused what it left: %v", err)
	}
	empty := filepath.Join(root, "empty")
	if err := os.MkdirAll(empty, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := CheckServedTarget(empty, filepath.Join(root, "other")); err == nil {
		t.Fatal("a base with no file passes")
	}
}
