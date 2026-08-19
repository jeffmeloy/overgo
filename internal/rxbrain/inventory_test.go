package rxbrain

import (
	"os"
	"testing"
)

// TestRxBrainInventoryValidatesRealCheckpoint pins stage 2 of the port: the
// sharded weight map classifies into the mixture-of-transformers components,
// every decoder layer carries whole base-text and vision branches plus the
// shared attention layernorms, and the visual tower and embedding exist.
// The generation branch and flow adapters are counted but not required by
// the vision-QA scope.
func TestRxBrainInventoryValidatesRealCheckpoint(t *testing.T) {
	if _, err := os.Stat(realCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: %s absent; RxBrain inventory NOT verified", realCheckpoint)
	}
	config, err := Load(realCheckpoint)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := LoadInventory(realCheckpoint, config)
	if err != nil {
		t.Fatal(err)
	}
	// 32 layers x 9 branch tensors for text and vision; shared layernorms 2/layer.
	if inventory.VisionBranch < config.NumHiddenLayers*len(branchSuffixes) {
		t.Fatalf("vision branch %d below the %d whole-layer bound", inventory.VisionBranch, config.NumHiddenLayers*len(branchSuffixes))
	}
	if inventory.Shared < config.NumHiddenLayers*2 {
		t.Fatalf("shared layernorms %d below %d", inventory.Shared, config.NumHiddenLayers*2)
	}
	if inventory.VisualTower == 0 || inventory.TextBranch == 0 {
		t.Fatalf("components missing: %+v", inventory)
	}
	t.Logf("inventory: text=%d vision=%d generation=%d visual_tower=%d flow=%d shared=%d tensors=%d",
		inventory.TextBranch, inventory.VisionBranch, inventory.GenerationBranch,
		inventory.VisualTower, inventory.FlowAdapters, inventory.Shared, len(inventory.TensorShard))
}
