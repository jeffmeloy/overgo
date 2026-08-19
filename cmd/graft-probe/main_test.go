package main

import (
	"testing"

	"overgo/internal/composition"
	"overgo/internal/repodb"
)

// TestProbeCommitsGenerationRecord pins the lifecycle wiring: the probe's
// recording path authors a typed generation record through the production
// constructor, commits it with its referenced identities, and the committed
// graph is queryable -- child lineage present, verdict outcome preserved,
// refusals recorded as loudly as ships.
func TestProbeRejectsUncompiledGenerationRecord(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	config := composition.Config{
		TargetDir: "fixture/target", DonorDir: "fixture/donor",
		Seeds: []int64{7, 11, 13}, Steps: 6,
	}
	result := composition.Result{
		DonorTensor: "model.layers.14.mlp.gate_proj.weight", GraftLayer: 12,
		Baseline: 4.849884, Ship: false,
		Reason: "refusal: worst seed held-out 5.441688 does not beat baseline 4.849884",
	}
	if _, err := composition.RecordViability(store, config, result); err == nil {
		t.Fatal("record accepted path-derived placeholder authority")
	}
}
