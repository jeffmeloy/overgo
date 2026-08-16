package main

import (
	"context"
	"testing"

	"overgo/internal/composition"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

// TestProbeCommitsGenerationRecord pins the lifecycle wiring: the probe's
// recording path authors a typed generation record through the production
// constructor, commits it with its referenced identities, and the committed
// graph is queryable -- child lineage present, verdict outcome preserved,
// refusals recorded as loudly as ships.
func TestProbeCommitsGenerationRecord(t *testing.T) {
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
	record, err := composition.RecordViability(store, config, result)
	if err != nil {
		t.Fatal(err)
	}
	if record.Outcome != runrecord.OutcomeFailed || len(record.Seeds) != 3 {
		t.Fatalf("record = outcome %s seeds %d", record.Outcome, len(record.Seeds))
	}
	parents, err := store.Parents(context.Background(), record.Child)
	if err != nil || len(parents) == 0 {
		t.Fatalf("child lineage not queryable: (%v, %v)", parents, err)
	}
	content, ok, err := store.Content(context.Background(), record.ID)
	if err != nil || !ok {
		t.Fatalf("record content absent: (%v, %v)", ok, err)
	}
	parsed, err := runrecord.ParseGenerationRecord(content.Data)
	if err != nil || parsed.ID != record.ID || parsed.Decision != record.Decision {
		t.Fatalf("round trip = (%+v, %v)", parsed, err)
	}
	ship := result
	ship.Ship = true
	ship.Reason = "envelope separation: worst seed beats baseline"
	shipped, err := composition.RecordViability(store, config, ship)
	if err != nil || shipped.Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("ship record = (%s, %v)", shipped.Outcome, err)
	}
}
