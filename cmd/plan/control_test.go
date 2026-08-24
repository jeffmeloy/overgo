package main

import (
	"context"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

func TestOverrideAndContainmentEvidence(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, event := range []plan.ControlEvent{
		{Kind: "override", Lane: "item/step", ReasonCode: "forced-advance", Detail: "owner accepted fixture", CodeCommit: fixtureCommit},
		{Kind: "containment", Lane: "gpu-0", ReasonCode: "device-instability", Detail: "ECC fault", CodeCommit: fixtureCommit},
	} {
		recorded, err := plan.RecordControlEvent(context.Background(), store, event)
		if err != nil {
			t.Fatal(err)
		}
		parsed, ok, err := plan.ReadControlEvent(context.Background(), store, recorded.ID)
		if err != nil || !ok || parsed.Kind != event.Kind || parsed.Lane != event.Lane {
			t.Fatalf("control event = (%+v, %v, %v)", parsed, ok, err)
		}
	}
	if _, err := plan.RecordControlEvent(context.Background(), store, plan.ControlEvent{
		Kind: "containment", Lane: "all", ReasonCode: "pause", Detail: "untyped", CodeCommit: fixtureCommit,
	}); err == nil {
		t.Fatal("untyped containment reason accepted")
	}
}

const fixtureCommit = "0123456789abcdef0123456789abcdef01234567"
