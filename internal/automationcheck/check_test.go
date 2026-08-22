package automationcheck

import (
	"context"
	"testing"

	"overgo/internal/runrecord"
)

func TestDescriptorRejectsIncompleteDefinitions(t *testing.T) {
	_, err := Plan([]Check{{Descriptor: Descriptor{Name: "incomplete", Phase: runrecord.PhaseValidate}, Run: pass}}, nil)
	if err == nil {
		t.Fatal("descriptor without applicability passed validation")
	}
}

func TestApplicabilitySelectsExactDerivedFacts(t *testing.T) {
	checks := []Check{
		{Descriptor: Descriptor{Name: "kernel", Phase: runrecord.PhaseValidate, Triggers: []Fact{"owner:kernel"}, Inapplicable: "other owner"}, Run: pass},
		{Descriptor: Descriptor{Name: "release", Phase: runrecord.PhaseValidate, Triggers: []Fact{"owner:release"}, Inapplicable: "other owner"}, Run: pass},
	}
	planned, err := Plan(checks, []Fact{"owner:release", "owner:release"})
	if err != nil {
		t.Fatal(err)
	}
	if len(planned) != 1 || planned[0].Check.Name != "release" || len(planned[0].Matched) != 1 {
		t.Fatalf("planned = %+v", planned)
	}
}

func TestPlanIncludesDependenciesInRegistrationOrder(t *testing.T) {
	checks := []Check{
		{Descriptor: Descriptor{Name: "inventory", Phase: runrecord.PhaseValidate, Always: true}, Run: pass},
		{Descriptor: Descriptor{Name: "device", Phase: runrecord.PhaseTest, Triggers: []Fact{"capability:cuda"}, Inapplicable: "no device impact", Dependencies: []string{"inventory"}, Resources: []Resource{{Name: "device", Exclusive: true}}}, Run: pass},
	}
	planned, err := Plan(checks, []Fact{"capability:cuda"})
	if err != nil {
		t.Fatal(err)
	}
	if len(planned) != 2 || planned[0].Check.Name != "inventory" || planned[1].Check.Name != "device" || !planned[1].ID.Valid() {
		t.Fatalf("planned = %+v", planned)
	}
}

func TestRunReturnsTypedFailureEvidence(t *testing.T) {
	checks := []Check{{Descriptor: Descriptor{Name: "device", Phase: runrecord.PhaseTest, Always: true}, Run: pass}}
	planned, err := Plan(checks, nil)
	if err != nil {
		t.Fatal(err)
	}
	planned[0].runner = func(context.Context, Invocation) (bool, string, error) {
		return false, "probe failed", runrecord.LaneError(runrecord.LaneUnavailable, "device unavailable")
	}
	evidence, err := Run(context.Background(), planned[0])
	if err == nil || evidence.Outcome != runrecord.LaneUnavailable || evidence.Detail != "probe failed" || !evidence.ID.Valid() || evidence.DurationNS == 0 {
		t.Fatalf("evidence = %+v, err = %v", evidence, err)
	}
}

func pass(context.Context, Invocation) (bool, string, error) { return false, "", nil }
