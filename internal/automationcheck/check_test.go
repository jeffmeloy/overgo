package automationcheck

import (
	"context"
	"testing"

	"overgo/internal/runrecord"
)

func TestDescriptorRejectsIncompleteDefinitions(t *testing.T) {
	_, err := Plan([]Check{{Descriptor: Descriptor{Name: "incomplete", Phase: runrecord.PhaseValidate}, Run: pass}}, Impact{})
	if err == nil {
		t.Fatal("descriptor without applicability passed validation")
	}
}

func TestApplicabilitySelectsExactDerivedFacts(t *testing.T) {
	checks := []Check{
		{Descriptor: Descriptor{Name: "kernel", Phase: runrecord.PhaseValidate, Triggers: []Fact{"owner:kernel"}, Inapplicable: "other owner"}, Run: pass},
		{Descriptor: Descriptor{Name: "release", Phase: runrecord.PhaseValidate, Triggers: []Fact{"owner:release"}, Inapplicable: "other owner"}, Run: pass},
	}
	planned, err := Plan(checks, Impact{Facts: []Fact{"owner:release", "owner:release"}, Exclusions: []Exclusion{{Check: "kernel", Reason: "release-only change"}}})
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
	planned, err := Plan(checks, Impact{Facts: []Fact{"capability:cuda"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(planned) != 2 || planned[0].Check.Name != "inventory" || planned[1].Check.Name != "device" || !planned[1].ID.Valid() {
		t.Fatalf("planned = %+v", planned)
	}
}

func TestRunReturnsTypedFailureEvidence(t *testing.T) {
	checks := []Check{{Descriptor: Descriptor{Name: "device", Phase: runrecord.PhaseTest, Always: true}, Run: pass}}
	planned, err := Plan(checks, Impact{})
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

func TestImpactDefaultsToRun(t *testing.T) {
	check := Check{Descriptor: Descriptor{
		Name: "unknown", Phase: runrecord.PhaseValidate,
		Triggers: []Fact{"owner:known"}, Inapplicable: "proven independent",
	}, Run: pass}
	planned, err := Plan([]Check{check}, Impact{})
	if err != nil || len(planned) != 1 || planned[0].Check.Name != "unknown" {
		t.Fatalf("unknown impact plan = %+v, %v", planned, err)
	}
}

func TestImpactRequiresReasonedExclusion(t *testing.T) {
	check := Check{Descriptor: Descriptor{
		Name: "owned", Phase: runrecord.PhaseValidate,
		Triggers: []Fact{"owner:owned"}, Inapplicable: "proven independent",
	}, Run: pass}
	if _, err := Plan([]Check{check}, Impact{Exclusions: []Exclusion{{Check: "owned"}}}); err == nil {
		t.Fatal("reasonless exclusion accepted")
	}
	if _, err := Plan([]Check{check}, Impact{Exclusions: []Exclusion{{Check: "unknown", Reason: "typo"}}}); err == nil {
		t.Fatal("unknown exclusion target accepted")
	}
	planned, err := Plan([]Check{check}, Impact{Exclusions: []Exclusion{{Check: "owned", Reason: "disjoint symbol closure"}}})
	if err != nil || len(planned) != 0 {
		t.Fatalf("reasoned exclusion plan = %+v, %v", planned, err)
	}
}

func TestImpactTriggeredCheck(t *testing.T) {
	check := Check{Descriptor: Descriptor{
		Name: "owned", Phase: runrecord.PhaseValidate,
		Triggers: []Fact{"owner:owned"}, Inapplicable: "proven independent",
	}, Run: pass}
	planned, err := Plan([]Check{check}, Impact{Facts: []Fact{"owner:owned"}})
	if err != nil || len(planned) != 1 || len(planned[0].Matched) != 1 {
		t.Fatalf("triggered impact plan = %+v, %v", planned, err)
	}
	if _, err := Plan([]Check{check}, Impact{
		Facts: []Fact{"owner:owned"}, Exclusions: []Exclusion{{Check: "owned", Reason: "contradiction"}},
	}); err == nil {
		t.Fatal("contradictory impact accepted")
	}
}
