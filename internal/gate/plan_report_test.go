package gate

import (
	"context"
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

func TestGatePlanReportNamesDependencyRelationships(t *testing.T) {
	t.Parallel()
	definitions := []automationcheck.Check{
		{Descriptor: automationcheck.Descriptor{
			Name: "prepare", Phase: runrecord.PhaseValidate,
			Triggers: []automationcheck.Fact{"owner:prepare"}, Inapplicable: "independent",
		}, Run: gateReportPass},
		{Descriptor: automationcheck.Descriptor{
			Name: "verify", Phase: runrecord.PhaseTest, Always: true, Dependencies: []string{"prepare"},
		}, Run: gateReportPass},
	}
	invocations, err := automationcheck.Plan(definitions, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	report := buildGatePlanReport(plannedPipeline{definitions: definitions, invocations: invocations})
	if report.Schema != 2 || len(report.Selected) != 2 ||
		len(report.Selected[0].RequiredBy) != 1 || report.Selected[0].RequiredBy[0] != "verify" ||
		report.Selected[0].Reason != "no impact producer proved this check independent" {
		t.Fatalf("gate plan report = %+v", report)
	}
}

func gateReportPass(context.Context, automationcheck.Invocation) (bool, string, error) {
	return false, "", nil
}
