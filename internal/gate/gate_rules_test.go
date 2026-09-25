package gate

import (
	"slices"
	"testing"

	"overgo/internal/automationcheck"
)

// TestGateRulesDeclareEveryPipelineCheckOnce holds the pipeline to one rule
// table: every check the gate builds has exactly one rule, in the table's
// order, and its waits, needs and store use come from that rule alone.
func TestGateRulesDeclareEveryPipelineCheckOnce(t *testing.T) {
	t.Parallel()
	g := &gateContext{repo: t.TempDir(), paths: []string{"internal/gate/gate.go"}}
	checks := g.pipelineChecks()
	names := make([]string, 0, len(checks))
	for _, check := range checks {
		names = append(names, check.Descriptor.Name)
	}
	if want := gateRuleNames(func(gateRule) bool { return true }); !slices.Equal(names, want) {
		t.Fatalf("pipeline checks = %v, want the rule table's order %v", names, want)
	}
	for _, check := range checks {
		rule := gateRuleNamed(check.Descriptor.Name)
		after := rule.after
		if rule.afterStatic {
			after = validateWave
		}
		if !slices.Equal(check.Descriptor.Dependencies, after) {
			t.Fatalf("%s waits for %v, its rule says %v", rule.name, check.Descriptor.Dependencies, after)
		}
		if rule.needs != (automationcheck.Requirements{}) && check.Descriptor.Requirements != rule.needs {
			t.Fatalf("%s needs %+v, its rule says %+v", rule.name, check.Descriptor.Requirements, rule.needs)
		}
		if rule.store != nil && !slices.Equal(check.Descriptor.Resources, []automationcheck.Resource{*rule.store}) {
			t.Fatalf("%s holds %v, its rule says %v", rule.name, check.Descriptor.Resources, *rule.store)
		}
	}
	if !slices.Equal(deferredLaneChecks, []string{testDeviceCheckName, "device", automationcheck.WebUICheckName, automationcheck.ModelJourneyCheckName}) {
		t.Fatalf("deferred lanes = %v", deferredLaneChecks)
	}
}
