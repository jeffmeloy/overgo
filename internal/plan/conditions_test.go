package plan

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

func conditionalPlanFixture() Plan {
	return Plan{Items: []Item{
		{ID: "guard", Status: StatusOpen, Steps: []Step{{
			ID: "small", Title: "Small guard", Status: StatusOpen,
			Verify: "go test ./internal/longform -run '^TestGuard$' -count=1",
		}}},
		{ID: "bench", Status: StatusOpen, Steps: []Step{{
			ID: "large", Title: "Large pass", Status: StatusOpen,
			Verify:    "go test ./cmd/benchmark -run '^TestLarge$' -count=1",
			DependsOn: []string{"guard/small"},
			Conditions: &StepConditions{
				SkipIf:   []OutcomeCondition{{Parent: "guard/small", Field: "decode_rate", Operator: "lt", Value: json.RawMessage(`160`)}},
				CancelIf: []OutcomeCondition{{Parent: "guard/small", Field: "outcome", Operator: "eq", Value: json.RawMessage(`"failed"`)}},
				WaitFor:  []WaitCondition{{Event: "device-idle"}, {Elapsed: "5m"}},
			},
		}}},
	}}
}

// TestConditionalRowsOverRecordedOutcomes pins: declaration round trip;
// refused declarations; cancel before skip before wait before proceed;
// absent facts wait; typed comparison per JSON kind; determinism.
func TestConditionalRowsOverRecordedOutcomes(t *testing.T) {
	document := conditionalPlanFixture()
	if err := Validate(document); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(data)
	if err != nil || !reflect.DeepEqual(parsed, document) {
		t.Fatalf("conditions round trip = %+v, %v", parsed, err)
	}
	for _, test := range []struct {
		name string
		edit func(*StepConditions)
		want string
	}{
		{"empty", func(c *StepConditions) { *c = StepConditions{} }, "declare nothing"},
		{"undeclared parent", func(c *StepConditions) { c.SkipIf[0].Parent = "other/row" }, "not a declared dependency"},
		{"upper field", func(c *StepConditions) { c.SkipIf[0].Field = "Rate" }, "lower identifier"},
		{"unknown operator", func(c *StepConditions) { c.SkipIf[0].Operator = "matches" }, "operator"},
		{"object literal", func(c *StepConditions) { c.SkipIf[0].Value = json.RawMessage(`{"a":1}`) }, "not a number, string or bool"},
		{"wait both", func(c *StepConditions) { c.WaitFor[0].Elapsed = "1s" }, "exactly one"},
		{"wait neither", func(c *StepConditions) { c.WaitFor[0].Event = "" }, "exactly one"},
		{"bad elapsed", func(c *StepConditions) { c.WaitFor[1].Elapsed = "soon" }, "positive duration"},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := conditionalPlanFixture()
			test.edit(candidate.Items[1].Steps[0].Conditions)
			err := Validate(candidate)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validate %s = %v, want %q", test.name, err, test.want)
			}
		})
	}

	conditions := document.Items[1].Steps[0].Conditions
	healthy := map[string]any{"decode_rate": 243.0, "outcome": "succeeded"}
	facts := func(outcome map[string]any, events map[string]bool, ready time.Duration) ConditionFacts {
		return ConditionFacts{Outcomes: map[string]map[string]any{"guard/small": outcome}, Events: events, Ready: ready}
	}
	idle := map[string]bool{"device-idle": true}
	for _, test := range []struct {
		name   string
		facts  ConditionFacts
		want   Disposition
		reason string
	}{
		{"proceed", facts(healthy, idle, 5*time.Minute), DispositionProceed, ""},
		{"cancel on failed outcome", facts(map[string]any{"decode_rate": 10.0, "outcome": "failed"}, idle, time.Hour), DispositionCancel, "guard/small.outcome eq \"failed\""},
		{"skip on low rate", facts(map[string]any{"decode_rate": 140.9, "outcome": "succeeded"}, idle, time.Hour), DispositionSkip, "guard/small.decode_rate lt 160"},
		{"wait for event", facts(healthy, nil, time.Hour), DispositionWait, "event device-idle not recorded"},
		{"wait for elapsed", facts(healthy, idle, time.Minute), DispositionWait, "elapsed 5m not reached"},
		{"absent outcome waits", ConditionFacts{Events: idle, Ready: time.Hour}, DispositionWait, "outcome of guard/small not recorded"},
		{"absent field waits", facts(map[string]any{"outcome": "succeeded"}, idle, time.Hour), DispositionWait, "lacks field decode_rate"},
		{"kind mismatch waits", facts(map[string]any{"decode_rate": "fast", "outcome": "succeeded"}, idle, time.Hour), DispositionWait, "not comparable"},
	} {
		t.Run(test.name, func(t *testing.T) {
			for repeat := range 2 {
				got, reason := conditions.Evaluate(test.facts)
				if got != test.want || !strings.Contains(reason, test.reason) {
					t.Fatalf("evaluate repeat %d = %q %q, want %q %q", repeat, got, reason, test.want, test.reason)
				}
			}
		})
	}
	var none *StepConditions
	if got, reason := none.Evaluate(ConditionFacts{}); got != DispositionProceed || reason != "" {
		t.Fatalf("unconditional row = %q %q, want proceed", got, reason)
	}
	boolean := OutcomeCondition{Parent: "guard/small", Field: "complete", Operator: "eq", Value: json.RawMessage(`true`)}
	if held, _ := boolean.holds(facts(map[string]any{"complete": true}, nil, 0)); !held {
		t.Fatal("bool equality did not hold")
	}
	if held, _ := boolean.holds(facts(map[string]any{"complete": false}, nil, 0)); held {
		t.Fatal("bool inequality held")
	}
}
