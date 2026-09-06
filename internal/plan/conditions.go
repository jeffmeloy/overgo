package plan

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/textcheck"
)

// StepConditions declares typed dispatch conditions over recorded parent
// outcomes and recorded events; the frontier evaluates them, never an
// expression language.
type StepConditions struct {
	SkipIf   []OutcomeCondition `json:"skip_if,omitempty"`
	CancelIf []OutcomeCondition `json:"cancel_if,omitempty"`
	WaitFor  []WaitCondition    `json:"wait_for,omitempty"`
}

// OutcomeCondition compares Field of Parent's recorded outcome with Operator
// against the literal Value (JSON number, string or bool); Parent must be a
// declared dependency.
type OutcomeCondition struct {
	Parent   string          `json:"parent"`
	Field    string          `json:"field"`
	Operator string          `json:"operator"`
	Value    json.RawMessage `json:"value"`
}

// WaitCondition holds a row until recorded Event exists or Elapsed has passed
// since the row became ready; exactly one of the two is declared.
type WaitCondition struct {
	Event   string `json:"event,omitzero"`
	Elapsed string `json:"elapsed,omitzero"`
}

// Disposition is the closed frontier verdict for a row whose dependencies hold.
type Disposition string

const (
	// DispositionProceed means every condition holds or none is declared.
	DispositionProceed Disposition = "proceed"
	// DispositionSkip means a skip-if predicate holds.
	DispositionSkip Disposition = "skip"
	// DispositionCancel means a cancel-if predicate holds.
	DispositionCancel Disposition = "cancel"
	// DispositionWait means a fact is absent or a wait-for is unmet.
	DispositionWait Disposition = "wait"
)

// ConditionFacts holds the recorded facts the evaluator reads: Outcomes keyed
// by parent ref (decoded outcome document), Events by recorded event name,
// Ready = elapsed since the row's dependencies were satisfied.
type ConditionFacts struct {
	Outcomes map[string]map[string]any
	Events   map[string]bool
	Ready    time.Duration
}

var conditionOperators = []string{"eq", "ne", "lt", "le", "gt", "ge"}

func validateStepConditions(step Step) error {
	conditions := step.Conditions
	if conditions == nil {
		return nil
	}
	if len(conditions.SkipIf)+len(conditions.CancelIf)+len(conditions.WaitFor) == 0 {
		return errors.New("conditions declare nothing")
	}
	for _, condition := range slices.Concat(conditions.SkipIf, conditions.CancelIf) {
		if !slices.Contains(step.DependsOn, condition.Parent) {
			return fmt.Errorf("condition parent %q is not a declared dependency", condition.Parent)
		}
		if !textcheck.LowerIdentifier(condition.Field, automationRoleMaxBytes) {
			return fmt.Errorf("condition field %q is not a lower identifier", condition.Field)
		}
		if !slices.Contains(conditionOperators, condition.Operator) {
			return fmt.Errorf("condition operator %q is not one of %v", condition.Operator, conditionOperators)
		}
		if _, err := decodeConditionLiteral(condition.Value); err != nil {
			return err
		}
	}
	for _, wait := range conditions.WaitFor {
		if (wait.Event == "") == (wait.Elapsed == "") {
			return errors.New("wait condition declares exactly one of event or elapsed")
		}
		if wait.Event != "" && !validAutomationText(wait.Event) {
			return fmt.Errorf("wait event %q is invalid", wait.Event)
		}
		if wait.Elapsed != "" {
			if elapsed, err := time.ParseDuration(wait.Elapsed); err != nil || elapsed <= 0 {
				return fmt.Errorf("wait elapsed %q is not a positive duration", wait.Elapsed)
			}
		}
	}
	return nil
}

// decodeConditionLiteral: JSON number, string or bool; anything else refused.
func decodeConditionLiteral(raw json.RawMessage) (any, error) {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("condition value %s is not JSON: %w", string(raw), err)
	}
	switch value.(type) {
	case float64, string, bool:
		return value, nil
	}
	return nil, fmt.Errorf("condition value %s is not a number, string or bool", string(raw))
}

// Evaluate orders cancel -> skip -> wait -> proceed; missing outcome, field or
// event -> wait with that reason (absence never satisfies a predicate).
func (c *StepConditions) Evaluate(facts ConditionFacts) (Disposition, string) {
	if c == nil {
		return DispositionProceed, ""
	}
	for _, condition := range c.CancelIf {
		if held, reason := condition.holds(facts); held {
			return DispositionCancel, reason
		} else if reason != "" {
			return DispositionWait, reason
		}
	}
	for _, condition := range c.SkipIf {
		if held, reason := condition.holds(facts); held {
			return DispositionSkip, reason
		} else if reason != "" {
			return DispositionWait, reason
		}
	}
	for _, wait := range c.WaitFor {
		if wait.Event != "" && !facts.Events[wait.Event] {
			return DispositionWait, "event " + wait.Event + " not recorded"
		}
		if wait.Elapsed != "" {
			elapsed, _ := time.ParseDuration(wait.Elapsed)
			if facts.Ready < elapsed {
				return DispositionWait, "elapsed " + wait.Elapsed + " not reached"
			}
		}
	}
	return DispositionProceed, ""
}

// holds: (true, description) when the predicate holds; (false, "") when it
// is decidable and false; (false, reason) when the fact is absent.
func (o OutcomeCondition) holds(facts ConditionFacts) (bool, string) {
	outcome, found := facts.Outcomes[o.Parent]
	if !found {
		return false, "outcome of " + o.Parent + " not recorded"
	}
	actual, found := outcome[o.Field]
	if !found {
		return false, "outcome of " + o.Parent + " lacks field " + o.Field
	}
	literal, err := decodeConditionLiteral(o.Value)
	if err != nil {
		return false, err.Error()
	}
	order, comparable := compareConditionValues(actual, literal)
	if !comparable {
		return false, fmt.Sprintf("outcome field %s of %s is not comparable with %s", o.Field, o.Parent, string(o.Value))
	}
	held := false
	switch o.Operator {
	case "eq":
		held = order == 0
	case "ne":
		held = order != 0
	case "lt":
		held = order < 0
	case "le":
		held = order <= 0
	case "gt":
		held = order > 0
	case "ge":
		held = order >= 0
	}
	if held {
		return true, fmt.Sprintf("%s.%s %s %s", o.Parent, o.Field, o.Operator, string(o.Value))
	}
	return false, ""
}

// compareConditionValues: three-way order within one JSON kind only; bools
// order false before true; a kind mismatch is incomparable.
func compareConditionValues(actual, literal any) (int, bool) {
	switch want := literal.(type) {
	case float64:
		if got, ok := actual.(float64); ok {
			return cmp.Compare(got, want), true
		}
	case string:
		if got, ok := actual.(string); ok {
			return strings.Compare(got, want), true
		}
	case bool:
		if got, ok := actual.(bool); ok {
			return strings.Compare(strconv.FormatBool(got), strconv.FormatBool(want)), true
		}
	}
	var incomparable int
	return incomparable, false
}

// RowDisposition pairs one ready row with its condition verdict and reason.
type RowDisposition struct {
	Ref         Ref         `json:"ref"`
	Disposition Disposition `json:"disposition"`
	Reason      string      `json:"reason,omitzero"`
}

// Dispositions evaluates each frontier row's conditions against facts in
// frontier order; rows without conditions proceed.
func Dispositions(d Plan, frontier []Ref, facts ConditionFacts) ([]RowDisposition, error) {
	dispositions := make([]RowDisposition, 0, len(frontier))
	for _, ref := range frontier {
		step, retained := exactPlanStep(d, ref.Item, ref.Step)
		if !retained {
			return nil, fmt.Errorf("plan: frontier row %s is absent from the plan", ref)
		}
		disposition, reason := step.Conditions.Evaluate(facts)
		dispositions = append(dispositions, RowDisposition{Ref: ref, Disposition: disposition, Reason: reason})
	}
	return dispositions, nil
}

// FormatDispositions renders every non-proceeding row as one line.
func FormatDispositions(rows []RowDisposition) string {
	lines := make([]string, 0, len(rows))
	for _, row := range rows {
		if row.Disposition == DispositionProceed {
			continue
		}
		lines = append(lines, fmt.Sprintf("condition: %s %s %s", row.Ref, row.Disposition, row.Reason))
	}
	return strings.Join(lines, "\n")
}
