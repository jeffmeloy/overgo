package plan

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// CompletedOutcomes returns the decoded outcome document of every completed
// reference whose completion snapshot recorded one; undecodable or absent
// outcomes are omitted, so a condition over them waits.
func (authority CompletionAuthority) CompletedOutcomes() map[string]map[string]any {
	outcomes := map[string]map[string]any{}
	for reference, evidence := range authority.completedReferences {
		if decoded, ok := decodeOutcomeDocument(json.RawMessage(evidence.outcome)); ok {
			outcomes[reference] = decoded
		}
	}
	return outcomes
}

// decodeOutcomeDocument: a JSON object with scalar fields; anything else is
// not a fact source.
func decodeOutcomeDocument(raw json.RawMessage) (map[string]any, bool) {
	if len(raw) == 0 {
		return nil, false
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil || decoded == nil {
		return nil, false
	}
	return decoded, true
}

// DocumentConditionFacts collects the facts the frontier evaluates: outcomes of
// completed parents, outcomes recorded on retained rows, and the disposition
// and reason of refused rows under the fields "disposition" and "reason".
func DocumentConditionFacts(d Plan, authority CompletionAuthority) ConditionFacts {
	facts := ConditionFacts{Outcomes: authority.CompletedOutcomes(), Events: map[string]bool{}}
	for _, item := range d.Items {
		for _, step := range item.Steps {
			reference := item.ID + "/" + step.ID
			if step.Refusal != nil {
				facts.Outcomes[reference] = map[string]any{"disposition": string(step.Refusal.Disposition), "reason": step.Refusal.Reason}
				continue
			}
			if decoded, ok := decodeOutcomeDocument(step.Outcome); ok {
				facts.Outcomes[reference] = decoded
			}
		}
	}
	return facts
}

// BlockedRow is one open row outside the frontier and the dependency states
// holding it.
type BlockedRow struct {
	Ref   Ref      `json:"ref"`
	Waits []string `json:"waits"`
}

// BlockedRows lists every open row the frontier excludes, with each unsatisfied
// dependency named by its state: open, its refusal, or absent without
// completion evidence.
func BlockedRows(d Plan, frontier []Ref, authority CompletionAuthority) []BlockedRow {
	ready := make(map[string]bool, len(frontier))
	for _, ref := range frontier {
		ready[ref.String()] = true
	}
	var blocked []BlockedRow
	for _, item := range d.Items {
		if item.Status != StatusOpen {
			continue
		}
		for _, step := range item.Steps {
			ref := Ref{Item: item.ID, Step: step.ID}
			if step.Status != StatusOpen || ready[ref.String()] {
				continue
			}
			blocked = append(blocked, BlockedRow{Ref: ref, Waits: unsatisfiedDependencies(d, step, authority)})
		}
	}
	return blocked
}

// unsatisfiedDependencies: state of each dependency that does not satisfy step.
func unsatisfiedDependencies(d Plan, step Step, authority CompletionAuthority) []string {
	var waits []string
	for _, reference := range step.DependsOn {
		itemID, stepID, _ := strings.Cut(reference, "/")
		dependency, present := exactPlanStep(d, itemID, stepID)
		switch {
		case present && dependency.Status == StatusOpen:
			waits = append(waits, reference+" open")
		case present && refused(dependency.Status) && !slices.Contains(step.AcceptsRefusal, reference):
			waits = append(waits, fmt.Sprintf("%s %s: %s", reference, dependency.Status, dependency.Refusal.Reason))
		case present && !refused(dependency.Status) && dependency.Status != StatusDone:
			waits = append(waits, reference+" "+dependency.Status)
		case !present && !authority.completed(reference):
			waits = append(waits, reference+" absent without completion evidence")
		}
	}
	return waits
}

// FormatBlocked renders each blocked row as one line naming what it waits on.
func FormatBlocked(rows []BlockedRow) string {
	lines := make([]string, 0, len(rows))
	for _, row := range rows {
		lines = append(lines, fmt.Sprintf("blocked: %s waits on %s", row.Ref, strings.Join(row.Waits, "; ")))
	}
	return strings.Join(lines, "\n")
}
