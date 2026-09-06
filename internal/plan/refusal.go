package plan

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

const (
	// StatusSkipped retains a row a skip disposition resolved; the refusal
	// record carries the reason.
	StatusSkipped = "skipped"
	// StatusCancelled retains a row a cancel disposition resolved.
	StatusCancelled = "cancelled"
)

// RowRefusal is the typed record of a skipped or cancelled row; a dependent
// resolves against it only when it declares that it accepts the refusal.
type RowRefusal struct {
	Disposition Disposition `json:"disposition"`
	Reason      string      `json:"reason"`
	Recorded    string      `json:"recorded"`
}

// refusalStatus: the retained status of a disposition; "" for any other.
func refusalStatus(disposition Disposition) string {
	switch disposition {
	case DispositionSkip:
		return StatusSkipped
	case DispositionCancel:
		return StatusCancelled
	}
	return ""
}

// refused reports whether a status is a retained refusal.
func refused(status string) bool {
	return status == StatusSkipped || status == StatusCancelled
}

// Refuse records a skip or cancel disposition with its reason on an open
// row; the row stays in the plan as the durable refusal record.
func Refuse(d Plan, itemID, stepID string, disposition Disposition, reason string, now time.Time) (Plan, error) {
	status := refusalStatus(disposition)
	reason = strings.TrimSpace(reason)
	if status == "" {
		return Plan{}, fmt.Errorf("plan: disposition %q is not a refusal", disposition)
	}
	if reason == "" || !validAutomationDetail(reason) {
		return Plan{}, errors.New("plan: a refusal requires a recorded reason")
	}
	d.Items = slices.Clone(d.Items)
	for itemIndex := range d.Items {
		if d.Items[itemIndex].ID != itemID {
			continue
		}
		d.Items[itemIndex].Steps = slices.Clone(d.Items[itemIndex].Steps)
		for stepIndex := range d.Items[itemIndex].Steps {
			step := &d.Items[itemIndex].Steps[stepIndex]
			if step.ID != stepID {
				continue
			}
			if step.Status != StatusOpen {
				return Plan{}, fmt.Errorf("plan: step %q in %q is not open", stepID, itemID)
			}
			step.Status = status
			step.Refusal = &RowRefusal{Disposition: disposition, Reason: reason, Recorded: now.UTC().Format(time.RFC3339)}
			return d, Validate(d)
		}
		return Plan{}, fmt.Errorf("plan: step %q not found in %q", stepID, itemID)
	}
	return Plan{}, fmt.Errorf("plan: item %q not found", itemID)
}

// validateRefusal: a refused status carries a matching record and nothing
// else does; accepted refusals name declared dependencies only.
func validateRefusal(step Step) error {
	switch {
	case refused(step.Status) && (step.Refusal == nil || refusalStatus(step.Refusal.Disposition) != step.Status):
		return fmt.Errorf("status %q lacks a matching refusal record", step.Status)
	case !refused(step.Status) && step.Refusal != nil:
		return fmt.Errorf("status %q carries a refusal record", step.Status)
	case step.Refusal != nil && (strings.TrimSpace(step.Refusal.Reason) == "" || step.Refusal.Recorded == ""):
		return errors.New("refusal record lacks a reason or a time")
	}
	for _, reference := range step.AcceptsRefusal {
		if !slices.Contains(step.DependsOn, reference) {
			return fmt.Errorf("accepts_refusal %q is not a declared dependency", reference)
		}
	}
	return nil
}

// dependencyRefusal: the refusal record of a present dependency, if refused.
func dependencyRefusal(d Plan, reference string) (RowRefusal, bool) {
	itemID, stepID, _ := strings.Cut(reference, "/")
	for _, item := range d.Items {
		if item.ID != itemID {
			continue
		}
		for _, candidate := range item.Steps {
			if candidate.ID == stepID && refused(candidate.Status) && candidate.Refusal != nil {
				return *candidate.Refusal, true
			}
		}
	}
	return RowRefusal{}, false
}

// RefusedDependencies lists the dependencies of step that were refused and that
// step does not accept; each line names the disposition and reason.
func RefusedDependencies(d Plan, step Step) []string {
	var lines []string
	for _, reference := range step.DependsOn {
		refusal, found := dependencyRefusal(d, reference)
		if found && !slices.Contains(step.AcceptsRefusal, reference) {
			lines = append(lines, fmt.Sprintf("%s %s: %s", reference, refusal.Disposition, refusal.Reason))
		}
	}
	return lines
}
