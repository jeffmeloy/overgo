// Package plan owns campaign state and dispatch order.
package plan

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const (
	automationRoleMaxBytes = 2048
	// AutomationRoleEnvironment is the shared lane-role input for dispatchers.
	AutomationRoleEnvironment = "OVERGO_AUTOMATION_ROLE"
	// UnassignedRole selects work without an explicit owner.
	UnassignedRole = "unassigned"
)

// Path is the campaign plan, relative to the repo root.
const Path = "docs/plan.json"

const (
	// StatusOpen permits dispatch.
	StatusOpen = "open"
	// StatusDone retains completed work.
	StatusDone = "done"
)

// Step is one action within an item. Verify is a shell command (run via sh -c)
// that exits 0 iff the step's acceptance holds; it is required before the step
// may be advanced, so "done" is machine-checked rather than self-declared.
type Step struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Status string `json:"status"`
	Verify string `json:"verify,omitempty"`
	// Campaign detail carried from the merged design record (owner directive
	// 2026-08-16: one plan document). Rationale says why the row exists;
	// DependsOn declares ordering the queue must respect; Capabilities name
	// the discipline contracts the row exercises; Outcome records the
	// measured verdict verbatim once the row has run.
	Rationale    string          `json:"rationale,omitempty"`
	DependsOn    []string        `json:"depends_on,omitempty"`
	Capabilities []string        `json:"capabilities,omitempty"`
	Outcome      json.RawMessage `json:"outcome,omitempty"`
}

// Item is one rung of the ladder.
type Item struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Owner  string `json:"owner,omitempty"`
	Status string `json:"status"`
	Steps  []Step `json:"steps"`
}

// Plan is the whole campaign surface.
type Plan struct {
	Campaign string       `json:"campaign"`
	Doctrine string       `json:"doctrine"`
	Census   *artifact.ID `json:"census_evidence,omitempty"`
	Items    []Item       `json:"items"`
}

// Load reads the plan from path (Path when empty).
func Load(path string) (Plan, error) {
	if path == "" {
		path = Path
	}
	var d Plan
	if err := jsonfile.DecodeStrict(filepath.FromSlash(path), &d); err != nil {
		return Plan{}, fmt.Errorf("parse %s: %w", path, err)
	}
	return d, nil
}

// Parse decodes an in-memory plan document.
func Parse(data []byte) (Plan, error) {
	var document Plan
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return Plan{}, err
	}
	return document, Validate(document)
}

// Save writes the plan back to path (Path when empty).
func Save(path string, d Plan) error {
	if err := Validate(d); err != nil {
		return err
	}
	if path == "" {
		path = Path
	}
	return jsonfile.Write(filepath.FromSlash(path), d, 0o644)
}

// Validate checks retained plan state.
func Validate(d Plan) error {
	if d.Census != nil && (!d.Census.Valid() || d.Census.Kind() != artifact.KindEvidence) {
		return errors.New("plan: census authority is not evidence")
	}
	items := map[string]bool{}
	for _, item := range d.Items {
		if item.ID == "" || items[item.ID] {
			return fmt.Errorf("plan item id %q is empty or duplicated", item.ID)
		}
		items[item.ID] = true
		if item.Owner != "" && !validAutomationText(item.Owner) {
			return fmt.Errorf("plan item %s has invalid owner", item.ID)
		}
		if !validStatus(item.Status) {
			return fmt.Errorf("plan item %s has invalid status %q", item.ID, item.Status)
		}
		steps := map[string]bool{}
		for _, step := range item.Steps {
			if step.ID == "" || steps[step.ID] {
				return fmt.Errorf("plan step %s/%s is empty or duplicated", item.ID, step.ID)
			}
			steps[step.ID] = true
			if !validStatus(step.Status) {
				return fmt.Errorf("plan step %s/%s has invalid status %q", item.ID, step.ID, step.Status)
			}
			if step.Status == StatusOpen && strings.TrimSpace(step.Verify) == "" {
				return fmt.Errorf("plan step %s/%s is open without a verifier", item.ID, step.ID)
			}
		}
		if item.Status == StatusDone && slices.ContainsFunc(item.Steps, func(step Step) bool { return step.Status != StatusDone }) {
			return fmt.Errorf("plan item %s is done with unfinished steps", item.ID)
		}
	}
	return validateDependencies(d)
}

// validateDependencies refuses dangling depends_on references and
// dependency cycles: a reference that names nothing can never satisfy,
// and a cycle deadlocks dispatch silently. References name an item or
// an item/step pair.
func validateDependencies(d Plan) error {
	itemIndex := map[string]Item{}
	for _, item := range d.Items {
		itemIndex[item.ID] = item
	}
	edges := map[string][]string{}
	for _, item := range d.Items {
		for _, step := range item.Steps {
			for _, reference := range step.DependsOn {
				target, targetStep, hasStep := strings.Cut(reference, "/")
				// A reference to an ABSENT item is a completed dependency
				// (completion removes rows), so only present targets join
				// the cycle graph.
				if _, present := itemIndex[target]; !present {
					continue
				}
				if target != item.ID {
					edges[item.ID] = append(edges[item.ID], target)
					continue
				}
				// A same-item reference must name a DIFFERENT step: a step
				// depending on its own item (or itself) can never satisfy
				// and would sit permanently undispatchable.
				if !hasStep || targetStep == step.ID {
					return fmt.Errorf(
						"plan step %s/%s: depends_on %q can never satisfy: a step cannot depend on its own item or itself",
						item.ID, step.ID, reference)
				}
				edges[item.ID+"/"+step.ID] = append(edges[item.ID+"/"+step.ID], item.ID+"/"+targetStep)
			}
		}
	}
	visiting, settled := map[string]bool{}, map[string]bool{}
	var walk func(string) error
	walk = func(id string) error {
		if visiting[id] {
			return fmt.Errorf("plan dependency cycle through item %s", id)
		}
		if settled[id] {
			return nil
		}
		visiting[id] = true
		for _, next := range edges[id] {
			if err := walk(next); err != nil {
				return err
			}
		}
		visiting[id], settled[id] = false, true
		return nil
	}
	for id := range edges {
		if err := walk(id); err != nil {
			return err
		}
	}
	return nil
}

// dependenciesSatisfied reports whether every depends_on reference is
// complete. Completion REMOVES rows, so a reference naming an item or
// step absent from the plan is satisfied: it was completed and left
// with its implementing commit. Only a reference to a still-present
// open row blocks.
func dependenciesSatisfied(d Plan, step Step) bool {
	for _, reference := range step.DependsOn {
		itemID, stepID, hasStep := strings.Cut(reference, "/")
		satisfied := true
		for _, item := range d.Items {
			if item.ID != itemID {
				continue
			}
			if !hasStep {
				satisfied = item.Status == StatusDone
				break
			}
			satisfied = !slices.ContainsFunc(item.Steps, func(candidate Step) bool {
				return candidate.ID == stepID && candidate.Status != StatusDone
			})
			break
		}
		if !satisfied {
			return false
		}
	}
	return true
}

var doctrineMetricLiteral = regexp.MustCompile(`(?i)\b[0-9][0-9,]*\s+(?:production\s+files?|files?|literals?|assumptions?|policy\s+copies)\b`)

// ValidateCampaignCensusAuthority requires OvergoDB census identity.
func ValidateCampaignCensusAuthority(d Plan) error {
	if d.Census == nil || !d.Census.Valid() || d.Census.Kind() != artifact.KindEvidence {
		return errors.New("plan: campaign lacks census evidence authority")
	}
	if doctrineMetricLiteral.MatchString(d.Doctrine) {
		return errors.New("plan: doctrine contains hand-typed census metrics")
	}
	return nil
}

func validStatus(status string) bool {
	return status == StatusOpen || status == StatusDone || strings.HasPrefix(status, "blocked")
}

// Current returns the role-owned open step, then the first unowned step.
// An open item with no open step yields the sentinel step "." (open the rung).
// ok is false when no open item remains.
func Current(d Plan, role string) (Item, Step, bool) {
	role = normalizedRole(role)
	if role != UnassignedRole {
		if item, step, ok := currentOwned(d, role); ok {
			return item, step, true
		}
	}
	return currentOwned(d, "")
}

func currentOwned(d Plan, owner string) (Item, Step, bool) {
	for _, it := range d.Items {
		if it.Status != StatusOpen || it.Owner != owner {
			continue
		}
		blocked := false
		for _, s := range it.Steps {
			if s.Status != StatusOpen {
				continue
			}
			// depends_on is enforced, not descriptive: a step whose
			// dependencies are not done cannot dispatch, whatever the
			// file order says.
			if dependenciesSatisfied(d, s) {
				return it, s, true
			}
			blocked = true
		}
		if blocked {
			continue
		}
		return it, Step{ID: ".", Title: "open the rung (define its steps)"}, true
	}
	return Item{}, Step{}, false
}

// AutomationRole resolves the explicit role, then the shared environment,
// and finally the unassigned fallback used by legacy open-work rows.
func AutomationRole(explicit string) (string, error) {
	role := strings.TrimSpace(explicit)
	if role == "" {
		role = strings.TrimSpace(os.Getenv(AutomationRoleEnvironment))
	}
	role = normalizedRole(role)
	if !validAutomationText(role) {
		return "", errors.New("plan: invalid automation role")
	}
	return role, nil
}

func validAutomationText(value string) bool {
	return textcheck.Bounded(value, automationRoleMaxBytes, "\x00\r\n")
}

func validAutomationDetail(value string) bool {
	return textcheck.Bounded(value, 2*automationRoleMaxBytes, "\x00\r\n")
}

func normalizedRole(role string) string {
	if role = strings.TrimSpace(role); role == "" {
		return UnassignedRole
	}
	return role
}

// Advance retains and completes one open row.
// Advance completes one step by REMOVING it: the plan holds only
// future, blocked, and in-progress work, and completion history lives
// in Git through the gate's structured trailers, not as retained rows.
// An item whose last step completes leaves the plan with it.
func Advance(d Plan, itemID, stepID string) (Plan, error) {
	d.Items = slices.Clone(d.Items)
	for itemIndex := range d.Items {
		if d.Items[itemIndex].ID != itemID {
			continue
		}
		if stepID == "." {
			if len(d.Items[itemIndex].Steps) != 0 {
				return Plan{}, fmt.Errorf("item %q has steps", itemID)
			}
			d.Items = slices.Delete(d.Items, itemIndex, itemIndex+1)
			return d, Validate(d)
		}
		d.Items[itemIndex].Steps = slices.Clone(d.Items[itemIndex].Steps)
		for stepIndex := range d.Items[itemIndex].Steps {
			if d.Items[itemIndex].Steps[stepIndex].ID != stepID {
				continue
			}
			if d.Items[itemIndex].Steps[stepIndex].Status != StatusOpen {
				return Plan{}, fmt.Errorf("step %q in %q is not open", stepID, itemID)
			}
			d.Items[itemIndex].Steps = slices.Delete(d.Items[itemIndex].Steps, stepIndex, stepIndex+1)
			if len(d.Items[itemIndex].Steps) == 0 {
				d.Items = slices.Delete(d.Items, itemIndex, itemIndex+1)
			}
			return d, Validate(d)
		}
		return Plan{}, fmt.Errorf("step %q not found in %q", stepID, itemID)
	}
	return Plan{}, fmt.Errorf("item %q not found", itemID)
}
