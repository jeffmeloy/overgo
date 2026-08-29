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
	return parsePlan(data, Validate)
}

func parsePlan(data []byte, validate func(Plan) error) (Plan, error) {
	var document Plan
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return Plan{}, err
	}
	return document, validate(document)
}

// ParseHistorical decodes one immutable Git plan snapshot without applying
// current live-plan lifecycle rules. Historical rows may retain completion
// states or old dependency syntax; their exact Git transition remains the
// completion authority.
func ParseHistorical(data []byte) (Plan, error) {
	return parsePlan(data, validatePlanGraph)
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
	if err := validatePlanGraph(d); err != nil {
		return err
	}
	for _, item := range d.Items {
		if item.Status == StatusDone {
			return fmt.Errorf("plan item %s retains completed state", item.ID)
		}
		for _, step := range item.Steps {
			if step.Status == StatusDone {
				return fmt.Errorf("plan step %s/%s retains completed state", item.ID, step.ID)
			}
		}
	}
	return validateDependencies(d)
}

func validatePlanGraph(d Plan) error {
	if d.Census != nil && (!d.Census.Valid() || d.Census.Kind() != artifact.KindEvidence) {
		return errors.New("plan: census authority is not evidence")
	}
	items := map[string]bool{}
	for _, item := range d.Items {
		if !validPlanID(item.ID) || items[item.ID] {
			return fmt.Errorf("plan item id %q is invalid or duplicated", item.ID)
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
			if !validPlanID(step.ID) || steps[step.ID] {
				return fmt.Errorf("plan step %s/%s is invalid or duplicated", item.ID, step.ID)
			}
			steps[step.ID] = true
			if !validStatus(step.Status) {
				return fmt.Errorf("plan step %s/%s has invalid status %q", item.ID, step.ID, step.Status)
			}
			if step.Status == StatusOpen && strings.TrimSpace(step.Verify) == "" {
				return fmt.Errorf("plan step %s/%s is open without a verifier", item.ID, step.ID)
			}
			if step.Verify != "" && !validAutomationDetail(step.Verify) {
				return fmt.Errorf("plan step %s/%s has an invalid verifier", item.ID, step.ID)
			}
		}
		if item.Status == StatusDone && slices.ContainsFunc(item.Steps, func(step Step) bool { return step.Status != StatusDone }) {
			return fmt.Errorf("plan item %s is done with unfinished steps", item.ID)
		}
	}
	return nil
}

// validateDependencies requires exact item/step references and refuses
// dependency cycles among retained steps. References to absent rows remain
// structurally representable because completion prunes them; dispatch requires
// ResolveCompletionAuthority to prove each such reference independently.
func validateDependencies(d Plan) error {
	itemIndex := map[string]Item{}
	for _, item := range d.Items {
		itemIndex[item.ID] = item
	}
	edges := map[string][]string{}
	for _, item := range d.Items {
		for _, step := range item.Steps {
			source := item.ID + "/" + step.ID
			for _, reference := range step.DependsOn {
				target, targetStep, hasStep := strings.Cut(reference, "/")
				if !hasStep || !validPlanID(target) || !validPlanID(targetStep) {
					return fmt.Errorf("plan step %s: depends_on %q must name one exact item/step", source, reference)
				}
				// Only present targets join the retained cycle graph. An
				// absent target is not presumed complete here; the runtime
				// completion authority decides whether it may dispatch.
				targetItem, present := itemIndex[target]
				if !present {
					continue
				}
				if !slices.ContainsFunc(targetItem.Steps, func(candidate Step) bool { return candidate.ID == targetStep }) {
					// Pruned sibling steps likewise leave the retained graph;
					// their Git-and-store proof is checked at dispatch.
					continue
				}
				if target == item.ID && targetStep == step.ID {
					return fmt.Errorf(
						"plan step %s: depends_on %q can never satisfy: a step cannot depend on itself",
						source, reference)
				}
				edges[source] = append(edges[source], target+"/"+targetStep)
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
// complete. A retained done row is self-evident; a pruned row is satisfied
// only by the derived Git-and-store completion authority. Absence alone is
// never completion evidence.
func dependenciesSatisfied(d Plan, step Step, authority CompletionAuthority) bool {
	for _, reference := range step.DependsOn {
		itemID, stepID, _ := strings.Cut(reference, "/")
		present := false
		satisfied := false
		for _, item := range d.Items {
			if item.ID != itemID {
				continue
			}
			for _, candidate := range item.Steps {
				if candidate.ID == stepID {
					present = true
					satisfied = candidate.Status == StatusDone
					break
				}
			}
			break
		}
		if !present {
			satisfied = authority.completed(reference)
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
func Current(d Plan, role string, authority CompletionAuthority) (Item, Step, bool) {
	if !authority.resolves(d) {
		return Item{}, Step{}, false
	}
	return currentResolved(d, role, authority)
}

func currentResolved(d Plan, role string, authority CompletionAuthority) (Item, Step, bool) {
	role = normalizedRole(role)
	if role != UnassignedRole {
		if item, step, ok := currentOwned(d, role, authority); ok {
			return item, step, true
		}
	}
	return currentOwned(d, "", authority)
}

func currentOwned(d Plan, owner string, authority CompletionAuthority) (Item, Step, bool) {
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
			if dependenciesSatisfied(d, s, authority) {
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

func validPlanID(value string) bool {
	return textcheck.BoundedToken(value, automationRoleMaxBytes, "/\\\x00\r\n")
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
