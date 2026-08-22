// Package plan is the single owner of the campaign plan schema (docs/plan.json)
// and the "current step" rule. cmd/plan dispatches and enforces the loop;
// cmd/gate binds every commit to the current step. Both read the plan through
// this package so they can never disagree about which step is active -- the
// disagreement that would let off-plan work slip through the gate.
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

// CompletionRef is the compact merge authority retained after an open step is
// removed. Its full receipt lives in RepoDB under the same evidence authority.
type CompletionRef struct {
	Item      string      `json:"item"`
	Step      string      `json:"step"`
	Owner     string      `json:"owner,omitempty"`
	Authority artifact.ID `json:"authority"`
}

// Plan is the whole campaign surface.
type Plan struct {
	Campaign  string          `json:"campaign"`
	Doctrine  string          `json:"doctrine"`
	Census    *artifact.ID    `json:"census_evidence,omitempty"`
	Items     []Item          `json:"items"`
	Completed []CompletionRef `json:"completed,omitempty"`
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

// Parse decodes an in-memory plan projection, including projections read from
// Git refs during semantic master synchronization.
func Parse(data []byte) (Plan, error) {
	var document Plan
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return Plan{}, err
	}
	return document, ValidateOpenWork(document)
}

// Save writes the plan back to path (Path when empty).
func Save(path string, d Plan) error {
	if err := ValidateOpenWork(d); err != nil {
		return err
	}
	if path == "" {
		path = Path
	}
	return jsonfile.Write(filepath.FromSlash(path), d, 0o644)
}

// ValidateOpenWork rejects chronology in the live plan. Git and RepoDB own
// completed evidence; the plan contains only dispatchable or blocked work.
func ValidateOpenWork(d Plan) error {
	if d.Census != nil && (!d.Census.Valid() || d.Census.Kind() != artifact.KindEvidence) {
		return errors.New("plan: census authority is not evidence")
	}
	completed := map[string]bool{}
	for _, receipt := range d.Completed {
		key := receipt.Item + "/" + receipt.Step
		if receipt.Item == "" || receipt.Step == "" || !receipt.Authority.Valid() ||
			receipt.Authority.Kind() != artifact.KindEvidence || completed[key] ||
			receipt.Owner != "" && !textcheck.Bounded(receipt.Owner, automationRoleMaxBytes, "\x00\r\n") {
			return fmt.Errorf("plan completion %q is invalid or duplicated", key)
		}
		completed[key] = true
	}
	items := map[string]bool{}
	for _, item := range d.Items {
		if item.ID == "" || items[item.ID] {
			return fmt.Errorf("plan item id %q is empty or duplicated", item.ID)
		}
		items[item.ID] = true
		if item.Owner != "" && !textcheck.Bounded(item.Owner, automationRoleMaxBytes, "\x00\r\n") {
			return fmt.Errorf("plan item %s has invalid owner", item.ID)
		}
		if !unfinished(item.Status) {
			return fmt.Errorf("plan item %s has chronology status %q", item.ID, item.Status)
		}
		steps := map[string]bool{}
		for _, step := range item.Steps {
			if step.ID == "" || steps[step.ID] {
				return fmt.Errorf("plan step %s/%s is empty or duplicated", item.ID, step.ID)
			}
			steps[step.ID] = true
			if !unfinished(step.Status) {
				return fmt.Errorf("plan step %s/%s has chronology status %q", item.ID, step.ID, step.Status)
			}
			if step.Status == "open" && strings.TrimSpace(step.Verify) == "" {
				return fmt.Errorf("plan step %s/%s is open without a verifier", item.ID, step.ID)
			}
		}
	}
	return nil
}

var doctrineMetricLiteral = regexp.MustCompile(`(?i)\b[0-9][0-9,]*\s+(?:production\s+files?|files?|literals?|assumptions?|policy\s+copies)\b`)

// ValidateCampaignCensusAuthority requires campaign baselines to point to
// immutable census evidence and rejects copied metric values in doctrine prose.
func ValidateCampaignCensusAuthority(d Plan) error {
	if d.Census == nil || !d.Census.Valid() || d.Census.Kind() != artifact.KindEvidence {
		return errors.New("plan: campaign lacks census evidence authority")
	}
	if doctrineMetricLiteral.MatchString(d.Doctrine) {
		return errors.New("plan: doctrine contains hand-typed census metrics")
	}
	return nil
}

func unfinished(status string) bool {
	return status == "open" || strings.HasPrefix(status, "blocked")
}

// Current returns the first open step of the first open item -- the single
// action the loop is allowed to work on and the only step a commit may serve.
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
		if it.Status != "open" || it.Owner != owner {
			continue
		}
		for _, s := range it.Steps {
			if s.Status == "open" {
				return it, s, true
			}
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
	if !textcheck.Bounded(role, automationRoleMaxBytes, "\x00\r\n") {
		return "", errors.New("plan: invalid automation role")
	}
	return role, nil
}

func normalizedRole(role string) string {
	if role = strings.TrimSpace(role); role == "" {
		return UnassignedRole
	}
	return role
}

// Advance returns the open-work plan after removing one completed step. The
// gate writes this result in the implementation commit so dispatch cannot lag
// the code it describes.
func Advance(d Plan, itemID, stepID string) (Plan, error) {
	return advance(d, itemID, stepID, nil)
}

// AdvanceWithEvidence removes one open step and retains its exact completion
// authority for cross-lane merge reconciliation.
func AdvanceWithEvidence(d Plan, itemID, stepID string, authority artifact.ID) (Plan, error) {
	if !authority.Valid() || authority.Kind() != artifact.KindEvidence {
		return Plan{}, errors.New("plan: completion authority is not evidence")
	}
	return advance(d, itemID, stepID, &authority)
}

func advance(d Plan, itemID, stepID string, authority *artifact.ID) (Plan, error) {
	d.Items = slices.Clone(d.Items)
	for itemIndex := range d.Items {
		if d.Items[itemIndex].ID != itemID {
			continue
		}
		if stepID == "." {
			if authority != nil {
				d.Completed = appendCompletion(d.Completed, CompletionRef{Item: itemID, Step: stepID, Owner: d.Items[itemIndex].Owner, Authority: *authority})
			}
			d.Items = append(d.Items[:itemIndex], d.Items[itemIndex+1:]...)
			return d, ValidateOpenWork(d)
		}
		d.Items[itemIndex].Steps = slices.Clone(d.Items[itemIndex].Steps)
		for stepIndex := range d.Items[itemIndex].Steps {
			if d.Items[itemIndex].Steps[stepIndex].ID != stepID {
				continue
			}
			if authority != nil {
				d.Completed = appendCompletion(d.Completed, CompletionRef{Item: itemID, Step: stepID, Owner: d.Items[itemIndex].Owner, Authority: *authority})
			}
			d.Items[itemIndex].Steps = append(
				d.Items[itemIndex].Steps[:stepIndex],
				d.Items[itemIndex].Steps[stepIndex+1:]...,
			)
			if len(d.Items[itemIndex].Steps) == 0 {
				d.Items = append(d.Items[:itemIndex], d.Items[itemIndex+1:]...)
			}
			return d, ValidateOpenWork(d)
		}
		return Plan{}, fmt.Errorf("step %q not found in %q", stepID, itemID)
	}
	return Plan{}, fmt.Errorf("item %q not found", itemID)
}

func appendCompletion(completed []CompletionRef, receipt CompletionRef) []CompletionRef {
	result := slices.Clone(completed)
	for _, existing := range result {
		if existing.Item == receipt.Item && existing.Step == receipt.Step {
			return result
		}
	}
	return append(result, receipt)
}
