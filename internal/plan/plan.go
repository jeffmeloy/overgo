// Package plan owns campaign state and dispatch order.
package plan

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
	"overgo/internal/worklease"
)

// AutomationRoleEnvironment is the shared lane-role input for dispatchers.
const AutomationRoleEnvironment = worklease.AutomationRoleEnvironment

// Path is the campaign plan, relative to the repo root.
const Path = "docs/plan.json"

// ScopeLane retains only work explicitly owned by the declared lane.
const ScopeLane = "lane"

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
	Verify string `json:"verify,omitzero"`
	// Campaign detail carried from the merged design record (owner directive
	// 2026-08-16: one plan document). Rationale says why the row exists;
	// DependsOn declares ordering the queue must respect; Capabilities name
	// the discipline contracts the row exercises; Outcome records the
	// measured verdict verbatim once the row has run.
	Rationale    string          `json:"rationale,omitzero"`
	DependsOn    []string        `json:"depends_on,omitempty"`
	Capabilities []string        `json:"capabilities,omitempty"`
	Outcome      json.RawMessage `json:"outcome,omitempty"`
	// VerificationBatch declares subordinate acceptance obligations. It does
	// not confer completion or permission to omit any full-gate check.
	VerificationBatch *VerificationBatch `json:"verification_batch,omitempty"`
}

// Budget is the scope a row declared through the plan. Nodes is the
// production syntax nodes the row may still add, over all its landings: each
// landing takes what it adds from the count, and returns what it removes, in
// the plan it publishes. A row without a count maintains: it may not grow
// production code at all, and a negative count is a row that must shrink.
// Reason says why the row grows; it explains a count and admits nothing by
// itself. A landing reads the count from the plan of the commit it builds on,
// so a change cannot raise its own. Paydown names the open row that pays for a
// harness baseline raise.
type Budget struct {
	Nodes   int    `json:"nodes"`
	Paydown string `json:"paydown,omitzero"`
	Reason  string `json:"reason"`
}

// A row's class decides the rules it lands under; an undeclared row is a feature.
const (
	// ClassConsolidation rows must shrink production lines.
	ClassConsolidation = "consolidation"
	// ClassMerge rows land a merge: the one plan-only lane commit, not awaited by closeout.
	ClassMerge = "merge"
)

// Item is one rung of the ladder.
type Item struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Class  string `json:"class,omitzero"`
	Owner  string `json:"owner,omitzero"`
	Status string `json:"status"`
	Budget Budget `json:"budget,omitzero"`
	Steps  []Step `json:"steps"`
}

// Plan is the whole campaign surface.
type Plan struct {
	Campaign string `json:"campaign"`
	Doctrine string `json:"doctrine"`
	// Lane names the worktree this plan dispatches for: under the unassigned
	// role it is the role, so the lane's own rows come first, unowned rows
	// after, and another lane's rows never.
	Lane  string `json:"lane,omitzero"`
	Scope string `json:"scope,omitzero"`
	Items []Item `json:"items"`
	// Decisions are the questions only the owner answers, as plain data
	// rather than prose in a rationale: plan -decisions lists the open
	// ones with how long they have waited.
	Decisions []Decision `json:"decisions,omitempty"`
}

// decisionOptions is the fewest options a decision offers: one option is
// no choice.
const decisionOptions = 2

// Answer records the owner's answer to an open decision. The answer is
// one of its options, given once.
func Answer(document Plan, id, option string) (Plan, error) {
	at := slices.IndexFunc(document.Decisions, func(decision Decision) bool { return decision.ID == id })
	switch {
	case at < 0:
		return document, fmt.Errorf("plan: no decision %q", id)
	case document.Decisions[at].Answer != "":
		return document, fmt.Errorf("plan: decision %q is already answered: %s", id, document.Decisions[at].Answer)
	case !slices.Contains(document.Decisions[at].Options, option):
		return document, fmt.Errorf("plan: %q is not one of decision %q's options: %s", option, id, strings.Join(document.Decisions[at].Options, " | "))
	}
	document.Decisions = slices.Clone(document.Decisions)
	document.Decisions[at].Answer = option
	return document, nil
}

// Decision is one question for the owner: its options, the recommended
// one, the items it blocks, the day it was asked (YYYY-MM-DD) and, once
// given, the answer.
type Decision struct {
	ID             string   `json:"id"`
	Question       string   `json:"question"`
	Options        []string `json:"options"`
	Recommendation string   `json:"recommendation"`
	Blocks         []string `json:"blocks,omitempty"`
	Asked          string   `json:"asked"`
	Answer         string   `json:"answer,omitzero"`
}

// Load reads the plan from path (Path when empty).
func Load(path string) (Plan, error) {
	path = cmp.Or(path, Path)
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

// Save writes the plan back to path (Path when empty).
func Save(path string, d Plan) error {
	if err := Validate(d); err != nil {
		return err
	}
	path = cmp.Or(path, Path)
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
	seen := map[string]bool{}
	for _, decision := range d.Decisions {
		// The recommendation is one of the options, so an answer is one
		// choice; the items it blocks may land and leave the plan, but each
		// is an item's name: a blank one blocks nothing and says it does.
		if _, err := time.Parse(time.DateOnly, decision.Asked); err != nil || decision.ID == "" || seen[decision.ID] ||
			decision.Question == "" || len(decision.Options) < decisionOptions || !slices.Contains(decision.Options, decision.Recommendation) ||
			slices.ContainsFunc(decision.Blocks, func(item string) bool { return !worklease.ValidPlanID(item) }) {
			return fmt.Errorf("plan decision %q needs a unique id, a question, two or more options, one of them recommended, the day it was asked, and item names for what it blocks", decision.ID)
		}
		seen[decision.ID] = true
	}
	return validateDependencies(d)
}

func validatePlanGraph(d Plan) error {
	if d.Scope != "" && (d.Scope != ScopeLane || normalizedRole(d.Lane) == worklease.UnassignedRole) {
		return errors.New("plan: lane scope requires an explicit lane")
	}
	if d.Lane != "" && !worklease.ValidAutomationText(d.Lane) {
		return errors.New("plan: invalid lane")
	}
	items := map[string]bool{}
	for _, item := range d.Items {
		if d.Scope == ScopeLane && item.Owner != d.Lane {
			return fmt.Errorf("plan: lane-scoped item %s must be owned by %s", item.ID, d.Lane)
		}
		if !worklease.ValidPlanID(item.ID) || items[item.ID] {
			return fmt.Errorf("plan item id %q is invalid or duplicated", item.ID)
		}
		items[item.ID] = true
		if item.Owner != "" && !worklease.ValidAutomationText(item.Owner) {
			return fmt.Errorf("plan item %s has invalid owner", item.ID)
		}
		if !validStatus(item.Status) || !slices.Contains([]string{"", ClassConsolidation, ClassMerge}, item.Class) {
			return fmt.Errorf("plan item %s has invalid status %q or class %q", item.ID, item.Status, item.Class)
		}
		steps := map[string]bool{}
		for _, step := range item.Steps {
			if !worklease.ValidPlanID(step.ID) || steps[step.ID] {
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
			if err := validateVerificationBatch(step.VerificationBatch); err != nil {
				return fmt.Errorf("plan step %s/%s: %w", item.ID, step.ID, err)
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
				if !hasStep || !worklease.ValidPlanID(target) || !worklease.ValidPlanID(targetStep) {
					return fmt.Errorf("plan step %s: depends_on %q must name one exact item/step", source, reference)
				}
				// Only present targets join the retained cycle graph. An
				// absent target is a retired or landed step, which dispatch
				// counts as satisfied.
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
// settled: done, or no longer in the plan. A row leaves the plan when a
// landing completes or retires it, and that commit's message says which.
func dependenciesSatisfied(d Plan, step Step) bool {
	for _, reference := range step.DependsOn {
		itemID, stepID, _ := strings.Cut(reference, "/")
		if candidate, present := exactPlanStep(d, itemID, stepID); present && candidate.Status != StatusDone {
			return false
		}
	}
	return true
}

func validStatus(status string) bool {
	return status == StatusOpen || status == StatusDone || strings.HasPrefix(status, "blocked")
}

// Current returns the role-owned open step, then the first unowned step.
// An open item with no open step yields the sentinel step "." (open the rung).
// ok is false when no open item remains.
func Current(d Plan, role string) (Item, Step, bool) {
	return currentResolved(d, role)
}

func currentResolved(d Plan, role string) (Item, Step, bool) {
	role = normalizedRole(role)
	// The plan's lane is the role nobody named: the rule reads the plan, so a
	// worktree and the gate's candidate tree dispatch the same rows.
	if role == worklease.UnassignedRole && d.Lane != "" {
		role = d.Lane
	}
	if role != worklease.UnassignedRole {
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
	if !worklease.ValidAutomationText(role) {
		return "", errors.New("plan: invalid automation role")
	}
	return role, nil
}

func validAutomationDetail(value string) bool {
	return textcheck.Bounded(value, 2*worklease.AutomationTextMaxBytes, "\x00\r\n")
}

func normalizedRole(role string) string {
	if role = strings.TrimSpace(role); role == "" {
		return worklease.UnassignedRole
	}
	return role
}

// Advance completes one step by REMOVING it: the plan holds only
// future, blocked, and in-progress work, and completion history lives
// in Git through the gate's structured trailers, not as retained rows.
// An item whose last step completes leaves the plan with it.
func Advance(d Plan, itemID, stepID string) (Plan, error) {
	advanced, err := advancePlan(d, itemID, stepID, stepID == ".")
	if err != nil {
		return Plan{}, err
	}
	return advanced, Validate(advanced)
}

// advancePlan owns row mutation for live and historical plans. Historical
// evidence selects an exact step and leaves schema validation to its reader;
// only the live API accepts the whole-item shortcut.
func advancePlan(d Plan, itemID, stepID string, wholeItem bool) (Plan, error) {
	d.Items = slices.Clone(d.Items)
	for itemIndex := range d.Items {
		if d.Items[itemIndex].ID != itemID {
			continue
		}
		if wholeItem {
			if len(d.Items[itemIndex].Steps) != 0 {
				return Plan{}, fmt.Errorf("item %q has steps", itemID)
			}
			d.Items = slices.Delete(d.Items, itemIndex, itemIndex+1)
			return d, nil
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
			return d, nil
		}
		return Plan{}, fmt.Errorf("step %q not found in %q", stepID, itemID)
	}
	return Plan{}, fmt.Errorf("item %q not found", itemID)
}

// Digest returns the existing canonical plan identity; unencodable state returns empty.
func (document Plan) Digest() string {
	encoded, err := json.Marshal(document)
	if err != nil {
		return ""
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:])
}

func exactPlanItem(document Plan, itemID string) (Item, bool) {
	for _, item := range document.Items {
		if item.ID == itemID {
			return item, true
		}
	}
	return Item{}, false
}

func exactPlanStep(document Plan, itemID, stepID string) (Step, bool) {
	item, _ := exactPlanItem(document, itemID)
	for _, step := range item.Steps {
		if step.ID == stepID {
			return step, true
		}
	}
	return Step{}, false
}
