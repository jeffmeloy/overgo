package plan

import (
	"errors"
	"overgo/internal/worklease"
	"slices"
	"strings"
)

// Ref names one plan row as an exact item/step identity.
type Ref struct {
	Item string `json:"item"`
	Step string `json:"step"`
}

// String renders the canonical item/step reference.
func (r Ref) String() string { return r.Item + stepReferenceSeparator + r.Step }

// ReadyFrontier returns every dispatchable open step in deterministic
// document order: each open step whose dependencies are all proven by gated
// completion evidence or retained done state. Current remains the
// single-lane compatibility selector; its selection is always a member of
// this frontier, so a single-lane driver and a leased multi-lane driver can
// never disagree about what is dispatchable.
func ReadyFrontier(d Plan, authority CompletionAuthority) ([]Ref, error) {
	if !authority.resolves(d) {
		return nil, errors.New("plan: ready frontier requires a resolved completion authority for this exact plan")
	}
	frontier := []Ref{}
	for _, item := range d.Items {
		if item.Status != StatusOpen {
			continue
		}
		for _, step := range item.Steps {
			if step.Status != StatusOpen {
				continue
			}
			if dependenciesSatisfied(d, step, authority) {
				frontier = append(frontier, Ref{Item: item.ID, Step: step.ID})
			}
		}
	}
	return frontier, nil
}

// FrontierClaimsOverlap compares repo-relative claims across lanes: distinct
// worktrees still collide when they claim the same relative surface, because
// their gated commits meet again at merge time.
func FrontierClaimsOverlap(left, right worklease.WorkspaceClaims) bool {
	if left.WholeWorktree || right.WholeWorktree {
		return true
	}
	for _, write := range left.Write {
		for _, path := range slices.Concat(right.Read, right.Write) {
			if worklease.ClaimPathsOverlap(write, path) {
				return true
			}
		}
	}
	for _, write := range right.Write {
		for _, read := range left.Read {
			if worklease.ClaimPathsOverlap(write, read) {
				return true
			}
		}
	}
	return false
}

// FormatFrontier renders one row per line for operator and driver consumption.
func FormatFrontier(frontier []Ref) string {
	lines := make([]string, 0, len(frontier))
	for _, ref := range frontier {
		lines = append(lines, ref.String())
	}
	return strings.Join(lines, "\n")
}

// A plan reference joins its item and step with a slash.
const stepReferenceSeparator = "/"
