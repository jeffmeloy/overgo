// Package plan is the single owner of the campaign plan schema (docs/plan.json)
// and the "current step" rule. cmd/plan dispatches and enforces the loop;
// cmd/gate binds every commit to the current step. Both read the plan through
// this package so they can never disagree about which step is active -- the
// disagreement that would let off-plan work slip through the gate.
package plan

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/jsonfile"
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
}

// Item is one rung of the ladder.
type Item struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Status string `json:"status"`
	Steps  []Step `json:"steps"`
}

// Plan is the whole campaign surface.
type Plan struct {
	Campaign string `json:"campaign"`
	Doctrine string `json:"doctrine"`
	Items    []Item `json:"items"`
}

// Load reads the plan from path (Path when empty).
func Load(path string) (Plan, error) {
	if path == "" {
		path = Path
	}
	var d Plan
	if err := jsonfile.Decode(filepath.FromSlash(path), &d); err != nil {
		return Plan{}, fmt.Errorf("parse %s: %w", path, err)
	}
	if err := ValidateOpenOnly(d); err != nil {
		return Plan{}, fmt.Errorf("validate %s: %w", path, err)
	}
	return d, nil
}

// ValidateOpenOnly rejects completed chronology from the live work queue.
// Completed and blocked work belongs in commits, evidence, or findings.
func ValidateOpenOnly(d Plan) error {
	for _, it := range d.Items {
		if it.Status != "open" {
			return fmt.Errorf("item %q has historical status %q", it.ID, it.Status)
		}
		for _, step := range it.Steps {
			if step.Status != "open" {
				return fmt.Errorf("step %q/%q has historical status %q", it.ID, step.ID, step.Status)
			}
		}
	}
	return nil
}

// Save writes the plan back to path (Path when empty).
func Save(path string, d Plan) error {
	if path == "" {
		path = Path
	}
	raw, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.FromSlash(path), append(raw, '\n'), 0o644)
}

// Current returns the first step of the first item -- the single action the
// loop may work on. Persisted plans contain open work only. An item with no
// steps yields the sentinel step "." (open the rung).
func Current(d Plan) (Item, Step, bool) {
	if len(d.Items) == 0 {
		return Item{}, Step{}, false
	}
	it := d.Items[0]
	if len(it.Steps) == 0 {
		return it, Step{ID: ".", Title: "open the rung (define its steps)"}, true
	}
	return it, it.Steps[0], true
}
