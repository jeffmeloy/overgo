// Package plan is the single owner of the campaign plan schema (docs/plan.json)
// and the "current step" rule. cmd/plan dispatches and enforces the loop;
// cmd/gate binds every commit to the current step. Both read the plan through
// this package so they can never disagree about which step is active -- the
// disagreement that would let off-plan work slip through the gate.
package plan

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"

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
	if err := jsonfile.DecodeStrict(filepath.FromSlash(path), &d); err != nil {
		return Plan{}, fmt.Errorf("parse %s: %w", path, err)
	}
	return d, nil
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
	items := map[string]bool{}
	for _, item := range d.Items {
		if item.ID == "" || items[item.ID] {
			return fmt.Errorf("plan item id %q is empty or duplicated", item.ID)
		}
		items[item.ID] = true
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

// Compact drops completed rows and normalizes partial rows back to open work.
func Compact(d Plan) Plan {
	items := d.Items[:0]
	for _, item := range d.Items {
		if item.Status == "done" {
			continue
		}
		steps := item.Steps[:0]
		for _, step := range item.Steps {
			if step.Status == "done" {
				continue
			}
			if step.Status == "partial" {
				if strings.HasPrefix(item.Status, "blocked") {
					step.Status = item.Status
				} else {
					step.Status = "open"
				}
			}
			steps = append(steps, step)
		}
		item.Steps = steps
		if item.Status == "open" && len(steps) > 0 && !slices.ContainsFunc(steps, func(step Step) bool { return step.Status == "open" }) {
			item.Status = steps[0].Status
			for _, step := range steps[1:] {
				if step.Status != item.Status {
					item.Status = "blocked-external-prereq"
					break
				}
			}
		}
		items = append(items, item)
	}
	d.Items = items
	return d
}

func unfinished(status string) bool {
	return status == "open" || strings.HasPrefix(status, "blocked")
}

// Current returns the first open step of the first open item -- the single
// action the loop is allowed to work on and the only step a commit may serve.
// An open item with no open step yields the sentinel step "." (open the rung).
// ok is false when no open item remains.
func Current(d Plan) (Item, Step, bool) {
	for _, it := range d.Items {
		if it.Status != "open" {
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

// Advance returns the open-work plan after removing one completed step. The
// gate writes this result in the implementation commit so dispatch cannot lag
// the code it describes.
func Advance(d Plan, itemID, stepID string) (Plan, error) {
	d.Items = slices.Clone(d.Items)
	for itemIndex := range d.Items {
		if d.Items[itemIndex].ID != itemID {
			continue
		}
		if stepID == "." {
			d.Items = append(d.Items[:itemIndex], d.Items[itemIndex+1:]...)
			return d, ValidateOpenWork(d)
		}
		d.Items[itemIndex].Steps = slices.Clone(d.Items[itemIndex].Steps)
		for stepIndex := range d.Items[itemIndex].Steps {
			if d.Items[itemIndex].Steps[stepIndex].ID != stepID {
				continue
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
