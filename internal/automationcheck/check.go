// Package automationcheck separates verification work from the policy that
// requires it and the scheduler that executes it.
package automationcheck

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// Fact is one normalized dimension of derived change impact. The producer
// owns its namespace; checks only match exact facts.
type Fact string

// Resource identifies an execution class without inventing a capacity.
type Resource struct {
	Name      string `json:"name"`
	Exclusive bool   `json:"exclusive,omitempty"`
}

// Descriptor is the immutable, policy-neutral definition of a check.
type Descriptor struct {
	Name         string          `json:"name"`
	Phase        runrecord.Phase `json:"phase"`
	Always       bool            `json:"always,omitempty"`
	Triggers     []Fact          `json:"triggers,omitempty"`
	Inapplicable string          `json:"inapplicable,omitempty"`
	Dependencies []string        `json:"dependencies,omitempty"`
	Resources    []Resource      `json:"resources,omitempty"`
}

// Runner returns whether work was inapplicable, diagnostic detail, and error.
type Runner func(context.Context, Invocation) (bool, string, error)

// Check pairs one descriptor with its concrete implementation.
type Check struct {
	Descriptor Descriptor
	Run        Runner
}

// Invocation is a content-addressed unit of verification work.
type Invocation struct {
	ID      artifact.ID `json:"id"`
	Check   Descriptor  `json:"check"`
	Matched []Fact      `json:"matched,omitempty"`
	runner  Runner
}

// Evidence is the typed terminal result of one invocation.
type Evidence struct {
	ID           artifact.ID           `json:"id"`
	InvocationID artifact.ID           `json:"invocation_id"`
	Name         string                `json:"name"`
	Phase        runrecord.Phase       `json:"phase"`
	Outcome      runrecord.LaneOutcome `json:"outcome"`
	DurationNS   uint64                `json:"duration_ns"`
	Skipped      bool                  `json:"skipped,omitempty"`
	Detail       string                `json:"detail,omitempty"`
}

// Plan selects applicable checks and returns them in dependency order.
func Plan(checks []Check, impact []Fact) ([]Invocation, error) {
	definitions := make(map[string]Check, len(checks))
	selected := map[string]bool{}
	for _, check := range checks {
		if err := validate(check); err != nil {
			return nil, err
		}
		name := check.Descriptor.Name
		if _, exists := definitions[name]; exists {
			return nil, fmt.Errorf("automation check: duplicate name %q", name)
		}
		definitions[name] = check
		selected[name] = check.Descriptor.Always || intersects(check.Descriptor.Triggers, impact)
	}
	for name, active := range selected {
		if active {
			if err := include(name, definitions, selected, map[string]bool{}); err != nil {
				return nil, err
			}
		}
	}
	var planned []Invocation
	done := map[string]bool{}
	for len(planned) < selectedCount(selected) {
		before := len(planned)
		for _, check := range checks {
			name := check.Descriptor.Name
			if !selected[name] || done[name] || !dependenciesDone(check.Descriptor.Dependencies, done) {
				continue
			}
			matched := matches(check.Descriptor.Triggers, impact)
			id, err := artifact.JSONID(artifact.KindRecipe, struct {
				Descriptor Descriptor `json:"descriptor"`
				Matched    []Fact     `json:"matched,omitempty"`
			}{check.Descriptor, matched})
			if err != nil {
				return nil, fmt.Errorf("automation check %q: identify: %w", name, err)
			}
			planned = append(planned, Invocation{ID: id, Check: check.Descriptor, Matched: matched, runner: check.Run})
			done[name] = true
		}
		if len(planned) == before {
			return nil, errors.New("automation check: dependency cycle")
		}
	}
	return planned, nil
}

// Run executes one planned check and returns evidence on passing and failing paths.
func Run(ctx context.Context, invocation Invocation) (Evidence, error) {
	if !invocation.ID.Valid() || invocation.runner == nil {
		return Evidence{}, errors.New("automation check: invalid invocation")
	}
	begin := time.Now()
	skipped, detail, runErr := invocation.runner(ctx, invocation)
	evidence := Evidence{
		InvocationID: invocation.ID, Name: invocation.Check.Name, Phase: invocation.Check.Phase,
		Outcome: runrecord.LaneOutcomeOf(runErr), DurationNS: max(uint64(time.Since(begin).Nanoseconds()), uint64(time.Nanosecond)),
		Skipped: skipped, Detail: strings.TrimSpace(detail),
	}
	id, err := artifact.JSONID(artifact.KindEvidence, struct {
		InvocationID artifact.ID           `json:"invocation_id"`
		Outcome      runrecord.LaneOutcome `json:"outcome"`
		Skipped      bool                  `json:"skipped,omitempty"`
		Detail       string                `json:"detail,omitempty"`
	}{evidence.InvocationID, evidence.Outcome, evidence.Skipped, evidence.Detail})
	if err != nil {
		return Evidence{}, fmt.Errorf("automation check %q: identify evidence: %w", evidence.Name, err)
	}
	evidence.ID = id
	return evidence, runErr
}

func validate(check Check) error {
	descriptor := check.Descriptor
	if strings.TrimSpace(descriptor.Name) == "" || descriptor.Phase == "" || check.Run == nil {
		return errors.New("automation check: name, phase, and runner are required")
	}
	if !descriptor.Always && len(descriptor.Triggers) == 0 {
		return fmt.Errorf("automation check %q: no applicability trigger", descriptor.Name)
	}
	if !descriptor.Always && strings.TrimSpace(descriptor.Inapplicable) == "" {
		return fmt.Errorf("automation check %q: no inapplicable evidence", descriptor.Name)
	}
	for _, fact := range descriptor.Triggers {
		if strings.TrimSpace(string(fact)) == "" {
			return fmt.Errorf("automation check %q: empty trigger", descriptor.Name)
		}
	}
	for _, resource := range descriptor.Resources {
		if strings.TrimSpace(resource.Name) == "" {
			return fmt.Errorf("automation check %q: empty resource", descriptor.Name)
		}
	}
	return nil
}

func include(name string, definitions map[string]Check, selected, visiting map[string]bool) error {
	if visiting[name] {
		return errors.New("automation check: dependency cycle")
	}
	visiting[name] = true
	for _, dependency := range definitions[name].Descriptor.Dependencies {
		if _, exists := definitions[dependency]; !exists {
			return fmt.Errorf("automation check %q: unknown dependency %q", name, dependency)
		}
		selected[dependency] = true
		if err := include(dependency, definitions, selected, visiting); err != nil {
			return err
		}
	}
	delete(visiting, name)
	return nil
}

func intersects(triggers, impact []Fact) bool { return len(matches(triggers, impact)) > 0 }

func matches(triggers, impact []Fact) []Fact {
	wanted := make(map[Fact]bool, len(triggers))
	for _, fact := range triggers {
		wanted[fact] = true
	}
	matched := slices.DeleteFunc(slices.Clone(impact), func(fact Fact) bool { return !wanted[fact] })
	slices.Sort(matched)
	return slices.Compact(matched)
}

func dependenciesDone(dependencies []string, done map[string]bool) bool {
	return !slices.ContainsFunc(dependencies, func(name string) bool { return !done[name] })
}

func selectedCount(selected map[string]bool) int {
	count := 0
	for _, active := range selected {
		if active {
			count++
		}
	}
	return count
}
