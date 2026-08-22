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

// Exclusion is affirmative evidence that one named check cannot observe the
// candidate change. Empty or unknown exclusions are invalid.
type Exclusion struct {
	Check  string `json:"check"`
	Reason string `json:"reason"`
}

// Impact is the complete applicability verdict from change analysis. An empty
// impact is unknown, so Plan runs checks unless an exclusion proves independence.
type Impact struct {
	Facts      []Fact      `json:"facts,omitempty"`
	Exclusions []Exclusion `json:"exclusions,omitempty"`
}

// MergeImpact combines independent producers without treating absent output as
// exclusion evidence.
func MergeImpact(parts ...Impact) Impact {
	var merged Impact
	for _, part := range parts {
		merged.Facts = append(merged.Facts, part.Facts...)
		merged.Exclusions = append(merged.Exclusions, part.Exclusions...)
	}
	slices.Sort(merged.Facts)
	merged.Facts = slices.Compact(merged.Facts)
	slices.SortFunc(merged.Exclusions, func(left, right Exclusion) int {
		return strings.Compare(left.Check, right.Check)
	})
	return merged
}

// ExclusionReason returns the producer's proof for one check.
func (impact Impact) ExclusionReason(check string) (string, bool) {
	for _, exclusion := range impact.Exclusions {
		if exclusion.Check == check {
			return exclusion.Reason, true
		}
	}
	return "", false
}

// Resource identifies an execution class without inventing a capacity.
type Resource struct {
	Name      string `json:"name"`
	Exclusive bool   `json:"exclusive,omitempty"`
}

// Ownership declares the exact package or symbol surface one check verifies.
// Fact must also be one of the descriptor's triggers.
type Ownership struct {
	Fact            Fact     `json:"fact"`
	Packages        []string `json:"packages,omitempty"`
	PackagePrefixes []string `json:"package_prefixes,omitempty"`
	Symbols         []Symbol `json:"symbols,omitempty"`
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
	Ownership    Ownership       `json:"ownership,omitempty"`
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
	Reused       bool                  `json:"reused,omitempty"`
	Detail       string                `json:"detail,omitempty"`
}

// Plan selects applicable checks and returns them in dependency order.
func Plan(checks []Check, impact Impact) ([]Invocation, error) {
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
	}
	exclusions, err := validateImpact(impact, definitions)
	if err != nil {
		return nil, err
	}
	for _, check := range checks {
		name := check.Descriptor.Name
		triggered := intersects(check.Descriptor.Triggers, impact.Facts)
		if triggered && exclusions[name] != "" {
			return nil, fmt.Errorf("automation check %q: impact both triggers and excludes the check", name)
		}
		selected[name] = check.Descriptor.Always || triggered || exclusions[name] == ""
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
			matched := matches(check.Descriptor.Triggers, impact.Facts)
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

func validateImpact(impact Impact, definitions map[string]Check) (map[string]string, error) {
	for _, fact := range impact.Facts {
		if strings.TrimSpace(string(fact)) == "" {
			return nil, errors.New("automation check: impact contains an empty fact")
		}
	}
	exclusions := make(map[string]string, len(impact.Exclusions))
	for _, exclusion := range impact.Exclusions {
		name, reason := strings.TrimSpace(exclusion.Check), strings.TrimSpace(exclusion.Reason)
		if name == "" || reason == "" {
			return nil, errors.New("automation check: exclusion requires a check and reason")
		}
		definition, exists := definitions[name]
		if !exists {
			return nil, fmt.Errorf("automation check: exclusion names unknown check %q", name)
		}
		if definition.Descriptor.Always {
			return nil, fmt.Errorf("automation check: exclusion targets always-required check %q", name)
		}
		if _, duplicate := exclusions[name]; duplicate {
			return nil, fmt.Errorf("automation check: duplicate exclusion for %q", name)
		}
		exclusions[name] = reason
	}
	return exclusions, nil
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
		Reused       bool                  `json:"reused,omitempty"`
		Detail       string                `json:"detail,omitempty"`
	}{evidence.InvocationID, evidence.Outcome, evidence.Skipped, evidence.Reused, evidence.Detail})
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
	if descriptor.Ownership.Fact != "" {
		if len(descriptor.Ownership.Packages) == 0 && len(descriptor.Ownership.PackagePrefixes) == 0 && len(descriptor.Ownership.Symbols) == 0 {
			return fmt.Errorf("automation check %q: ownership has no surface", descriptor.Name)
		}
		if !slices.Contains(descriptor.Triggers, descriptor.Ownership.Fact) {
			return fmt.Errorf("automation check %q: ownership fact is not a trigger", descriptor.Name)
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
