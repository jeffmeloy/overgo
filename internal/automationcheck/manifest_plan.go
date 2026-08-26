package automationcheck

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// PlannedInvocation is the immutable runner-free portion of an invocation.
type PlannedInvocation struct {
	ID      artifact.ID `json:"id"`
	Check   Descriptor  `json:"check"`
	Matched []Fact      `json:"matched,omitempty"`
}

// ManifestPlan binds verification selection to exact structural authority.
type ManifestPlan struct {
	BaseManifest      artifact.ID         `json:"base_manifest"`
	CandidateManifest artifact.ID         `json:"candidate_manifest"`
	SurfaceIdentity   string              `json:"surface_identity"`
	Facts             []Fact              `json:"facts,omitempty"`
	Exclusions        []Exclusion         `json:"exclusions,omitempty"`
	Unknown           []string            `json:"unknown,omitempty"`
	Invocations       []PlannedInvocation `json:"invocations"`
	ID                artifact.ID         `json:"-"`
}

// BindManifestPlan creates an immutable plan from already-selected checks.
func BindManifestPlan(base, candidate artifact.ID, surface Surface, impact Impact, invocations []Invocation) (ManifestPlan, error) {
	plan := ManifestPlan{
		BaseManifest: base, CandidateManifest: candidate, SurfaceIdentity: surface.Identity,
		Facts: slices.Clone(impact.Facts), Exclusions: slices.Clone(impact.Exclusions),
		Unknown: slices.Clone(surface.Unknown), Invocations: make([]PlannedInvocation, len(invocations)),
	}
	for index, invocation := range invocations {
		plan.Invocations[index] = PlannedInvocation{ID: invocation.ID, Check: invocation.Check, Matched: slices.Clone(invocation.Matched)}
	}
	canonicalizeManifestPlan(&plan)
	if err := validateManifestPlan(plan); err != nil {
		return ManifestPlan{}, err
	}
	id, err := manifestPlanID(plan)
	if err != nil {
		return ManifestPlan{}, err
	}
	plan.ID = id
	return plan, nil
}

// Validate verifies canonical plan structure and content identity.
func (p ManifestPlan) Validate() error {
	canonical := cloneManifestPlan(p)
	canonical.ID = artifact.ID{}
	canonicalizeManifestPlan(&canonical)
	if err := validateManifestPlan(canonical); err != nil {
		return err
	}
	withID := canonical
	withID.ID = p.ID
	if !reflect.DeepEqual(p, withID) {
		return errors.New("automation manifest plan: document is not canonical")
	}
	want, err := manifestPlanID(canonical)
	if err != nil {
		return err
	}
	if p.ID != want {
		return errors.New("automation manifest plan: identity mismatch")
	}
	return nil
}

func cloneManifestPlan(value ManifestPlan) ManifestPlan {
	value.Facts = slices.Clone(value.Facts)
	value.Exclusions = slices.Clone(value.Exclusions)
	value.Unknown = slices.Clone(value.Unknown)
	value.Invocations = slices.Clone(value.Invocations)
	for index := range value.Invocations {
		value.Invocations[index].Matched = slices.Clone(value.Invocations[index].Matched)
		value.Invocations[index].Check.Triggers = slices.Clone(value.Invocations[index].Check.Triggers)
		value.Invocations[index].Check.Dependencies = slices.Clone(value.Invocations[index].Check.Dependencies)
		value.Invocations[index].Check.Resources = slices.Clone(value.Invocations[index].Check.Resources)
		value.Invocations[index].Check.Ownership.Packages = slices.Clone(value.Invocations[index].Check.Ownership.Packages)
		value.Invocations[index].Check.Ownership.PackagePrefixes = slices.Clone(value.Invocations[index].Check.Ownership.PackagePrefixes)
		value.Invocations[index].Check.Ownership.Symbols = slices.Clone(value.Invocations[index].Check.Ownership.Symbols)
	}
	return value
}

func canonicalizeManifestPlan(plan *ManifestPlan) {
	slices.Sort(plan.Facts)
	plan.Facts = slices.Compact(plan.Facts)
	slices.SortFunc(plan.Exclusions, func(left, right Exclusion) int { return strings.Compare(left.Check, right.Check) })
	slices.Sort(plan.Unknown)
	plan.Unknown = slices.Compact(plan.Unknown)
	for index := range plan.Invocations {
		slices.Sort(plan.Invocations[index].Matched)
		plan.Invocations[index].Matched = slices.Compact(plan.Invocations[index].Matched)
	}
}

func validateManifestPlan(plan ManifestPlan) error {
	if plan.BaseManifest.Kind() != artifact.KindProfile || plan.CandidateManifest.Kind() != artifact.KindProfile || strings.TrimSpace(plan.SurfaceIdentity) == "" || len(plan.Invocations) == 0 {
		return errors.New("automation manifest plan: invalid authority or empty invocation set")
	}
	excluded := map[string]bool{}
	for _, exclusion := range plan.Exclusions {
		if strings.TrimSpace(exclusion.Check) == "" || strings.TrimSpace(exclusion.Reason) == "" || excluded[exclusion.Check] {
			return errors.New("automation manifest plan: invalid exclusion")
		}
		excluded[exclusion.Check] = true
	}
	done := map[string]bool{}
	for name := range excluded {
		done[name] = true
	}
	seenIDs := map[artifact.ID]bool{}
	seenNames := map[string]bool{}
	for _, invocation := range plan.Invocations {
		if invocation.ID.Kind() != artifact.KindRecipe || strings.TrimSpace(invocation.Check.Name) == "" || seenIDs[invocation.ID] || seenNames[invocation.Check.Name] {
			return errors.New("automation manifest plan: invalid or duplicate invocation")
		}
		for _, dependency := range invocation.Check.Dependencies {
			if !done[dependency] {
				return fmt.Errorf("automation manifest plan: invocation %s precedes dependency %s", invocation.Check.Name, dependency)
			}
		}
		seenIDs[invocation.ID] = true
		seenNames[invocation.Check.Name] = true
		done[invocation.Check.Name] = true
	}
	return nil
}

func manifestPlanID(plan ManifestPlan) (artifact.ID, error) {
	plan.ID = artifact.ID{}
	return artifact.JSONID(artifact.KindRecipe, plan)
}
