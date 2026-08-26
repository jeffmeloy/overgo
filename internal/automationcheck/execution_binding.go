package automationcheck

import (
	"errors"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// ExecutionAuthority is the immutable lineage attached to a planned
// invocation and every terminal result it produces.
type ExecutionAuthority struct {
	Plan       artifact.ID   `json:"plan"`
	Definition artifact.ID   `json:"definition"`
	Inputs     []artifact.ID `json:"inputs"`
}

// BindManifestExecution turns one definition-level invocation into work
// authorized by an exact manifest plan and explicit relevant inputs.
func BindManifestExecution(plan ManifestPlan, invocation Invocation, inputs []artifact.ID) (Invocation, error) {
	if err := plan.Validate(); err != nil {
		return Invocation{}, err
	}
	if invocation.Authority != nil || !invocation.ID.Valid() || len(inputs) == 0 {
		return Invocation{}, errors.New("automation check: execution requires one unbound invocation and relevant inputs")
	}
	found := false
	for _, planned := range plan.Invocations {
		if planned.ID == invocation.ID && reflect.DeepEqual(planned.Check, invocation.Check) && reflect.DeepEqual(planned.Matched, invocation.Matched) {
			found = true
			break
		}
	}
	if !found {
		return Invocation{}, errors.New("automation check: invocation is absent from manifest plan")
	}
	boundInputs := slices.Clone(inputs)
	for _, input := range boundInputs {
		if !input.Valid() {
			return Invocation{}, errors.New("automation check: execution input identity is invalid")
		}
	}
	slices.SortFunc(boundInputs, func(left, right artifact.ID) int { return strings.Compare(left.String(), right.String()) })
	boundInputs = slices.Compact(boundInputs)
	definition := invocation.ID
	id, err := artifact.JSONID(artifact.KindRecipe, struct {
		Plan       artifact.ID   `json:"plan"`
		Definition artifact.ID   `json:"definition"`
		Inputs     []artifact.ID `json:"inputs"`
	}{plan.ID, definition, boundInputs})
	if err != nil {
		return Invocation{}, err
	}
	invocation.ID = id
	invocation.Authority = &ExecutionAuthority{Plan: plan.ID, Definition: definition, Inputs: boundInputs}
	return invocation, nil
}

// EvidenceLineage exposes the immutable authorities carried by a terminal
// check result. The plan binds candidate tree, source, manifests, impact, and
// definitions; inputs bind the execution-specific resources.
func EvidenceLineage(evidence Evidence) []artifact.ID {
	if evidence.Authority == nil {
		return nil
	}
	parents := []artifact.ID{evidence.Authority.Plan, evidence.Authority.Definition}
	parents = append(parents, evidence.Authority.Inputs...)
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(parents, func(left, right artifact.ID) int { return strings.Compare(left.String(), right.String()) })
	return slices.Compact(parents)
}

func cloneExecutionAuthority(authority *ExecutionAuthority) *ExecutionAuthority {
	if authority == nil {
		return nil
	}
	cloned := *authority
	cloned.Inputs = slices.Clone(authority.Inputs)
	return &cloned
}
