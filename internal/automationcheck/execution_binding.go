package automationcheck

import (
	"errors"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// ExecutionAuthority is the immutable lineage attached to a planned
// invocation and every terminal result it produces.
type ExecutionAuthority struct {
	Plan       artifact.ID   `json:"plan"`
	Definition artifact.ID   `json:"definition"`
	Inputs     []artifact.ID `json:"inputs"`
	Reuse      *ReuseBinding `json:"reuse,omitempty"`
}

// ReuseBinding fixes the roles of a computation's input and environment.
type ReuseBinding struct {
	Input       artifact.ID `json:"input"`
	Environment artifact.ID `json:"environment"`
}

func (binding ReuseBinding) valid() bool {
	return binding.Input.Valid() && binding.Environment.Valid()
}

func (binding ReuseBinding) matches(authority *ExecutionAuthority) bool {
	return binding.valid() && authority != nil && authority.Reuse != nil && *authority.Reuse == binding
}

// ReuseSource is the original execution a reused result stands on: its
// evidence, the invocation and authority that produced it, the definition it
// executed, and the environment and input it was recorded under.
type ReuseSource struct {
	Evidence     artifact.ID         `json:"evidence"`
	InvocationID artifact.ID         `json:"invocation_id"`
	Authority    *ExecutionAuthority `json:"authority,omitempty"`
	Definition   artifact.ID         `json:"definition"`
	Environment  artifact.ID         `json:"environment,omitzero"`
	Input        artifact.ID         `json:"input"`
	Detail       string              `json:"detail,omitzero"`
}

// complete verifies the cited result was an original passing execution.
func (source *ReuseSource) complete() bool {
	if source == nil || source.Evidence.Kind() != artifact.KindEvidence ||
		!source.InvocationID.Valid() || !source.Definition.Valid() || !source.Input.Valid() || !source.Environment.Valid() {
		return false
	}
	original := Evidence{
		ID: source.Evidence, InvocationID: source.InvocationID, Authority: source.Authority,
		Outcome: runrecord.LanePassed, Detail: source.Detail,
	}
	if source.Authority != nil && !(ReuseBinding{Input: source.Input, Environment: source.Environment}).matches(source.Authority) {
		return false
	}
	return executionDefinition(source.Authority, source.InvocationID) == source.Definition && original.VerifyIdentity() == nil
}

// BindManifestExecution turns one definition-level invocation into work
// authorized by an exact manifest plan and explicit relevant inputs.
func BindManifestExecution(plan ManifestPlan, invocation Invocation, inputs []artifact.ID, reuse *ReuseBinding) (Invocation, error) {
	if err := plan.Validate(); err != nil {
		return Invocation{}, err
	}
	if invocation.Authority != nil || !invocation.ID.Valid() || len(inputs) == 0 {
		return Invocation{}, errors.New("automation check: execution requires one unbound invocation and relevant inputs")
	}
	if reuse != nil && !reuse.valid() {
		return Invocation{}, errors.New("automation check: reuse binding requires input and environment")
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
	authority := cloneExecutionAuthority(&ExecutionAuthority{Plan: plan.ID, Definition: definition, Inputs: boundInputs, Reuse: reuse})
	id, err := artifact.JSONID(artifact.KindRecipe, struct {
		Plan       artifact.ID   `json:"plan"`
		Definition artifact.ID   `json:"definition"`
		Inputs     []artifact.ID `json:"inputs"`
		Reuse      *ReuseBinding `json:"reuse,omitempty"`
	}{plan.ID, definition, boundInputs, authority.Reuse})
	if err != nil {
		return Invocation{}, err
	}
	invocation.ID = id
	invocation.Authority = authority
	return invocation, nil
}

// EvidenceLineage exposes the immutable authorities carried by a terminal
// check result. The plan binds candidate tree, source, manifests, impact, and
// definitions; inputs bind the execution-specific resources; a reused result
// also carries the original execution it stands on.
func EvidenceLineage(evidence Evidence) []artifact.ID {
	if evidence.Authority == nil {
		return nil
	}
	var parents []artifact.ID
	authorities := []*ExecutionAuthority{evidence.Authority}
	if source := evidence.Source; evidence.Reused && source != nil {
		parents = append(parents, source.Evidence, source.InvocationID, source.Definition)
		authorities = append(authorities, source.Authority)
	}
	for _, authority := range authorities {
		if authority == nil {
			continue
		}
		parents = append(parents, authority.Plan, authority.Definition)
		parents = append(parents, authority.Inputs...)
		if authority.Reuse != nil {
			parents = append(parents, authority.Reuse.Input, authority.Reuse.Environment)
		}
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(parents, func(left, right artifact.ID) int { return strings.Compare(left.String(), right.String()) })
	return slices.Compact(parents)
}

// ValidateReuseAuthority admits a reused result for one planned definition:
// its own identity holds, its successor authority names that definition, and
// it stands on a complete original execution of the same definition that
// passed under a bound plan of its own.
func ValidateReuseAuthority(evidence Evidence, definition artifact.ID) error {
	if !evidence.Reused {
		return errors.New("automation check: evidence is not a reused result")
	}
	if evidence.Outcome != runrecord.LanePassed || evidence.Inapplicable {
		return errors.New("automation check: reuse requires a passing applicable result")
	}
	if err := evidence.VerifyIdentity(); err != nil {
		return err
	}
	if evidence.Authority == nil || evidence.Authority.Definition != definition {
		return errors.New("automation check: reuse authority names another definition")
	}
	source := evidence.Source
	if !source.complete() {
		return errors.New("automation check: reused evidence lacks its original execution")
	}
	if source.Authority == nil || !source.Authority.Plan.Valid() ||
		source.Authority.Definition != definition || source.Definition != definition {
		return errors.New("automation check: reused evidence stands on another definition or an unbound execution")
	}
	if !(ReuseBinding{Input: source.Input, Environment: source.Environment}).matches(evidence.Authority) {
		return errors.New("automation check: reuse inputs differ from successor execution")
	}
	return nil
}

// Bound authority names the definition; unbound execution names itself.
func executionDefinition(authority *ExecutionAuthority, invocation artifact.ID) artifact.ID {
	if authority != nil {
		return authority.Definition
	}
	return invocation
}

func cloneExecutionAuthority(authority *ExecutionAuthority) *ExecutionAuthority {
	if authority == nil {
		return nil
	}
	cloned := *authority
	cloned.Inputs = slices.Clone(authority.Inputs)
	if authority.Reuse != nil {
		binding := *authority.Reuse
		cloned.Reuse = &binding
	}
	return &cloned
}

func cloneReuseSource(source *ReuseSource) *ReuseSource {
	if source == nil {
		return nil
	}
	cloned := *source
	cloned.Authority = cloneExecutionAuthority(source.Authority)
	return &cloned
}
