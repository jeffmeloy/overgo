// Package invocation defines the transport-neutral facts shared by capability
// admission, approval, execution evidence, and durable receipts. It deliberately
// owns no transport, registry, policy, or execution path.
package invocation

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// EffectMediaType identifies one resolved invocation effect.
	EffectMediaType = "application/vnd.overgo.invocation-effect+json"
	// EffectSchema identifies the transport-neutral effect contract.
	EffectSchema = "overgo/invocation-effect/v1"
)

// Class distinguishes observation from mutation. It is policy input, never a
// grant: a mutation still needs an action-bound preflight and authority.
type Class string

const (
	// ClassInspection identifies an observation-only invocation.
	ClassInspection Class = "inspection"
	// ClassMutation identifies an invocation that may change state.
	ClassMutation Class = "mutation"
)

// Valid reports whether class belongs to the closed invocation vocabulary.
func (class Class) Valid() bool {
	return slices.Contains([]Class{ClassInspection, ClassMutation}, class)
}

// Scope names the authority boundary containing an invocation target.
type Scope string

const (
	// ScopeWorkspace identifies an exact path within the active workspace.
	ScopeWorkspace Scope = "workspace"
	// ScopeRepository identifies an exact repository authority boundary.
	ScopeRepository Scope = "repository"
	// ScopeHost identifies an exact host-local authority boundary.
	ScopeHost Scope = "host"
	// ScopeExternal identifies an exact authority boundary outside the host.
	ScopeExternal Scope = "external"
)

// Valid reports whether scope belongs to the closed target vocabulary.
func (scope Scope) Valid() bool {
	return slices.Contains([]Scope{ScopeWorkspace, ScopeRepository, ScopeHost, ScopeExternal}, scope)
}

// Target is one exact authority boundary touched by a concrete invocation.
type Target struct {
	Scope Scope  `json:"scope"`
	Value string `json:"value"`
}

// Effect is the canonical concrete effect shared by admission, approval,
// execution evidence, trajectories, and policy. Unknown writers are opaque.
type Effect struct {
	ID             artifact.ID  `json:"-"`
	Manual         artifact.ID  `json:"manual"`
	Arguments      artifact.ID  `json:"arguments"`
	ObservedResult *artifact.ID `json:"observed_result,omitempty"`
	Class          Class        `json:"class"`
	Targets        []Target     `json:"targets,omitempty"`
	Known          bool         `json:"known"`
	OpaqueMutation bool         `json:"opaque_mutation,omitzero"`
	Network        bool         `json:"network,omitzero"`
	Executable     bool         `json:"executable,omitzero"`
	Destructive    bool         `json:"destructive,omitzero"`
	Privileged     bool         `json:"privileged,omitzero"`
	Irreversible   bool         `json:"irreversible,omitzero"`
}

var effectCodec = artifact.JSONDocumentCodec(
	"invocation effect", artifact.KindEvidence, EffectMediaType, EffectSchema,
	canonicalizeEffect,
	func(value Effect) artifact.ID { return value.ID },
	func(value *Effect, id artifact.ID) { value.ID = id },
	cloneEffect,
)

// NewEffect validates and content-addresses a resolved effect.
func NewEffect(value Effect) (Effect, error) {
	return effectCodec.New(effectWithoutIdentity(value))
}

func effectWithoutIdentity(value Effect) Effect {
	value.ID = artifact.ID{}
	return value
}

// RequireEffect loads one exact effect from an artifact reader.
func RequireEffect(ctx context.Context, reader artifact.Reader, id artifact.ID) (Effect, error) {
	return effectCodec.Require(ctx, reader, id)
}

// Content returns the canonical effect document.
func (value Effect) Content() (artifact.Content, error) { return effectCodec.Content(value) }

// ValidateIdentity verifies that ID identifies exactly these effect facts.
func (value Effect) ValidateIdentity() error { return effectCodec.ValidateIdentity(value) }

// Lineage binds the effect to its manual, arguments, and optional result.
func (value Effect) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Manual, value.Arguments}
	if value.ObservedResult != nil {
		parents = append(parents, *value.ObservedResult)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

// InspectionCovers reports whether an exact inspection is relevant to every
// mutation target. Equality is sufficient for non-hierarchical scopes;
// workspace paths may cover descendants but never siblings.
func InspectionCovers(inspection, mutation Effect) bool {
	if inspection.Class != ClassInspection || mutation.Class != ClassMutation ||
		inspection.ValidateIdentity() != nil || mutation.ValidateIdentity() != nil ||
		!mutation.Known || mutation.OpaqueMutation || len(mutation.Targets) == 0 {
		return false
	}
	return !slices.ContainsFunc(mutation.Targets, func(target Target) bool {
		return !slices.ContainsFunc(inspection.Targets, func(observed Target) bool {
			if observed.Scope != target.Scope {
				return false
			}
			if observed.Value == target.Value {
				return true
			}
			return target.Scope == ScopeWorkspace &&
				strings.HasPrefix(strings.TrimSuffix(target.Value, "/"), strings.TrimSuffix(observed.Value, "/")+"/")
		})
	})
}

func canonicalizeEffect(value *Effect) error {
	if value == nil || value.Manual.Kind() != artifact.KindRecipe || value.Arguments.Kind() != artifact.KindEvidence ||
		!value.Class.Valid() || value.ObservedResult != nil && value.ObservedResult.Kind() != artifact.KindEvidence {
		return errors.New("invocation: invalid effect authority")
	}
	for _, target := range value.Targets {
		if !target.Scope.Valid() || strings.TrimSpace(target.Value) == "" || target.Value != strings.TrimSpace(target.Value) {
			return errors.New("invocation: invalid effect target")
		}
	}
	slices.SortFunc(value.Targets, func(left, right Target) int {
		if order := strings.Compare(string(left.Scope), string(right.Scope)); order != 0 {
			return order
		}
		return strings.Compare(left.Value, right.Value)
	})
	value.Targets = slices.Compact(value.Targets)
	if value.Class == ClassInspection && (!value.Known || value.OpaqueMutation || value.Destructive || value.Privileged || value.Irreversible) {
		return errors.New("invocation: inspection cannot carry mutation authority")
	}
	if value.Class == ClassMutation && value.OpaqueMutation == value.Known {
		return errors.New("invocation: mutation must be exactly known or explicitly opaque")
	}
	return nil
}

func cloneEffect(value Effect) Effect {
	value.Targets = slices.Clone(value.Targets)
	if value.ObservedResult != nil {
		result := *value.ObservedResult
		value.ObservedResult = &result
	}
	return value
}
