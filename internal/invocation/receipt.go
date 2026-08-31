package invocation

import (
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

// Boundary classifies where invocation resolution and transport occur. An
// internal action is still receipted, but deterministic orchestration calls its
// Go owner directly; it never resolves a UTCP transport.
type Boundary string

const (
	// BoundaryInternal identifies a direct call to a compiled Go owner.
	BoundaryInternal Boundary = "internal-go"
	// BoundaryAgent identifies an agent-initiated capability invocation.
	BoundaryAgent Boundary = "agent"
	// BoundaryPeer identifies a capability invocation across an Overgo peer boundary.
	BoundaryPeer Boundary = "peer"
	// BoundaryExternal identifies a capability invocation across an external boundary.
	BoundaryExternal Boundary = "external"
)

// Valid reports whether boundary belongs to the closed invocation vocabulary.
func (boundary Boundary) Valid() bool {
	return slices.Contains([]Boundary{BoundaryInternal, BoundaryAgent, BoundaryPeer, BoundaryExternal}, boundary)
}

// MutationKind is the closed RSI lifecycle vocabulary plus the exact generic
// tool action used at agent/cross-boundary capability seams.
type MutationKind string

const (
	// MutationTool identifies one generic agent or cross-boundary tool mutation.
	MutationTool MutationKind = "tool"
	// MutationTrainingLaunch identifies a training-launch lifecycle mutation.
	MutationTrainingLaunch MutationKind = "training-launch"
	// MutationPromotion identifies a promotion lifecycle mutation.
	MutationPromotion MutationKind = "promotion"
	// MutationModelActivation identifies a model-activation lifecycle mutation.
	MutationModelActivation MutationKind = "model-activation"
	// MutationContainment identifies a containment lifecycle mutation.
	MutationContainment MutationKind = "containment"
	// MutationRetry identifies a retry lifecycle mutation.
	MutationRetry MutationKind = "retry"
	// MutationRollback identifies a rollback lifecycle mutation.
	MutationRollback MutationKind = "rollback"
)

// Valid reports whether kind belongs to the closed RSI mutation vocabulary.
func (kind MutationKind) Valid() bool {
	switch kind {
	case MutationTool, MutationTrainingLaunch, MutationPromotion, MutationModelActivation,
		MutationContainment, MutationRetry, MutationRollback:
		return true
	}
	return false
}

// ReceiptBinding is the immutable authority carried by every state in one
// mutation receipt chain. Authority is exactly one HumanDecision today or an
// active automation-policy lifecycle record when unattended authority lands.
type ReceiptBinding struct {
	Boundary      Boundary          `json:"boundary"`
	Kind          MutationKind      `json:"kind"`
	Action        string            `json:"action"`
	Subject       artifact.ID       `json:"subject"`
	Arguments     artifact.ID       `json:"arguments"`
	Effect        artifact.ID       `json:"effect"`
	Preflight     artifact.ID       `json:"preflight"`
	Inspection    artifact.ID       `json:"inspection"`
	Ceiling       artifact.ID       `json:"ceiling"`
	Authority     artifact.ID       `json:"authority"`
	CausalContext artifact.ID       `json:"causal_context"`
	Head          artifact.CommitID `json:"head"`
}

// Validate refuses incomplete or ambiguous mutation authority.
func (binding ReceiptBinding) Validate() error {
	if !binding.Boundary.Valid() || !binding.Kind.Valid() || strings.TrimSpace(binding.Action) == "" ||
		binding.Action != strings.TrimSpace(binding.Action) || binding.Subject.Kind() != artifact.KindRecipe ||
		binding.Arguments.Kind() != artifact.KindEvidence || binding.Effect.Kind() != artifact.KindEvidence ||
		binding.Preflight.Kind() != artifact.KindEvidence || binding.Inspection.Kind() != artifact.KindEvidence ||
		binding.Ceiling.Kind() != artifact.KindRecipe || binding.Authority.Kind() != artifact.KindEvidence ||
		binding.CausalContext.Kind() != artifact.KindEvidence || !binding.Head.Valid() {
		return errors.New("invocation: incomplete mutation receipt binding")
	}
	return nil
}

// Parents returns the complete immutable authority set for lineage.
func (binding ReceiptBinding) Parents() []artifact.ID {
	return []artifact.ID{binding.Subject, binding.Arguments, binding.Effect, binding.Preflight,
		binding.Inspection, binding.Ceiling, binding.Authority, binding.CausalContext}
}

// ApprovalArguments is the canonical immutable fact vector a HumanDecision
// grants for one mutation. It is shared by agent-facing and direct-Go callers.
func ApprovalArguments(subject, arguments, effect, preflight artifact.ID) []string {
	return []string{subject.String(), arguments.String(), effect.String(), preflight.String()}
}
