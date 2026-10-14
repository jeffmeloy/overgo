package agentloop

import (
	"context"
	"encoding/json"
	"errors"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// DelegatedCoordinator applies an invocation's exact manual grant at every
// proposal while retaining the coordinator's normal approval requirements.
type DelegatedCoordinator struct {
	coordinator *Coordinator
	manuals     []artifact.ID
}

// NewDelegatedCoordinator builds a coordinator bound to the invocation's
// recipe, model, and step budget, requiring a non-empty manual grant.
func NewDelegatedCoordinator(store artifact.Repository, executor *agenttool.Executor, invocation recipe.DelegatedAgentInvocation, node recipe.NodeID) (*DelegatedCoordinator, error) {
	if err := invocation.ValidateIdentity(); err != nil {
		return nil, err
	}
	coordinator, err := New(store, executor, Identity{Recipe: invocation.ID, Model: invocation.Model, Node: node}, invocation.Scheduler.MaxSteps)
	if err != nil {
		return nil, err
	}
	if len(invocation.Grant.Manuals) == 0 {
		return nil, errors.New("agent loop: delegated manual grant is empty")
	}
	return &DelegatedCoordinator{coordinator: coordinator, manuals: append([]artifact.ID(nil), invocation.Grant.Manuals...)}, nil
}

// Propose forwards one tool proposal through the coordinator restricted to the
// delegated manual grant.
func (delegated *DelegatedCoordinator) Propose(ctx context.Context, session *Session, name string, arguments json.RawMessage) (json.RawMessage, error) {
	if delegated == nil {
		return nil, errors.New("agent loop: delegated coordinator is absent")
	}
	session.Ceiling = delegated.coordinator.identity.Recipe
	return delegated.coordinator.ProposeWithManuals(ctx, session, name, arguments, delegated.manuals)
}

// PreviewMutationDecision projects the decision facts of the session's next step through the coordinator.
//   - it publishes and executes nothing.
func (delegated *DelegatedCoordinator) PreviewMutationDecision(ctx context.Context, session *Session, name string, arguments json.RawMessage) (DecisionPreview, error) {
	if delegated == nil {
		return DecisionPreview{}, errors.New("agent loop: delegated coordinator is absent")
	}
	return delegated.coordinator.PreviewMutationDecision(ctx, session, name, arguments)
}

// ApproveMutation records the operator's grant for the session's next step through the coordinator.
//   - the session's ceiling = the invocation, as Propose binds it, because Propose refuses a preflight that binds another ceiling.
func (delegated *DelegatedCoordinator) ApproveMutation(ctx context.Context, session *Session, name string, arguments json.RawMessage, approved artifact.ID) (artifact.ID, error) {
	if delegated == nil || session == nil {
		return artifact.ID{}, errors.New("agent loop: delegated coordinator or session is absent")
	}
	session.Ceiling = delegated.coordinator.identity.Recipe
	return delegated.coordinator.ApproveMutation(ctx, session, name, arguments, approved)
}
