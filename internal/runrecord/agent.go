package runrecord

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/textcheck"
)

const (
	// AgentActivationMediaType identifies immutable agent lifecycle evidence.
	AgentActivationMediaType = "application/vnd.overgo.agent-activation+json"
	// AgentActivationSchema identifies the agent lifecycle contract.
	AgentActivationSchema = "overgo/agent-activation/v1"
	// AgentActiveAliasRoot scopes current activation authority by agent name.
	AgentActiveAliasRoot = "agent/active/"
	// AgentAuthorityAliasRoot prevents one decision from authorizing multiple transitions.
	AgentAuthorityAliasRoot = "agent/authority/"
)

// AgentState controls admission without mutating an agent definition.
type AgentState string

const (
	// AgentActive admits new sessions and steps.
	AgentActive AgentState = "active"
	// AgentPaused refuses new work until an immutable resume transition.
	AgentPaused AgentState = "paused"
)

// AgentActivation records one exact definition and pause/resume transition.
type AgentActivation struct {
	Version    uint16      `json:"version"`
	Name       string      `json:"name"`
	Definition artifact.ID `json:"definition"`
	State      AgentState  `json:"state"`
	Authority  artifact.ID `json:"authority"`
	Prior      artifact.ID `json:"prior,omitzero"`
	ID         artifact.ID `json:"-"`
}

// ActiveAgent joins the current lifecycle event and immutable definition.
type ActiveAgent struct {
	Definition recipe.AgentDefinition `json:"definition"`
	Activation AgentActivation        `json:"activation"`
}

// AgentAuthority owns active aliases and immutable pause/resume transitions.
type AgentAuthority struct {
	Repository artifact.Repository
}

var agentActivationCodec = artifact.JSONDocumentCodec(
	"agent activation", artifact.KindEvidence, AgentActivationMediaType, AgentActivationSchema,
	canonicalizeAgentActivation,
	func(value AgentActivation) artifact.ID { return value.ID },
	func(value *AgentActivation, id artifact.ID) { value.ID = id }, nil,
)

// Resolve returns current agent authority, including a paused state.
func (authority AgentAuthority) Resolve(ctx context.Context, name string) (ActiveAgent, bool, error) {
	if ctx == nil || authority.Repository == nil || !textcheck.LowerIdentifier(name, len(name)) {
		return ActiveAgent{}, false, errors.New("run record: invalid agent resolution")
	}
	id, found, err := artifact.ResolveAlias(ctx, authority.Repository, AgentActiveAliasRoot+name)
	if err != nil || !found {
		return ActiveAgent{}, found, err
	}
	activation, err := agentActivationCodec.Require(ctx, authority.Repository, id)
	if err != nil {
		return ActiveAgent{}, false, err
	}
	definition, err := recipe.RequireAgentDefinition(ctx, authority.Repository, activation.Definition)
	if err != nil || activation.Name != definition.Name || activation.Name != name {
		return ActiveAgent{}, false, errors.Join(errors.New("run record: agent activation subject differs"), err)
	}
	return ActiveAgent{Definition: definition, Activation: activation}, true, nil
}

// RequireActive refuses absent and paused agent definitions.
func (authority AgentAuthority) RequireActive(ctx context.Context, name string) (ActiveAgent, error) {
	active, found, err := authority.Resolve(ctx, name)
	if err != nil || !found || active.Activation.State != AgentActive {
		return ActiveAgent{}, errors.Join(errors.New("run record: active agent is unavailable"), err)
	}
	return active, nil
}

// Activate publishes a new definition or replaces the current one with exact authority evidence.
func (authority AgentAuthority) Activate(ctx context.Context, key string, definition recipe.AgentDefinition, decision artifact.ID) (ActiveAgent, error) {
	identified, identifyErr := recipe.NewAgentDefinition(definition)
	if ctx == nil || authority.Repository == nil || key == "" || decision.Kind() != artifact.KindEvidence ||
		definition.ValidateIdentity() != nil || identifyErr != nil || identified.ID != definition.ID {
		return ActiveAgent{}, errors.New("run record: invalid agent activation")
	}
	current, found, err := authority.Resolve(ctx, definition.Name)
	if err != nil {
		return ActiveAgent{}, err
	}
	prior := artifact.ID{}
	if found {
		prior = current.Activation.ID
	}
	return authority.publish(ctx, key, definition, AgentActive, decision, prior)
}

// Pause refuses new work while retaining exact active definition authority.
func (authority AgentAuthority) Pause(ctx context.Context, key, name string, decision artifact.ID) (ActiveAgent, error) {
	return authority.transition(ctx, key, name, AgentPaused, decision)
}

// Resume reactivates the same immutable definition.
func (authority AgentAuthority) Resume(ctx context.Context, key, name string, decision artifact.ID) (ActiveAgent, error) {
	return authority.transition(ctx, key, name, AgentActive, decision)
}

func (authority AgentAuthority) transition(ctx context.Context, key, name string, state AgentState, decision artifact.ID) (ActiveAgent, error) {
	current, found, err := authority.Resolve(ctx, name)
	if err != nil || !found || key == "" || decision.Kind() != artifact.KindEvidence ||
		current.Activation.State == state || state == AgentPaused && current.Activation.State != AgentActive ||
		state == AgentActive && current.Activation.State != AgentPaused {
		return ActiveAgent{}, errors.Join(errors.New("run record: invalid agent state transition"), err)
	}
	return authority.publish(ctx, key, current.Definition, state, decision, current.Activation.ID)
}

func (authority AgentAuthority) publish(
	ctx context.Context,
	key string,
	definition recipe.AgentDefinition,
	state AgentState,
	decision, prior artifact.ID,
) (ActiveAgent, error) {
	activation, err := agentActivationCodec.New(AgentActivation{
		Version: artifact.InitialDocumentVersion, Name: definition.Name, Definition: definition.ID,
		State: state, Authority: decision, Prior: prior,
	})
	if err != nil {
		return ActiveAgent{}, err
	}
	current, found, err := authority.Resolve(ctx, definition.Name)
	if err != nil || found != prior.Valid() || found && current.Activation.ID != prior {
		return ActiveAgent{}, errors.Join(errors.New("run record: agent activation chain differs"), err)
	}
	if _, used, err := artifact.ResolveAlias(ctx, authority.Repository, AgentAuthorityAliasRoot+decision.String()); err != nil || used {
		return ActiveAgent{}, errors.Join(errors.New("run record: agent lifecycle authority was already consumed"), err)
	}
	definitionContent, err := definition.ArtifactContent()
	if err != nil {
		return ActiveAgent{}, err
	}
	activationContent, err := agentActivationCodec.Content(activation)
	if err != nil {
		return ActiveAgent{}, err
	}
	lineage := definition.Lineage()
	parents := []artifact.ID{definition.ID, decision}
	if prior.Valid() {
		parents = append(parents, prior)
	}
	lineage = append(lineage, artifact.DependencyLineage(activation.ID, parents...)...)
	alias := artifact.AliasBinding{Name: AgentActiveAliasRoot + definition.Name, Target: activation.ID}
	if prior.Valid() {
		alias.Previous = artifact.IDPointer(prior)
	}
	batch, err := artifact.NewDocumentBatch(key, []artifact.Content{definitionContent, activationContent}, lineage, []artifact.AliasBinding{
		alias, {Name: AgentAuthorityAliasRoot + decision.String(), Target: activation.ID},
	})
	if err != nil {
		return ActiveAgent{}, err
	}
	if _, err := artifact.CommitBatch(ctx, authority.Repository, batch); err != nil {
		return ActiveAgent{}, err
	}
	return ActiveAgent{Definition: definition, Activation: activation}, nil
}

func canonicalizeAgentActivation(value *AgentActivation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !textcheck.LowerIdentifier(value.Name, len(value.Name)) ||
		value.Definition.Kind() != artifact.KindRecipe || value.Authority.Kind() != artifact.KindEvidence ||
		(value.State != AgentActive && value.State != AgentPaused) ||
		value.Prior.Valid() && value.Prior.Kind() != artifact.KindEvidence || value.Prior == value.Authority {
		return errors.New("run record: invalid agent activation")
	}
	return nil
}
