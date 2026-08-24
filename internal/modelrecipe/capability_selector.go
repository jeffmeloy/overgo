package modelrecipe

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

type SessionSelection string

const (
	SessionPin       SessionSelection = "pin"
	SessionWarm      SessionSelection = "warm"
	SessionSpillover SessionSelection = "spillover"
)

func (selection SessionSelection) Valid() bool {
	return selection == SessionPin || selection == SessionWarm || selection == SessionSpillover
}

// CapabilityEvidenceSelector defines typed model alias plus serving intent.
type CapabilityEvidenceSelector struct {
	Alias         string
	Task          recipe.Task
	Session       SessionSelection
	Compatibility artifact.ID
}

// CapabilityEvidenceSelection defines verified executable alias target.
type CapabilityEvidenceSelection struct {
	Identity   artifact.ID
	Scope      artifact.ID
	Alias      string
	Session    SessionSelection
	Activation Activation
	Program    recipe.Program
	Resources  ComponentSessionPlan
	Bundles    []CapabilityBundle
	Peer       RemotePeerCompatibility
}

type capabilitySelectionIdentity struct {
	Scope     artifact.ID      `json:"scope"`
	Alias     string           `json:"alias"`
	Session   SessionSelection `json:"session"`
	Model     artifact.ID      `json:"model"`
	Recipe    artifact.ID      `json:"recipe"`
	Resources artifact.ID      `json:"resources"`
	Bundles   []artifact.ID    `json:"bundles,omitempty"`
	Evidence  *artifact.ID     `json:"evidence,omitempty"`
	Peer      *artifact.ID     `json:"peer,omitempty"`
}

type capabilityScopeIdentity struct {
	Alias string      `json:"alias,omitempty"`
	Model artifact.ID `json:"model"`
	Task  recipe.Task `json:"task"`
}

// CompileCandidateExecution binds one immutable candidate program for verification.
func CompileCandidateExecution(
	ctx context.Context,
	store artifact.Reader,
	program recipe.Program,
) (CapabilityEvidenceSelection, error) {
	resources, err := CompileComponentSessionPlan(ctx, store, program)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	return compileCapabilitySelection("", SessionWarm, Activation{Definition: program.Definition()}, program, resources, nil, RemotePeerCompatibility{})
}

// ResolveActiveExecution resolves one model's active local execution.
func ResolveActiveExecution(
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
	task recipe.Task,
	session SessionSelection,
) (CapabilityEvidenceSelection, error) {
	if session == SessionSpillover || !session.Valid() {
		return CapabilityEvidenceSelection{}, errors.New("model recipe: active local execution has invalid session selection")
	}
	activation, program, err := resolveActiveCapability(ctx, store, modelID, task)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	resources, err := CompileComponentSessionPlan(ctx, store, program)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	bundles, err := resolveCapabilityBundles(ctx, store, activation)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	return compileCapabilitySelection("", session, activation, program, resources, bundles, RemotePeerCompatibility{})
}

// ResolveCapabilityEvidenceSelector resolves OvergoDB alias, recipe, evidence, and resources.
func ResolveCapabilityEvidenceSelector(
	ctx context.Context,
	store artifact.Reader,
	selector CapabilityEvidenceSelector,
) (CapabilityEvidenceSelection, error) {
	if ctx == nil || store == nil || strings.TrimSpace(selector.Alias) != selector.Alias ||
		selector.Alias == "" || !selector.Task.Valid() || !selector.Session.Valid() {
		return CapabilityEvidenceSelection{}, errors.New("model recipe: invalid capability evidence selector")
	}
	modelID, found, err := artifact.ResolveAlias(ctx, store, selector.Alias)
	if err != nil || !found || modelID.Kind() != artifact.KindModel {
		return CapabilityEvidenceSelection{}, errors.Join(errors.New("model recipe: capability alias has no model"), err)
	}
	activation, program, err := resolveActiveCapability(ctx, store, modelID, selector.Task)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	resources, err := CompileComponentSessionPlan(ctx, store, program)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	bundles, err := resolveCapabilityBundles(ctx, store, activation)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	if selector.Session != SessionSpillover && resources.RequestScoped() {
		return CapabilityEvidenceSelection{}, errors.New("model recipe: retained selector requires capacity session")
	}
	var peer RemotePeerCompatibility
	if selector.Session == SessionSpillover {
		if !selector.Compatibility.Valid() {
			return CapabilityEvidenceSelection{}, errors.New("model recipe: spillover requires compatibility evidence")
		}
		peer, err = resolveRemotePeerCompatibility(
			ctx, store, selector.Compatibility, modelID, activation.Definition.ID, resources.Identity, selector.Task,
		)
		if err != nil {
			return CapabilityEvidenceSelection{}, err
		}
	} else if selector.Compatibility.Valid() {
		return CapabilityEvidenceSelection{}, errors.New("model recipe: local selection carries remote compatibility")
	}
	return compileCapabilitySelection(selector.Alias, selector.Session, activation, program, resources, bundles, peer)
}

// RefreshCapabilityExecution returns current authority or rejects incompatible replacement.
func RefreshCapabilityExecution(
	ctx context.Context,
	store artifact.Reader,
	selection CapabilityEvidenceSelection,
) (CapabilityEvidenceSelection, error) {
	if ctx == nil || store == nil || !selection.Identity.Valid() || !selection.Scope.Valid() {
		return CapabilityEvidenceSelection{}, errors.New("model recipe: invalid capability execution authority")
	}
	validated, err := compileCapabilitySelection(
		selection.Alias, selection.Session, selection.Activation,
		selection.Program, selection.Resources, selection.Bundles, selection.Peer,
	)
	if err != nil || validated.Identity != selection.Identity {
		return CapabilityEvidenceSelection{}, errors.Join(errors.New("model recipe: capability execution identity differs"), err)
	}
	if !selection.Activation.Event.ID.Valid() {
		return selection, nil
	}
	current, err := capabilityExecutionCurrent(ctx, store, selection)
	if err != nil || current {
		return selection, err
	}
	refreshed, err := currentCapabilityExecution(ctx, store, selection)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	if !sameCapabilityExecution(selection, refreshed) {
		return CapabilityEvidenceSelection{}, errors.New("model recipe: capability execution authority changed")
	}
	return refreshed, nil
}

func capabilityExecutionCurrent(
	ctx context.Context,
	store artifact.Reader,
	selection CapabilityEvidenceSelection,
) (bool, error) {
	definition := selection.Program.Definition()
	if selection.Alias != "" {
		modelID, found, err := artifact.ResolveAlias(ctx, store, selection.Alias)
		if err != nil || !found || modelID != definition.Model {
			return false, err
		}
	}
	recipeID, found, err := artifact.ResolveAlias(ctx, store, activeAlias(definition.Model, definition.Task))
	if err != nil || !found || recipeID != definition.ID {
		return false, err
	}
	eventID, found, err := artifact.ResolveAlias(ctx, store, statusAlias(definition.ID))
	return err == nil && found && eventID == selection.Activation.Event.ID, err
}

func currentCapabilityExecution(
	ctx context.Context,
	store artifact.Reader,
	selection CapabilityEvidenceSelection,
) (CapabilityEvidenceSelection, error) {
	definition := selection.Program.Definition()
	if selection.Alias != "" {
		return ResolveCapabilityEvidenceSelector(ctx, store, CapabilityEvidenceSelector{
			Alias: selection.Alias, Task: definition.Task, Session: selection.Session,
			Compatibility: selection.Peer.ID,
		})
	}
	return ResolveActiveExecution(ctx, store, definition.Model, definition.Task, selection.Session)
}

func sameCapabilityExecution(left, right CapabilityEvidenceSelection) bool {
	return left.Program.Definition().Model == right.Program.Definition().Model &&
		left.Program.Definition().ID == right.Program.Definition().ID &&
		left.Resources.Identity == right.Resources.Identity && left.Session == right.Session &&
		sameCapabilityBundles(left.Bundles, right.Bundles) && left.Peer.ID == right.Peer.ID
}

func compileCapabilitySelection(
	alias string,
	session SessionSelection,
	activation Activation,
	program recipe.Program,
	resources ComponentSessionPlan,
	bundles []CapabilityBundle,
	peer RemotePeerCompatibility,
) (CapabilityEvidenceSelection, error) {
	definition := program.Definition()
	if err := activation.Definition.ValidateIdentity(); err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	if !session.Valid() || activation.Definition.ID != definition.ID ||
		activation.Definition.Model != definition.Model || activation.Definition.Task != definition.Task ||
		definition.Model.Kind() != artifact.KindModel || resources.Recipe != definition.ID ||
		resources.Identity.Kind() != artifact.KindProfile {
		return CapabilityEvidenceSelection{}, errors.New("model recipe: invalid capability execution")
	}
	if alias != "" && !activation.Event.ID.Valid() {
		return CapabilityEvidenceSelection{}, errors.New("model recipe: capability alias lacks active evidence")
	}
	bundleIDs, err := validateCapabilityBundles(activation, bundles)
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	var peerID *artifact.ID
	if peer.ID.Valid() {
		peerID = artifact.CloneID(&peer.ID)
	}
	var evidenceID *artifact.ID
	if activation.Event.ID.Valid() {
		evidenceID = artifact.CloneID(&activation.Event.ID)
	}
	scope := definition.ID
	if activation.Event.ID.Valid() {
		var err error
		scope, err = artifact.JSONID(artifact.KindProfile, capabilityScopeIdentity{
			Alias: alias, Model: definition.Model, Task: definition.Task,
		})
		if err != nil {
			return CapabilityEvidenceSelection{}, err
		}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, capabilitySelectionIdentity{
		Scope: scope, Alias: alias, Session: session,
		Model: definition.Model, Recipe: definition.ID,
		Resources: resources.Identity, Bundles: bundleIDs, Evidence: evidenceID, Peer: peerID,
	})
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	return CapabilityEvidenceSelection{
		Identity: identity, Scope: scope, Alias: alias, Session: session,
		Activation: activation, Program: program, Resources: resources, Bundles: cloneCapabilityBundles(bundles), Peer: peer,
	}, nil
}
