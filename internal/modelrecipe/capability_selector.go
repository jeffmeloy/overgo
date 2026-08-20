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

// CapabilityEvidenceSelector: typed model alias plus serving intent.
type CapabilityEvidenceSelector struct {
	Alias         string
	Task          recipe.Task
	Session       SessionSelection
	Compatibility artifact.ID
}

// CapabilityEvidenceSelection: verified executable alias target.
type CapabilityEvidenceSelection struct {
	Identity   artifact.ID
	Alias      string
	Session    SessionSelection
	Activation Activation
	Program    recipe.Program
	Resources  ComponentSessionPlan
	Peer       RemotePeerCompatibility
}

type capabilitySelectionIdentity struct {
	Alias     string           `json:"alias"`
	Session   SessionSelection `json:"session"`
	Model     artifact.ID      `json:"model"`
	Recipe    artifact.ID      `json:"recipe"`
	Resources artifact.ID      `json:"resources"`
	Evidence  artifact.ID      `json:"evidence"`
	Peer      *artifact.ID     `json:"peer,omitempty"`
}

// ResolveCapabilityEvidenceSelector resolves RepoDB alias, recipe, evidence, and resources.
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
	var peerID *artifact.ID
	if peer.ID.Valid() {
		peerID = artifact.CloneID(&peer.ID)
	}
	identity, err := artifact.JSONID(artifact.KindProfile, capabilitySelectionIdentity{
		Alias: selector.Alias, Session: selector.Session,
		Model: modelID, Recipe: activation.Definition.ID,
		Resources: resources.Identity, Evidence: activation.Event.ID, Peer: peerID,
	})
	if err != nil {
		return CapabilityEvidenceSelection{}, err
	}
	return CapabilityEvidenceSelection{
		Identity: identity, Alias: selector.Alias, Session: selector.Session,
		Activation: activation, Program: program, Resources: resources, Peer: peer,
	}, nil
}
