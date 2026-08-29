package modelrecipe

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const remotePeerCompatibilityVersion uint16 = 1

var remotePeerCompatibilityCodec = artifact.JSONDocumentCodec(
	"remote peer compatibility", artifact.KindEvidence,
	"application/vnd.overgo.remote-peer-compatibility+json", "overgo/remote-peer-compatibility/v1",
	canonicalizeRemotePeerCompatibility,
	func(value RemotePeerCompatibility) artifact.ID { return value.ID },
	func(value *RemotePeerCompatibility, id artifact.ID) { value.ID = id }, nil,
)

// RemotePeerCompatibility defines observed parity for one recipe resource plan.
type RemotePeerCompatibility struct {
	Version          uint16               `json:"version"`
	Model            artifact.ID          `json:"model"`
	Recipe           artifact.ID          `json:"recipe"`
	Resources        artifact.ID          `json:"resources"`
	Task             recipe.Task          `json:"task"`
	LocalEnvironment artifact.ID          `json:"local_environment"`
	PeerEnvironment  artifact.ID          `json:"peer_environment"`
	PeerCapability   artifact.ID          `json:"peer_capability"`
	LocalObservation artifact.ID          `json:"local_observation"`
	PeerObservation  artifact.ID          `json:"peer_observation"`
	Capability       RemotePeerCapability `json:"-"`
	ID               artifact.ID          `json:"-"`
}

// PublishRemotePeerAuthority commits one capability and exact compatibility record.
func PublishRemotePeerAuthority(
	ctx context.Context,
	repository artifact.Repository,
	key string,
	capability RemotePeerCapability,
	compatibility RemotePeerCompatibility,
) (RemotePeerCompatibility, error) {
	if ctx == nil || repository == nil {
		return RemotePeerCompatibility{}, errors.New("model recipe: remote peer repository is absent")
	}
	capability.Version, capability.ID = remotePeerCapabilityVersion, artifact.ID{}
	identifiedCapability, err := remotePeerCapabilityCodec.New(capability)
	if err != nil {
		return RemotePeerCompatibility{}, err
	}
	if compatibility.PeerCapability.Valid() && compatibility.PeerCapability != identifiedCapability.ID {
		return RemotePeerCompatibility{}, errors.New("model recipe: remote peer capability identity differs")
	}
	compatibility.Version, compatibility.ID = remotePeerCompatibilityVersion, artifact.ID{}
	compatibility.PeerCapability = identifiedCapability.ID
	identifiedCompatibility, err := remotePeerCompatibilityCodec.New(compatibility)
	if err != nil || identifiedCapability.Environment != identifiedCompatibility.PeerEnvironment ||
		!slices.Contains(identifiedCapability.Tasks, identifiedCompatibility.Task) {
		return RemotePeerCompatibility{}, errors.Join(errors.New("model recipe: remote peer authority differs"), err)
	}
	if err := validatePeerCompatibilityReferences(ctx, repository, identifiedCompatibility); err != nil {
		return RemotePeerCompatibility{}, err
	}
	capabilityContent, err := remotePeerCapabilityCodec.Content(identifiedCapability)
	if err != nil {
		return RemotePeerCompatibility{}, err
	}
	compatibilityContent, err := remotePeerCompatibilityCodec.Content(identifiedCompatibility)
	if err != nil {
		return RemotePeerCompatibility{}, err
	}
	lineage := artifact.DependencyLineage(identifiedCapability.ID, identifiedCapability.Environment)
	lineage = append(lineage, artifact.DependencyLineage(identifiedCompatibility.ID,
		identifiedCompatibility.Model, identifiedCompatibility.Recipe, identifiedCompatibility.Resources,
		identifiedCompatibility.LocalEnvironment, identifiedCompatibility.PeerEnvironment,
		identifiedCompatibility.PeerCapability, identifiedCompatibility.LocalObservation,
		identifiedCompatibility.PeerObservation)...)
	batch, err := artifact.NewDocumentBatch(
		key+"/"+identifiedCompatibility.ID.String(),
		[]artifact.Content{capabilityContent, compatibilityContent}, lineage, nil,
	)
	if err != nil {
		return RemotePeerCompatibility{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return RemotePeerCompatibility{}, err
	}
	identifiedCompatibility.Capability = identifiedCapability
	return identifiedCompatibility, nil
}

func (value RemotePeerCompatibility) batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{
		value.Model, value.Recipe, value.Resources, value.LocalEnvironment, value.PeerEnvironment,
		value.PeerCapability, value.LocalObservation, value.PeerObservation,
	}
	return remotePeerCompatibilityCodec.Batch(key, value, artifact.DependencyLineage(value.ID, parents...), nil)
}

func canonicalizeRemotePeerCompatibility(value *RemotePeerCompatibility) error {
	if value == nil || value.Version != remotePeerCompatibilityVersion ||
		value.Model.Kind() != artifact.KindModel || value.Recipe.Kind() != artifact.KindRecipe ||
		value.Resources.Kind() != artifact.KindProfile || !value.Task.Valid() ||
		value.LocalEnvironment.Kind() != artifact.KindEvidence || value.PeerEnvironment.Kind() != artifact.KindEvidence ||
		value.LocalEnvironment == value.PeerEnvironment ||
		value.PeerCapability.Kind() != artifact.KindEvidence ||
		value.LocalObservation.Kind() != artifact.KindEvidence || value.PeerObservation.Kind() != artifact.KindEvidence ||
		value.LocalObservation == value.PeerObservation {
		return errors.New("model recipe: invalid remote peer compatibility authority")
	}
	return nil
}

func resolveRemotePeerCompatibility(
	ctx context.Context,
	store artifact.Reader,
	id artifact.ID,
	model, recipeID, resources artifact.ID,
	task recipe.Task,
) (RemotePeerCompatibility, error) {
	compatibility, err := remotePeerCompatibilityCodec.Require(ctx, store, id)
	if err != nil || compatibility.ID != id || compatibility.Model != model ||
		compatibility.Recipe != recipeID || compatibility.Resources != resources || compatibility.Task != task {
		return RemotePeerCompatibility{}, errors.Join(errors.New("model recipe: remote peer compatibility differs"), err)
	}
	capability, err := remotePeerCapabilityCodec.Require(ctx, store, compatibility.PeerCapability)
	if err != nil || capability.Environment != compatibility.PeerEnvironment || !slices.Contains(capability.Tasks, task) {
		return RemotePeerCompatibility{}, errors.Join(errors.New("model recipe: remote peer capability differs"), err)
	}
	compatibility.Capability = capability
	if err := validatePeerCompatibilityReferences(ctx, store, compatibility); err != nil {
		return RemotePeerCompatibility{}, err
	}
	return compatibility, nil
}

func validatePeerCompatibilityReferences(
	ctx context.Context,
	store artifact.Reader,
	compatibility RemotePeerCompatibility,
) error {
	definition, err := recipe.RequireDefinition(ctx, store, compatibility.Recipe)
	if err != nil || definition.ID != compatibility.Recipe || definition.Model != compatibility.Model ||
		definition.Task != compatibility.Task {
		return errors.Join(errors.New("model recipe: peer recipe definition differs"), err)
	}
	if err := requirePeerObservation(
		ctx, store, compatibility.LocalObservation, compatibility.LocalEnvironment, compatibility,
	); err != nil {
		return err
	}
	if err := requirePeerObservation(
		ctx, store, compatibility.PeerObservation, compatibility.PeerEnvironment, compatibility,
	); err != nil {
		return err
	}
	return nil
}

func requirePeerObservation(
	ctx context.Context,
	store artifact.Reader,
	observationID, environment artifact.ID,
	compatibility RemotePeerCompatibility,
) error {
	observation, err := runrecord.RequireServingAttemptObservation(ctx, store, observationID)
	if err != nil || observation.Outcome != runrecord.OutcomeSucceeded || observation.Model != compatibility.Model ||
		observation.Recipe != compatibility.Recipe || observation.Task != compatibility.Task ||
		observation.Environment != environment {
		return errors.Join(errors.New("model recipe: peer observation is incompatible"), err)
	}
	return nil
}
