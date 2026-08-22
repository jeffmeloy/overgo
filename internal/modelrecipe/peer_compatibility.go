package modelrecipe

import (
	"context"
	"errors"
	"net/url"

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
	Version          uint16      `json:"version"`
	Model            artifact.ID `json:"model"`
	Recipe           artifact.ID `json:"recipe"`
	Resources        artifact.ID `json:"resources"`
	Task             recipe.Task `json:"task"`
	LocalEnvironment artifact.ID `json:"local_environment"`
	PeerEnvironment  artifact.ID `json:"peer_environment"`
	LocalObservation artifact.ID `json:"local_observation"`
	PeerObservation  artifact.ID `json:"peer_observation"`
	Endpoint         string      `json:"endpoint"`
	ID               artifact.ID `json:"-"`
}

func (value RemotePeerCompatibility) batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{
		value.Model, value.Recipe, value.Resources, value.LocalEnvironment, value.PeerEnvironment,
		value.LocalObservation, value.PeerObservation,
	}
	return remotePeerCompatibilityCodec.Batch(key, value, artifact.DependencyLineage(value.ID, parents...), nil)
}

func canonicalizeRemotePeerCompatibility(value *RemotePeerCompatibility) error {
	if value == nil || value.Version != remotePeerCompatibilityVersion ||
		value.Model.Kind() != artifact.KindModel || value.Recipe.Kind() != artifact.KindRecipe ||
		value.Resources.Kind() != artifact.KindProfile || !value.Task.Valid() ||
		value.LocalEnvironment.Kind() != artifact.KindEvidence || value.PeerEnvironment.Kind() != artifact.KindEvidence ||
		value.LocalEnvironment == value.PeerEnvironment ||
		value.LocalObservation.Kind() != artifact.KindEvidence || value.PeerObservation.Kind() != artifact.KindEvidence ||
		value.LocalObservation == value.PeerObservation {
		return errors.New("model recipe: invalid remote peer compatibility authority")
	}
	endpoint, err := url.ParseRequestURI(value.Endpoint)
	if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") || endpoint.Host == "" ||
		endpoint.User != nil || endpoint.Fragment != "" {
		return errors.New("model recipe: invalid remote peer endpoint")
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
	validateObservation := func(observationID, environment artifact.ID) error {
		observation, readErr := runrecord.RequireServingObservation(ctx, store, observationID)
		if readErr != nil || observation.Outcome != runrecord.OutcomeSucceeded || observation.Model != model ||
			observation.Recipe != recipeID || observation.Task != task || observation.Environment != environment {
			return errors.Join(errors.New("model recipe: peer observation is incompatible"), readErr)
		}
		return nil
	}
	if err := validateObservation(compatibility.LocalObservation, compatibility.LocalEnvironment); err != nil {
		return RemotePeerCompatibility{}, err
	}
	if err := validateObservation(compatibility.PeerObservation, compatibility.PeerEnvironment); err != nil {
		return RemotePeerCompatibility{}, err
	}
	return compatibility, nil
}
