package modelrecipe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/recipe"
	"llamacpp2go/internal/strictjson"
)

const (
	ProfileParityVersion   uint16 = 1
	ProfileParityMediaType        = "application/vnd.llamacpp2go.model-profile-parity+json"
	ProfileParitySchema           = "llamacpp2go/model-profile-parity/v1"
)

var profileParityContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: ProfileParityMediaType, Schema: ProfileParitySchema,
}

type profileParityBody struct {
	Version      uint16                  `json:"version"`
	Recipe       artifact.ID             `json:"recipe"`
	Model        artifact.ID             `json:"model"`
	Profile      artifact.ID             `json:"profile"`
	Architecture string                  `json:"architecture"`
	Layers       uint32                  `json:"layers"`
	CacheLayers  uint32                  `json:"cache_layers"`
	CachedGraph  model.CachedGraphPolicy `json:"cached_graph"`
}

// ProfileParityEvidence: registry/profile equivalence fact.
type ProfileParityEvidence struct {
	ID           artifact.ID
	Version      uint16
	Recipe       artifact.ID
	Model        artifact.ID
	Profile      artifact.ID
	Architecture string
	Layers       uint32
	CacheLayers  uint32
	CachedGraph  model.CachedGraphPolicy
}

func newProfileParityEvidence(
	definition recipe.Definition,
	document ProfileDocument,
	plan Plan,
) (ProfileParityEvidence, error) {
	evidence := ProfileParityEvidence{
		Version: ProfileParityVersion, Recipe: definition.ID, Model: definition.Model,
		Profile: document.ID, Architecture: document.Architecture,
		Layers: uint32(len(plan.Model.Layers)), CacheLayers: plan.Model.CacheLayers,
		CachedGraph: plan.Model.CachedGraph,
	}
	if err := evidence.validateShape(); err != nil {
		return ProfileParityEvidence{}, err
	}
	content, err := profileParityContent(evidence)
	if err != nil {
		return ProfileParityEvidence{}, err
	}
	evidence.ID, err = profileParityContract.Identify(content)
	return evidence, err
}

func ParseProfileParityEvidence(content []byte) (ProfileParityEvidence, error) {
	var body profileParityBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return ProfileParityEvidence{}, fmt.Errorf("model recipe: decode profile parity: %w", err)
	}
	evidence := ProfileParityEvidence{
		Version: body.Version, Recipe: body.Recipe, Model: body.Model,
		Profile: body.Profile, Architecture: body.Architecture,
		Layers: body.Layers, CacheLayers: body.CacheLayers, CachedGraph: body.CachedGraph,
	}
	if err := evidence.validateShape(); err != nil {
		return ProfileParityEvidence{}, err
	}
	canonical, err := profileParityContent(evidence)
	if err != nil {
		return ProfileParityEvidence{}, err
	}
	if !bytes.Equal(canonical, content) {
		return ProfileParityEvidence{}, errors.New("model recipe: non-canonical profile parity content")
	}
	evidence.ID, err = profileParityContract.Identify(content)
	return evidence, err
}

func (e ProfileParityEvidence) Content() (artifact.Content, error) {
	if err := e.ValidateIdentity(); err != nil {
		return artifact.Content{}, err
	}
	content, err := profileParityContent(e)
	if err != nil {
		return artifact.Content{}, err
	}
	return profileParityContract.Content(e.ID, content)
}

func (e ProfileParityEvidence) ValidateIdentity() error {
	if e.ID.Kind() != artifact.KindEvidence {
		return errors.New("model recipe: invalid profile parity identity")
	}
	if err := e.validateShape(); err != nil {
		return err
	}
	content, err := profileParityContent(e)
	if err != nil {
		return err
	}
	if err := profileParityContract.ValidateIdentity(e.ID, content); err != nil {
		return errors.New("model recipe: profile parity identity mismatch")
	}
	return nil
}

func (e ProfileParityEvidence) validateShape() error {
	if e.Version != ProfileParityVersion || e.Recipe.Kind() != artifact.KindRecipe ||
		e.Model.Kind() != artifact.KindModel || e.Profile.Kind() != artifact.KindProfile ||
		e.Architecture == "" {
		return errors.New("model recipe: invalid profile parity envelope")
	}
	return nil
}

func profileParityContent(e ProfileParityEvidence) ([]byte, error) {
	content, err := json.Marshal(profileParityBody{
		Version: e.Version, Recipe: e.Recipe, Model: e.Model, Profile: e.Profile,
		Architecture: e.Architecture, Layers: e.Layers, CacheLayers: e.CacheLayers,
		CachedGraph: e.CachedGraph,
	})
	if err != nil {
		return nil, fmt.Errorf("model recipe: encode profile parity: %w", err)
	}
	return content, nil
}

func ValidateProfileCandidate(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	spec model.Spec,
	weights model.Weights,
) (artifact.CommitID, recipe.LifecycleEvent, ProfileParityEvidence, error) {
	document, err := boundProfile(ctx, store, definition)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, ProfileParityEvidence{}, err
	}
	plan, err := VerifyProfileParity(definition, document, spec, weights)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, ProfileParityEvidence{}, err
	}
	evidence, err := newProfileParityEvidence(definition, document, plan)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, ProfileParityEvidence{}, err
	}
	content, err := evidence.Content()
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, ProfileParityEvidence{}, err
	}
	commit, event, err := transition(
		ctx, store, key, definition, recipe.StatusValidated,
		[]artifact.ID{evidence.ID}, nil, []artifact.Content{content},
	)
	return commit, event, evidence, err
}

func ActivateProfileCandidate(
	ctx context.Context,
	store artifact.Repository,
	key string,
	definition recipe.Definition,
	supersedes *artifact.ID,
) (artifact.CommitID, recipe.LifecycleEvent, error) {
	document, err := boundProfile(ctx, store, definition)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	event, err := currentEvent(ctx, store, definition.ID)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	evidence, err := matchingProfileParity(ctx, store, event.Evidence, definition, document)
	if err != nil {
		return artifact.CommitID{}, recipe.LifecycleEvent{}, err
	}
	return Transition(
		ctx, store, key, definition, recipe.StatusActive, []artifact.ID{evidence.ID}, supersedes,
	)
}

func boundProfile(
	ctx context.Context,
	store artifact.Reader,
	definition recipe.Definition,
) (ProfileDocument, error) {
	id, ok := definition.Dependency(recipe.DependencyProfile, 0)
	if !ok && definition.Version == recipe.LegacyVersion {
		var err error
		id, ok, err = store.ResolveAlias(ctx, legacyRecipeProfileAlias(definition.ID))
		if err != nil {
			return ProfileDocument{}, err
		}
	}
	if !ok {
		return ProfileDocument{}, errors.New("model recipe: profile binding is absent")
	}
	return loadProfile(ctx, store, id)
}

func matchingProfileParity(
	ctx context.Context,
	store artifact.Reader,
	ids []artifact.ID,
	definition recipe.Definition,
	document ProfileDocument,
) (ProfileParityEvidence, error) {
	for _, id := range ids {
		content, ok, err := store.Content(ctx, id)
		if err != nil {
			return ProfileParityEvidence{}, err
		}
		if !ok || profileParityContract.ValidateContent(content, id) != nil {
			continue
		}
		evidence, parseErr := ParseProfileParityEvidence(content.Data)
		if parseErr != nil {
			return ProfileParityEvidence{}, parseErr
		}
		if evidence.Recipe == definition.ID && evidence.Model == definition.Model &&
			evidence.Profile == document.ID && evidence.Architecture == document.Architecture {
			return evidence, nil
		}
	}
	return ProfileParityEvidence{}, errors.New("model recipe: matching profile parity evidence is absent")
}
