package modelrecipe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/recipe"
	"llamacpp2go/internal/strictjson"
)

const (
	ProfileVersion   uint16 = 1
	ProfileMediaType        = "application/vnd.llamacpp2go.model-profile+json"
	ProfileSchema           = "llamacpp2go/model-profile/v1"
)

type profileBody struct {
	Version      uint16                    `json:"version"`
	Architecture string                    `json:"architecture"`
	Policy       model.ArchitectureProfile `json:"policy"`
}

// ProfileDocument: immutable architecture policy.
type ProfileDocument struct {
	ID           artifact.ID
	Version      uint16
	Architecture string
	Policy       model.ArchitectureProfile
}

func NewProfileDocument(profile model.ArchitectureProfile) (ProfileDocument, error) {
	document := ProfileDocument{
		Version: ProfileVersion, Architecture: profile.Name, Policy: profile,
	}
	if err := document.validateShape(); err != nil {
		return ProfileDocument{}, err
	}
	content, err := profileDocumentContent(document)
	if err != nil {
		return ProfileDocument{}, err
	}
	id, err := artifact.IdentifyBytes(artifact.KindProfile, content)
	if err != nil {
		return ProfileDocument{}, err
	}
	document.ID = id
	return document, nil
}

func ParseProfileDocument(content []byte) (ProfileDocument, error) {
	var body profileBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return ProfileDocument{}, fmt.Errorf("model recipe: decode profile: %w", err)
	}
	if body.Version != ProfileVersion {
		return ProfileDocument{}, errors.New("model recipe: unsupported profile version")
	}
	document, err := NewProfileDocument(body.Policy)
	if err != nil {
		return ProfileDocument{}, err
	}
	if body.Architecture != document.Architecture {
		return ProfileDocument{}, errors.New("model recipe: profile architecture mismatch")
	}
	canonical, err := document.Content()
	if err != nil {
		return ProfileDocument{}, err
	}
	if !bytes.Equal(canonical, content) {
		return ProfileDocument{}, errors.New("model recipe: non-canonical profile content")
	}
	return document, nil
}

func (d ProfileDocument) ValidateIdentity() error {
	if d.ID.Kind() != artifact.KindProfile {
		return errors.New("model recipe: invalid profile identity")
	}
	if err := d.validateShape(); err != nil {
		return err
	}
	content, err := profileDocumentContent(d)
	if err != nil {
		return err
	}
	want, err := artifact.IdentifyBytes(artifact.KindProfile, content)
	if err != nil {
		return err
	}
	if d.ID != want {
		return errors.New("model recipe: profile identity mismatch")
	}
	return nil
}

func (d ProfileDocument) Content() ([]byte, error) {
	if err := d.ValidateIdentity(); err != nil {
		return nil, err
	}
	return profileDocumentContent(d)
}

func (d ProfileDocument) Descriptor() (artifact.Descriptor, error) {
	content, err := d.Content()
	if err != nil {
		return artifact.Descriptor{}, err
	}
	return artifact.Descriptor{
		ID: d.ID, Size: uint64(len(content)), MediaType: ProfileMediaType, Schema: ProfileSchema,
	}, nil
}

func (d ProfileDocument) validateShape() error {
	if d.Version != ProfileVersion || d.Architecture == "" || d.Policy.Name != d.Architecture {
		return errors.New("model recipe: invalid profile envelope")
	}
	if err := model.ValidateArchitectureProfile(d.Policy); err != nil {
		return fmt.Errorf("model recipe: invalid profile policy: %w", err)
	}
	return nil
}

func profileDocumentContent(d ProfileDocument) ([]byte, error) {
	content, err := json.Marshal(profileBody{
		Version: d.Version, Architecture: d.Architecture, Policy: d.Policy,
	})
	if err != nil {
		return nil, fmt.Errorf("model recipe: encode profile: %w", err)
	}
	return content, nil
}

func ProfileContent(document ProfileDocument) (artifact.Content, error) {
	descriptor, err := document.Descriptor()
	if err != nil {
		return artifact.Content{}, err
	}
	content, err := document.Content()
	if err != nil {
		return artifact.Content{}, err
	}
	return artifact.Content{Descriptor: descriptor, Data: content}, nil
}

func PublishProfiles(
	ctx context.Context,
	store artifact.Repository,
	key string,
) (artifact.CommitID, []ProfileDocument, error) {
	documents, err := SeedProfileDocuments()
	if err != nil {
		return artifact.CommitID{}, nil, err
	}
	commit, err := PublishProfileCatalog(ctx, store, key, documents)
	return commit, documents, err
}

// PublishProfileCatalog stores externally supplied profile authority.
func PublishProfileCatalog(
	ctx context.Context,
	store artifact.Repository,
	key string,
	documents []ProfileDocument,
) (artifact.CommitID, error) {
	if len(documents) == 0 {
		return artifact.CommitID{}, errors.New("model recipe: profile catalog is empty")
	}
	batch := artifact.Batch{Key: key, Contents: make([]artifact.Content, 0, len(documents))}
	architectures := make(map[string]struct{}, len(documents))
	for _, document := range documents {
		if err := document.ValidateIdentity(); err != nil {
			return artifact.CommitID{}, err
		}
		if _, exists := architectures[document.Architecture]; exists {
			return artifact.CommitID{}, fmt.Errorf("model recipe: duplicate profile architecture %q", document.Architecture)
		}
		architectures[document.Architecture] = struct{}{}
		content, contentErr := ProfileContent(document)
		if contentErr != nil {
			return artifact.CommitID{}, contentErr
		}
		batch.Contents = append(batch.Contents, content)
		alias := registeredProfileAlias(document.Architecture)
		current, ok, lookupErr := store.ResolveAlias(ctx, alias)
		if lookupErr != nil {
			return artifact.CommitID{}, lookupErr
		}
		if ok && current == document.ID {
			continue
		}
		binding := artifact.AliasBinding{Name: alias, Target: document.ID}
		if ok {
			binding.Previous = &current
		}
		batch.Aliases = append(batch.Aliases, binding)
	}
	return store.Commit(ctx, batch)
}

func RegisteredProfile(
	ctx context.Context,
	store artifact.Reader,
	architecture string,
) (ProfileDocument, bool, error) {
	id, ok, err := store.ResolveAlias(ctx, registeredProfileAlias(architecture))
	if err != nil || !ok {
		return ProfileDocument{}, ok, err
	}
	document, err := loadProfile(ctx, store, id)
	return document, err == nil, err
}

// SeedProfileDocuments: canonical snapshot of registered policies.
func SeedProfileDocuments() ([]ProfileDocument, error) {
	names := model.SupportedArchitectures()
	documents := make([]ProfileDocument, 0, len(names))
	for _, name := range names {
		profile, ok := model.LookupArchitecture(name)
		if !ok {
			return nil, fmt.Errorf("model recipe: registered profile %q is absent", name)
		}
		document, err := NewProfileDocument(profile)
		if err != nil {
			return nil, err
		}
		documents = append(documents, document)
	}
	return documents, nil
}

func loadProfile(ctx context.Context, store artifact.Reader, id artifact.ID) (ProfileDocument, error) {
	content, ok, err := store.Content(ctx, id)
	if err != nil {
		return ProfileDocument{}, err
	}
	if !ok || content.Descriptor.Schema != ProfileSchema {
		return ProfileDocument{}, errors.New("model recipe: profile content is absent or incompatible")
	}
	return ParseProfileDocument(content.Data)
}

func registeredProfileAlias(architecture string) string {
	return "profile.registered." + architecture
}

func CompileWithProfile(
	definition recipe.Definition,
	document ProfileDocument,
	spec model.Spec,
	weights model.Weights,
) (Plan, error) {
	if err := document.ValidateIdentity(); err != nil {
		return Plan{}, err
	}
	if document.Architecture != spec.Architecture {
		return Plan{}, errors.New("model recipe: profile does not match inference recipe")
	}
	if definition.Version != recipe.LegacyVersion {
		profileID, ok := definition.Dependency(recipe.DependencyProfile, 0)
		if !ok || profileID != document.ID {
			return Plan{}, errors.New("model recipe: definition profile dependency mismatch")
		}
	}
	return compileDefinition(definition, func() (model.ModelPlan, error) {
		return model.CompileModelPlanWithProfile(spec, weights, document.Policy)
	})
}

// VerifyProfileParity: registry/profile plan equivalence gate.
func VerifyProfileParity(
	definition recipe.Definition,
	document ProfileDocument,
	spec model.Spec,
	weights model.Weights,
) (Plan, error) {
	legacy, err := Compile(definition, spec, weights)
	if err != nil {
		return Plan{}, err
	}
	candidate, err := CompileWithProfile(definition, document, spec, weights)
	if err != nil {
		return Plan{}, err
	}
	if !reflect.DeepEqual(legacy.Model, candidate.Model) {
		return Plan{}, errors.New("model recipe: profile parity mismatch")
	}
	return candidate, nil
}
