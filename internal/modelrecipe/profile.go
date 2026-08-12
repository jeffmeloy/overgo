package modelrecipe

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

const (
	LegacyProfileVersion uint16 = 1
	ProfileVersion       uint16 = 2
	ProfileMediaType            = "application/vnd.overgo.model-profile+json"
	LegacyProfileSchema         = "overgo/model-profile/v1"
	ProfileSchema               = "overgo/model-profile/v2"
)

var profileContract = artifact.DocumentContract{
	Kind: artifact.KindProfile, MediaType: ProfileMediaType, Schema: ProfileSchema,
}

var legacyProfileContract = artifact.DocumentContract{
	Kind: artifact.KindProfile, MediaType: ProfileMediaType, Schema: LegacyProfileSchema,
}

var profileCodec = artifact.DocumentCodec[ProfileDocument]{
	Name: "model profile", ContractFor: func(value ProfileDocument) artifact.DocumentContract {
		return profileContractForVersion(value.Version)
	},
	Decode: func(data []byte, value *ProfileDocument) error {
		var body profileBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = ProfileDocument{
			Version: body.Version, Architecture: body.Architecture,
			Policy: body.Policy, Provenance: slices.Clone(body.Provenance),
		}
		return nil
	},
	Encode:       profileDocumentContent,
	Canonicalize: func(value *ProfileDocument) error { return value.validateShape() },
	Clone: func(value ProfileDocument) ProfileDocument {
		value.Provenance = slices.Clone(value.Provenance)
		return value
	},
	Identity:    func(value ProfileDocument) artifact.ID { return value.ID },
	SetIdentity: func(value *ProfileDocument, id artifact.ID) { value.ID = id },
}

type profileBody struct {
	Version      uint16                    `json:"version"`
	Architecture string                    `json:"architecture"`
	Policy       model.ArchitectureProfile `json:"policy"`
	Provenance   []ProfileFactProvenance   `json:"provenance,omitempty"`
}

// ProfileDocument: immutable architecture policy.
type ProfileDocument struct {
	ID           artifact.ID
	Version      uint16
	Architecture string
	Policy       model.ArchitectureProfile
	Provenance   []ProfileFactProvenance
}

func NewProfileDocument(profile model.ArchitectureProfile) (ProfileDocument, error) {
	provenance, err := CatalogProfileProvenance(profile)
	if err != nil {
		return ProfileDocument{}, err
	}
	return NewProfileDocumentWithProvenance(profile, provenance)
}

func NewProfileDocumentWithProvenance(
	profile model.ArchitectureProfile,
	provenance []ProfileFactProvenance,
) (ProfileDocument, error) {
	document := ProfileDocument{
		Version: ProfileVersion, Architecture: profile.Name, Policy: profile,
		Provenance: slices.Clone(provenance),
	}
	return profileCodec.New(document)
}

func ParseProfileDocument(content []byte) (ProfileDocument, error) {
	return profileCodec.Parse(content)
}

func (d ProfileDocument) ValidateIdentity() error {
	return profileCodec.ValidateIdentity(d)
}

func (d ProfileDocument) Content() ([]byte, error) {
	return profileCodec.ContentBytes(d)
}

func (d ProfileDocument) Descriptor() (artifact.Descriptor, error) {
	content, err := d.Content()
	if err != nil {
		return artifact.Descriptor{}, err
	}
	return profileContractForVersion(d.Version).Descriptor(d.ID, uint64(len(content)))
}

func (d ProfileDocument) validateShape() error {
	if d.Version != LegacyProfileVersion && d.Version != ProfileVersion ||
		d.Architecture == "" || d.Policy.Name != d.Architecture {
		return errors.New("model recipe: invalid profile envelope")
	}
	if d.Version == LegacyProfileVersion {
		if len(d.Provenance) != 0 {
			return errors.New("model recipe: legacy profile carries provenance")
		}
	} else {
		provenance := slices.Clone(d.Provenance)
		if err := canonicalizeProfileProvenance(&provenance); err != nil || !slices.Equal(provenance, d.Provenance) {
			if err != nil {
				return err
			}
			return errors.New("model recipe: profile provenance is not canonical")
		}
	}
	if err := model.ValidateArchitectureProfile(d.Policy); err != nil {
		return fmt.Errorf("model recipe: invalid profile policy: %w", err)
	}
	return nil
}

func profileDocumentContent(d ProfileDocument) ([]byte, error) {
	content, err := json.Marshal(profileBody{
		Version: d.Version, Architecture: d.Architecture, Policy: d.Policy,
		Provenance: slices.Clone(d.Provenance),
	})
	if err != nil {
		return nil, fmt.Errorf("model recipe: encode profile: %w", err)
	}
	return content, nil
}

func ProfileContent(document ProfileDocument) (artifact.Content, error) {
	return profileCodec.Content(document)
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
	contents := make([]artifact.Content, 0, len(documents)*2)
	lineage := make([]artifact.Lineage, 0, len(documents))
	aliases := make([]artifact.AliasBinding, 0, len(documents))
	architectures := make(map[string]struct{}, len(documents))
	for _, document := range documents {
		if err := document.ValidateIdentity(); err != nil {
			return artifact.CommitID{}, err
		}
		if _, exists := architectures[document.Architecture]; exists {
			return artifact.CommitID{}, fmt.Errorf("model recipe: duplicate profile architecture %q", document.Architecture)
		}
		architectures[document.Architecture] = struct{}{}
		profileContents, profileLineage, contentErr := profilePublicationFacts(document)
		if contentErr != nil {
			return artifact.CommitID{}, contentErr
		}
		contents = append(contents, profileContents...)
		lineage = append(lineage, profileLineage...)
		alias := registeredProfileAlias(document.Architecture)
		current, ok, lookupErr := artifact.ResolveAlias(ctx, store, alias)
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
		aliases = append(aliases, binding)
	}
	batch, err := artifact.NewDocumentBatch(key, contents, lineage, aliases)
	if err != nil {
		return artifact.CommitID{}, err
	}
	return artifact.CommitBatch(ctx, store, batch)
}

func RegisteredProfile(
	ctx context.Context,
	store artifact.Reader,
	architecture string,
) (ProfileDocument, bool, error) {
	id, ok, err := artifact.ResolveAlias(ctx, store, registeredProfileAlias(architecture))
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
	if !ok || content.Descriptor.MediaType != ProfileMediaType ||
		content.Descriptor.Schema != ProfileSchema && content.Descriptor.Schema != LegacyProfileSchema {
		return ProfileDocument{}, errors.New("model recipe: profile content is absent or incompatible")
	}
	document, err := ParseProfileDocument(content.Data)
	if err != nil {
		return ProfileDocument{}, err
	}
	if document.ID != id {
		return ProfileDocument{}, errors.New("model recipe: profile content identity differs")
	}
	if err := validateStoredProfileProvenance(ctx, store, document); err != nil {
		return ProfileDocument{}, err
	}
	return document, nil
}

func profileContractForVersion(version uint16) artifact.DocumentContract {
	if version == LegacyProfileVersion {
		return legacyProfileContract
	}
	return profileContract
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
	legacy, err := CompileInference(definition, spec, weights)
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
