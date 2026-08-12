package modelrecipe

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/strictjson"
)

const (
	ProfileVersion   uint16 = 2
	ProfileMediaType        = "application/vnd.overgo.model-profile+json"
	ProfileSchema           = "overgo/model-profile/v2"
)

var profileContract = artifact.DocumentContract{
	Kind: artifact.KindProfile, MediaType: ProfileMediaType, Schema: ProfileSchema,
}

var profileCodec = artifact.DocumentCodec[ProfileDocument]{
	Name: "model profile", ContractFor: func(ProfileDocument) artifact.DocumentContract { return profileContract },
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
	return profileContract.Descriptor(d.ID, uint64(len(content)))
}

func (d ProfileDocument) validateShape() error {
	if d.Version != ProfileVersion ||
		d.Architecture == "" || d.Policy.Name != d.Architecture {
		return errors.New("model recipe: invalid profile envelope")
	}
	provenance := slices.Clone(d.Provenance)
	if err := canonicalizeProfileProvenance(&provenance); err != nil || !slices.Equal(provenance, d.Provenance) {
		if err != nil {
			return err
		}
		return errors.New("model recipe: profile provenance is not canonical")
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

func profileContent(document ProfileDocument) (artifact.Content, error) {
	return profileCodec.Content(document)
}

func loadProfile(ctx context.Context, store artifact.Reader, id artifact.ID) (ProfileDocument, error) {
	content, ok, err := artifact.ReadDocument(ctx, store, id, profileContract)
	if err != nil {
		return ProfileDocument{}, err
	}
	if !ok {
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
