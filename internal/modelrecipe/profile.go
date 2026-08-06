package modelrecipe

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

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
	if d.Version != ProfileVersion || d.Architecture == "" || d.Policy.Name != d.Architecture ||
		strings.TrimSpace(d.Architecture) != d.Architecture {
		return errors.New("model recipe: invalid profile envelope")
	}
	for _, character := range d.Architecture {
		if character != '-' && character != '_' && (character < 'a' || character > 'z') &&
			(character < '0' || character > '9') {
			return errors.New("model recipe: invalid profile architecture")
		}
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

func CompileWithProfile(
	definition recipe.Definition,
	document ProfileDocument,
	spec model.Spec,
	weights model.Weights,
) (Plan, error) {
	if err := definition.Validate(catalog); err != nil {
		return Plan{}, err
	}
	if err := document.ValidateIdentity(); err != nil {
		return Plan{}, err
	}
	if definition.Task != recipe.TaskInference || document.Architecture != spec.Architecture {
		return Plan{}, errors.New("model recipe: profile does not match inference recipe")
	}
	foundCompile, foundForward := false, false
	for _, node := range definition.Nodes {
		foundCompile = foundCompile || node.Module == ModuleCompileModelPlan
		foundForward = foundForward || node.Module == ModuleForwardTokens
	}
	if !foundCompile || !foundForward {
		return Plan{}, errors.New("model recipe: inference path is incomplete")
	}
	modelPlan, err := model.CompileModelPlanWithProfile(spec, weights, document.Policy)
	if err != nil {
		return Plan{}, err
	}
	return Plan{
		Recipe: definition, Model: modelPlan, Nodes: append([]recipe.Node(nil), definition.Nodes...),
	}, nil
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
