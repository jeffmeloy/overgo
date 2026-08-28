package runrecord

import (
	"context"
	"encoding/json"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const (
	// AutomationActivationMediaType identifies automation activation documents.
	AutomationActivationMediaType = "application/vnd.overgo.automation-activation+json"
	// AutomationActivationSchema identifies the activation contract.
	AutomationActivationSchema = "overgo.automation-activation.v1"
	// AutomationActiveAliasRoot scopes the active definition by automation name.
	AutomationActiveAliasRoot = "automation/active/"
)

// AutomationActivation records one exact active-definition transition.
type AutomationActivation struct {
	Version    uint16      `json:"version"`
	Name       string      `json:"name"`
	Definition artifact.ID `json:"definition"`
	Authority  artifact.ID `json:"authority"`
	Prior      artifact.ID `json:"prior,omitzero"`
	ID         artifact.ID `json:"-"`
}

// ActiveAutomation is the validated active definition and activation evidence.
type ActiveAutomation struct {
	Definition recipe.AutomationDefinition
	Activation AutomationActivation
}

// AutomationAuthority projects and advances active automation definitions
// through one repository. It owns no mutable state outside OvergoDB.
type AutomationAuthority struct {
	Repository artifact.Repository
}

type automationActivationBody struct {
	Version    uint16      `json:"version"`
	Name       string      `json:"name"`
	Definition artifact.ID `json:"definition"`
	Authority  artifact.ID `json:"authority"`
	Prior      artifact.ID `json:"prior,omitzero"`
}

var automationActivationCodec = artifact.DocumentCodec[AutomationActivation]{
	Name: "automation activation",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: AutomationActivationMediaType, Schema: AutomationActivationSchema,
	},
	Decode: func(data []byte, value *AutomationActivation) error {
		var body automationActivationBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = AutomationActivation{
			Version: body.Version, Name: body.Name, Definition: body.Definition,
			Authority: body.Authority, Prior: body.Prior,
		}
		return nil
	},
	Encode: func(value AutomationActivation) ([]byte, error) {
		return json.Marshal(automationActivationBody{
			Version: value.Version, Name: value.Name, Definition: value.Definition,
			Authority: value.Authority, Prior: value.Prior,
		})
	},
	Canonicalize: canonicalizeAutomationActivation,
	Identity:     func(value AutomationActivation) artifact.ID { return value.ID },
	SetIdentity:  func(value *AutomationActivation, id artifact.ID) { value.ID = id },
}

func newAutomationActivation(
	definition recipe.AutomationDefinition,
	authority, prior artifact.ID,
) (AutomationActivation, error) {
	if err := definition.ValidateIdentity(); err != nil {
		return AutomationActivation{}, err
	}
	return automationActivationCodec.New(AutomationActivation{
		Version: artifact.InitialDocumentVersion, Name: definition.Name,
		Definition: definition.ID, Authority: authority, Prior: prior,
	})
}

func resolveAutomationActivation(
	ctx context.Context,
	reader artifact.Reader,
	name string,
) (ActiveAutomation, bool, error) {
	if !textcheck.LowerIdentifier(name, len(name)) {
		return ActiveAutomation{}, false, errors.New("run record: invalid automation name")
	}
	activation, found, err := automationActivationCodec.Resolve(ctx, reader, AutomationActiveAliasRoot+name)
	if err != nil || !found {
		return ActiveAutomation{}, found, err
	}
	definition, err := recipe.RequireAutomationDefinition(ctx, reader, activation.Definition)
	if err != nil {
		return ActiveAutomation{}, false, err
	}
	if activation.Name != name || activation.Name != definition.Name {
		return ActiveAutomation{}, false, errors.New("run record: automation alias subject differs")
	}
	return ActiveAutomation{Definition: definition, Activation: activation}, true, nil
}

func publishAutomationActivation(
	ctx context.Context,
	repository artifact.Repository,
	key string,
	definition recipe.AutomationDefinition,
	activation AutomationActivation,
) (artifact.CommitID, error) {
	identified, identifyErr := recipe.NewAutomationDefinition(definition)
	if ctx == nil || repository == nil || key == "" || activation.Definition != definition.ID ||
		activation.Name != definition.Name || automationActivationCodec.ValidateIdentity(activation) != nil ||
		definition.ValidateIdentity() != nil || identifyErr != nil || identified.ID != definition.ID {
		return artifact.CommitID{}, errors.New("run record: invalid automation activation publication")
	}
	current, found, err := resolveAutomationActivation(ctx, repository, definition.Name)
	if err != nil {
		return artifact.CommitID{}, err
	}
	if found != activation.Prior.Valid() || found && current.Activation.ID != activation.Prior {
		return artifact.CommitID{}, errors.New("run record: automation activation chain differs")
	}
	definitionContent, err := definition.ArtifactContent()
	if err != nil {
		return artifact.CommitID{}, err
	}
	activationContent, err := automationActivationCodec.Content(activation)
	if err != nil {
		return artifact.CommitID{}, err
	}
	lineage := definition.Lineage()
	parents := []artifact.ID{definition.ID, activation.Authority}
	if activation.Prior.Valid() {
		parents = append(parents, activation.Prior)
	}
	lineage = append(lineage, artifact.DependencyLineage(activation.ID, parents...)...)
	alias := artifact.AliasBinding{Name: AutomationActiveAliasRoot + definition.Name, Target: activation.ID}
	if found {
		alias.Previous = artifact.IDPointer(current.Activation.ID)
	}
	batch, err := artifact.NewDocumentBatch(
		key, []artifact.Content{definitionContent, activationContent}, lineage, []artifact.AliasBinding{alias},
	)
	if err != nil {
		return artifact.CommitID{}, err
	}
	return artifact.CommitBatch(ctx, repository, batch)
}

// Resolve returns the exact active automation by name.
func (authority AutomationAuthority) Resolve(ctx context.Context, name string) (ActiveAutomation, bool, error) {
	return resolveAutomationActivation(ctx, authority.Repository, name)
}

// Activate advances one named automation from its current active activation,
// or creates the first activation when the name is unbound.
func (authority AutomationAuthority) Activate(
	ctx context.Context,
	key string,
	definition recipe.AutomationDefinition,
	decision artifact.ID,
) (ActiveAutomation, error) {
	current, found, err := resolveAutomationActivation(ctx, authority.Repository, definition.Name)
	if err != nil {
		return ActiveAutomation{}, err
	}
	prior := artifact.ID{}
	if found {
		prior = current.Activation.ID
	}
	activation, err := newAutomationActivation(definition, decision, prior)
	if err != nil {
		return ActiveAutomation{}, err
	}
	if _, err := publishAutomationActivation(ctx, authority.Repository, key, definition, activation); err != nil {
		return ActiveAutomation{}, err
	}
	return ActiveAutomation{Definition: definition, Activation: activation}, nil
}

func canonicalizeAutomationActivation(value *AutomationActivation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		!textcheck.LowerIdentifier(value.Name, len(value.Name)) ||
		value.Definition.Kind() != artifact.KindRecipe || value.Authority.Kind() != artifact.KindEvidence ||
		value.Authority == value.Prior || value.Prior.Valid() && value.Prior.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid automation activation")
	}
	return nil
}
