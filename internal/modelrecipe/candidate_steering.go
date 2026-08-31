package modelrecipe

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/runrecord"
)

const (
	// SteeringRealizationMediaType identifies one compiled steering arm.
	SteeringRealizationMediaType = "application/vnd.overgo.steering-realization+json"
	// SteeringRealizationSchema identifies the steering realization contract.
	SteeringRealizationSchema = "overgo/steering-realization/v1"
	// SteeringResourcePolicyMediaType identifies the steering resource bound.
	SteeringResourcePolicyMediaType = "application/vnd.overgo.steering-resource-policy+json"
	// SteeringResourcePolicySchema identifies the steering resource contract.
	SteeringResourcePolicySchema = "overgo/steering-resource-policy/v1"
)

// SteeringRealization is one compiled residual-intervention arm. It states
// the direction and its predeclared alpha selector; it never carries a
// concrete alpha, because alpha is derived at activation by the selector and
// is never a request knob. InspectionOnly binds the sole phase a positive
// alpha may run in; the AlphaZero arm is the identity comparison the shared
// evaluator requires.
type SteeringRealization struct {
	Version        uint16                       `json:"version"`
	Direction      artifact.ID                  `json:"direction"`
	Selector       representation.AlphaSelector `json:"selector"`
	InspectionOnly bool                         `json:"inspection_only"`
	AlphaZero      bool                         `json:"alpha_zero"`
	ID             artifact.ID                  `json:"-"`
}

var steeringRealizationCodec = artifact.JSONDocumentCodec(
	"steering realization", artifact.KindProfile, SteeringRealizationMediaType, SteeringRealizationSchema,
	func(value *SteeringRealization) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || !value.Direction.Valid() {
			return errors.New("modelrecipe: invalid steering realization")
		}
		if err := value.Selector.Validate(); err != nil {
			return err
		}
		// A positive-alpha arm exists only inside the inspection-only
		// phase; the identity arm needs no phase because it changes nothing.
		if !value.AlphaZero && !value.InspectionOnly {
			return errors.New("modelrecipe: a steered arm must declare the inspection-only phase")
		}
		return nil
	},
	func(value SteeringRealization) artifact.ID { return value.ID },
	func(value *SteeringRealization, id artifact.ID) { value.ID = id },
	func(value SteeringRealization) SteeringRealization { return value },
)

// Content returns the immutable steering realization.
func (value SteeringRealization) Content() (artifact.Content, error) {
	return steeringRealizationCodec.Content(value)
}

// SteeringResourcePolicy bounds the residual intervention's resident cost:
// exactly the direction vector's bytes, which the runtime holds alongside
// the model.
type SteeringResourcePolicy struct {
	Version          uint16      `json:"version"`
	MaxResidentBytes uint64      `json:"max_resident_bytes"`
	Rationale        string      `json:"rationale"`
	ReopenTrigger    string      `json:"reopen_trigger"`
	ID               artifact.ID `json:"-"`
}

var steeringResourcePolicyCodec = artifact.JSONDocumentCodec(
	"steering resource policy", artifact.KindProfile, SteeringResourcePolicyMediaType, SteeringResourcePolicySchema,
	func(value *SteeringResourcePolicy) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion ||
			!checked.Nonzero(value.MaxResidentBytes) || value.Rationale == "" || value.ReopenTrigger == "" {
			return errors.New("modelrecipe: invalid steering resource policy")
		}
		return nil
	},
	func(value SteeringResourcePolicy) artifact.ID { return value.ID },
	func(value *SteeringResourcePolicy, id artifact.ID) { value.ID = id },
	func(value SteeringResourcePolicy) SteeringResourcePolicy { return value },
)

// Content returns the immutable steering resource policy.
func (value SteeringResourcePolicy) Content() (artifact.Content, error) {
	return steeringResourcePolicyCodec.Content(value)
}

// SteeringCandidatePlugin is the thin residual tool-call propensity plugin:
// extraction and Pareto evaluation ride the common Candidate spine, and the
// plugin owns no recipe, activation, approval, dispatch, or receipt path.
type SteeringCandidatePlugin struct{}

// CandidateDomain selects the closed steering domain.
func (SteeringCandidatePlugin) CandidateDomain() string {
	return string(CandidateSteering)
}

// AdmissionAdapter returns the same stateless plugin for pre-compilation checks.
func (plugin SteeringCandidatePlugin) AdmissionAdapter() runrecord.CandidateComponentAdmissionAdapter {
	return plugin
}

// EvaluatorPlugin returns the typed common evaluation-intent validator.
func (SteeringCandidatePlugin) EvaluatorPlugin() CandidateEvaluatorPlugin {
	return CandidateEvaluationIntentValidator{}
}

// ValidateCandidateComponent refuses a steering component whose direction is
// not the candidate subject bound to the candidate's parent definition, or
// whose candidate carries no direct measurement of the direction grounded in
// an exact tool-surface efficiency trace: propensity claims rest on measured
// tool work, never on prose.
func (SteeringCandidatePlugin) ValidateCandidateComponent(
	ctx context.Context,
	reader artifact.Reader,
	facts runrecord.CandidateAdmissionFacts,
	component runrecord.CandidateAdmissionComponent,
) error {
	direction, err := requireBoundResidualDirection(ctx, reader, facts.Parent, facts.Subject, component.Specification)
	if err != nil {
		return err
	}
	for _, reference := range facts.References {
		if reference.Role != string(CandidateReferenceMeasurement) || reference.Subject != direction.ID {
			continue
		}
		decision, err := recipe.RequireDecision(ctx, reader, reference.Evidence)
		if err != nil {
			return err
		}
		for _, source := range decision.Evidence {
			content, found, readErr := artifact.ReadContent(ctx, reader, source)
			if readErr != nil {
				return readErr
			}
			if !found {
				continue
			}
			trace, parseErr := runrecord.ParseEfficiencyTrace(content.Data)
			if parseErr != nil {
				continue
			}
			if trace.Surface == runrecord.SurfaceTool || trace.Surface == runrecord.SurfaceAgent {
				return nil
			}
		}
	}
	return errors.New("modelrecipe: steering candidate carries no tool-surface efficiency measurement of its direction")
}

// CompileCandidateComponent derives the steered arm, its alpha-zero identity
// ablation, and the exact resident resource bound. The compiled plan carries
// the selector, never an alpha value.
func (SteeringCandidatePlugin) CompileCandidateComponent(
	ctx context.Context,
	reader artifact.Reader,
	facts CandidateCompileFacts,
	component CandidateComponent,
) (CandidateComponentCompilation, error) {
	direction, err := requireBoundResidualDirection(ctx, reader, facts.Parent, facts.Subject, component.Specification)
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	vector, found, err := reader.Artifact(ctx, direction.Vector)
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	if !found || !checked.Nonzero(vector.Size) {
		return CandidateComponentCompilation{}, errors.New("modelrecipe: steering direction vector is absent or empty")
	}
	steered, err := steeringRealizationCodec.NewInitial(SteeringRealization{
		Direction: direction.ID, Selector: direction.Selector, InspectionOnly: true,
	})
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	identity, err := steeringRealizationCodec.NewInitial(SteeringRealization{
		Direction: direction.ID, Selector: direction.Selector, AlphaZero: true,
	})
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	policy, err := steeringResourcePolicyCodec.NewInitial(SteeringResourcePolicy{
		MaxResidentBytes: vector.Size,
		Rationale:        "the intervention holds exactly the direction vector beside the model",
		ReopenTrigger:    "re-extract when the direction, contract, or model definition changes",
	})
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	steeredContent, err := steered.Content()
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	identityContent, err := identity.Content()
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	policyContent, err := policy.Content()
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	lineage := artifact.DependencyLineage(steered.ID, direction.ID)
	lineage = append(lineage, artifact.DependencyLineage(identity.ID, direction.ID)...)
	return CandidateComponentCompilation{
		Plan: CandidateComponentPlan{
			Domain: CandidateSteering, Specification: direction.ID,
			Subject: direction.ID, Realization: steered.ID, ResourcePolicy: policy.ID,
			Inputs: []artifact.ID{
				direction.ID, direction.Contract, direction.Model, direction.Definition,
				direction.Manuals, direction.Readout, direction.Corpus, direction.Split,
				direction.Vector,
			},
			Documents:         []artifact.ID{steered.ID, identity.ID, policy.ID},
			PeakResidentBytes: vector.Size, ArtifactBytes: vector.Size,
		},
		// The identity arm omits the direction vector: the alpha-zero
		// comparison every admission demands.
		Ablations: []CandidateDomainAblation{{Omitted: direction.Vector, Realization: identity.ID}},
		Contents:  []artifact.Content{steeredContent, identityContent, policyContent},
		Lineage:   lineage,
	}, nil
}

func requireBoundResidualDirection(
	ctx context.Context,
	reader artifact.Reader,
	parent, subject, specification artifact.ID,
) (representation.ResidualDirection, error) {
	if ctx == nil || reader == nil || specification.Kind() != artifact.KindRecipe {
		return representation.ResidualDirection{}, errors.New("modelrecipe: steering candidate authority is absent")
	}
	direction, err := representation.RequireResidualDirection(ctx, reader, specification)
	if err != nil {
		return representation.ResidualDirection{}, err
	}
	if direction.ID != subject || direction.Definition != parent {
		return representation.ResidualDirection{}, fmt.Errorf(
			"modelrecipe: steering direction %s is not the candidate subject bound to its parent definition", direction.ID,
		)
	}
	return direction, nil
}
