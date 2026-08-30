package modelrecipe

import (
	"cmp"
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/runrecord"
)

const (
	// CandidateMaterializationMediaType identifies one closed candidate realization.
	CandidateMaterializationMediaType = "application/vnd.overgo.candidate-materialization+json"
	// CandidateMaterializationSchema identifies the immutable realization contract.
	CandidateMaterializationSchema = "overgo/candidate-materialization/v1"
)

// CandidateMaterializedArm binds one primary or declared drop-delta arm to
// the exact produced model and its typed execution evidence.
type CandidateMaterializedArm struct {
	Component         artifact.ID `json:"component"`
	Ablation          artifact.ID `json:"ablation,omitzero"`
	Omitted           artifact.ID `json:"omitted,omitzero"`
	Realization       artifact.ID `json:"realization"`
	Model             artifact.ID `json:"model"`
	TensorInventory   artifact.ID `json:"tensor_inventory"`
	ModelDefinition   artifact.ID `json:"model_definition"`
	Output            artifact.ID `json:"output"`
	Run               artifact.ID `json:"run"`
	Observation       artifact.ID `json:"observation"`
	TensorBytes       uint64      `json:"tensor_bytes"`
	StoredBytes       uint64      `json:"stored_bytes"`
	PeakResidentBytes uint64      `json:"peak_resident_bytes"`
}

// CandidateMaterialization is the content-addressed closure of one supervised
// realize decision. It grants no activation, placement, or serving authority.
type CandidateMaterialization struct {
	Version        uint16                     `json:"version"`
	Decision       artifact.ID                `json:"decision"`
	Candidate      artifact.ID                `json:"candidate"`
	Admission      artifact.ID                `json:"admission"`
	Trial          artifact.ID                `json:"trial"`
	EvaluationPlan artifact.ID                `json:"evaluation_plan"`
	Code           artifact.ID                `json:"code"`
	Environment    artifact.ID                `json:"environment"`
	ResourceCharge artifact.ID                `json:"resource_charge"`
	Arms           []CandidateMaterializedArm `json:"arms"`
	ID             artifact.ID                `json:"-"`
}

var candidateMaterializationCodec = artifact.JSONDocumentCodec(
	"candidate materialization", artifact.KindEvidence,
	CandidateMaterializationMediaType, CandidateMaterializationSchema,
	canonicalizeCandidateMaterialization,
	func(value CandidateMaterialization) artifact.ID { return value.ID },
	func(value *CandidateMaterialization, id artifact.ID) { value.ID = id },
	func(value CandidateMaterialization) CandidateMaterialization {
		value.Arms = slices.Clone(value.Arms)
		return value
	},
)

// NewCandidateMaterialization validates exact decision, compilation, budget,
// primary-arm, and ablation closure before identifying the record.
func NewCandidateMaterialization(
	decision runrecord.DriverDecision,
	compiled CandidateCompilation,
	charge runrecord.BudgetCharge,
	arms []CandidateMaterializedArm,
) (CandidateMaterialization, error) {
	if _, err := decision.Content(); err != nil {
		return CandidateMaterialization{}, err
	}
	if err := charge.ValidateIdentity(); err != nil {
		return CandidateMaterialization{}, err
	}
	trial := compiled.Trial
	if decision.Stop != runrecord.DriverStopContinue || decision.Action != runrecord.DriverActionRealize ||
		decision.Subject != trial.Candidate || decision.Target != trial.Candidate ||
		decision.Need != runrecord.DriverNeedRealization || charge.Budget != decision.Resources.Grant ||
		charge.Consumer != decision.ID {
		return CandidateMaterialization{}, errors.New("model recipe: materialization decision or charge differs")
	}
	selectedCost, selectedAdmission, found := materializationDecisionFacts(decision, trial.Candidate)
	if !found || selectedAdmission != trial.Admission || charge.Amount != selectedCost {
		return CandidateMaterialization{}, errors.New("model recipe: materialization selection differs")
	}
	if err := validateCandidateMaterializedArms(compiled, arms); err != nil {
		return CandidateMaterialization{}, err
	}
	return candidateMaterializationCodec.New(CandidateMaterialization{
		Version: artifact.InitialDocumentVersion, Decision: decision.ID,
		Candidate: trial.Candidate, Admission: trial.Admission, Trial: trial.ID,
		EvaluationPlan: trial.EvaluationPlan, Code: trial.Code, Environment: trial.Environment,
		ResourceCharge: charge.ID, Arms: slices.Clone(arms),
	})
}

// RequireCandidateMaterialization loads one exact realization closure and
// requires its complete authority lineage.
func RequireCandidateMaterialization(
	ctx context.Context, reader artifact.Reader, id artifact.ID,
) (CandidateMaterialization, error) {
	return candidateMaterializationCodec.RequireExactLineage(
		ctx, reader, id, CandidateMaterialization.Lineage,
	)
}

// ValidateIdentity verifies the exact materialization content identity.
func (value CandidateMaterialization) ValidateIdentity() error {
	return candidateMaterializationCodec.ValidateIdentity(value)
}

// Content returns the exact materialization document.
func (value CandidateMaterialization) Content() (artifact.Content, error) {
	return candidateMaterializationCodec.Content(value)
}

// Lineage binds the realization closure to every decision, compilation,
// produced-model, and execution-evidence authority it cites.
func (value CandidateMaterialization) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Decision, value.Candidate, value.Admission, value.Trial, value.EvaluationPlan,
		value.Code, value.Environment, value.ResourceCharge,
	}
	for _, arm := range value.Arms {
		parents = append(parents,
			arm.Component, arm.Ablation, arm.Omitted, arm.Realization, arm.Model,
			arm.TensorInventory, arm.ModelDefinition, arm.Output, arm.Run,
			arm.Observation,
		)
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(parents, artifact.CompareID)
	return artifact.DependencyLineage(value.ID, slices.Compact(parents)...)
}

func materializationDecisionFacts(
	decision runrecord.DriverDecision,
	candidate artifact.ID,
) (uint64, artifact.ID, bool) {
	for _, option := range decision.Options {
		if option.Candidate != candidate || option.State != runrecord.DriverCandidateEligible {
			continue
		}
		for _, gap := range option.Missing {
			if gap.Need == runrecord.DriverNeedRealization && gap.Target == candidate &&
				gap.CostUnit == decision.Resources.Unit {
				return gap.CostUnits, option.Admission, true
			}
		}
	}
	var absentCostUnits uint64
	return absentCostUnits, artifact.ID{}, false
}

func validateCandidateMaterializedArms(
	compiled CandidateCompilation,
	arms []CandidateMaterializedArm,
) error {
	if len(arms) != len(compiled.Components)+len(compiled.Ablations) {
		return errors.New("model recipe: materialization arm closure differs")
	}
	byAblation := make(map[artifact.ID]CandidateAblation, len(compiled.Ablations))
	for _, ablation := range compiled.Ablations {
		byAblation[ablation.ID] = ablation
	}
	actual := slices.Clone(arms)
	slices.SortFunc(actual, compareCandidateMaterializedArms)
	seen := make(map[artifact.ID]bool, len(actual))
	for _, component := range compiled.Components {
		var tensorBytes uint64
		primary := false
		for _, arm := range actual {
			if arm.Component != component.ID {
				continue
			}
			if arm.Ablation.Valid() {
				ablation, found := byAblation[arm.Ablation]
				if !found || !slices.Contains(component.Ablations, arm.Ablation) ||
					ablation.Omitted != arm.Omitted || ablation.Realization != arm.Realization {
					return errors.New("model recipe: materialized ablation differs")
				}
				if seen[arm.Ablation] {
					return errors.New("model recipe: materialized ablation is duplicated")
				}
				seen[arm.Ablation] = true
			} else if primary || arm.Omitted.Valid() || arm.Realization != component.Realization {
				return errors.New("model recipe: materialized primary arm differs")
			} else {
				primary = true
			}
			if err := validateCandidateMaterializedArm(arm, component); err != nil {
				return err
			}
			var ok bool
			tensorBytes, ok = checked.Add64(tensorBytes, arm.TensorBytes)
			if !ok {
				return errors.New("model recipe: materialized tensor extent overflows")
			}
		}
		if !primary || tensorBytes != component.ArtifactBytes || len(seen) > len(compiled.Ablations) {
			return errors.New("model recipe: materialized component closure differs")
		}
	}
	if len(seen) != len(compiled.Ablations) {
		return errors.New("model recipe: materialized ablation closure differs")
	}
	return nil
}

func validateCandidateMaterializedArm(
	arm CandidateMaterializedArm,
	component CandidateComponentPlan,
) error {
	if arm.Component.Kind() != artifact.KindProfile || arm.Realization.Kind() != artifact.KindProfile ||
		arm.Ablation.Valid() && (arm.Ablation.Kind() != artifact.KindProfile ||
			arm.Omitted.Kind() != artifact.KindModelDefinition) ||
		!arm.Ablation.Valid() && arm.Omitted.Valid() || arm.Model.Kind() != artifact.KindModel ||
		arm.TensorInventory.Kind() != artifact.KindTensorInventory ||
		arm.ModelDefinition.Kind() != artifact.KindModelDefinition || arm.Output.Kind() != artifact.KindOutput ||
		arm.Run.Kind() != artifact.KindRun || arm.Observation.Kind() != artifact.KindEvidence ||
		!checked.Nonzero(arm.TensorBytes) || !checked.Nonzero(arm.StoredBytes) ||
		!checked.Nonzero(arm.PeakResidentBytes) ||
		arm.PeakResidentBytes > component.PeakResidentBytes {
		return errors.New("model recipe: invalid materialized arm")
	}
	return nil
}

func canonicalizeCandidateMaterialization(value *CandidateMaterialization) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Decision.Kind() != artifact.KindEvidence || value.Candidate.Kind() != artifact.KindRecipe ||
		value.Admission.Kind() != artifact.KindEvidence || value.Trial.Kind() != artifact.KindProfile ||
		value.EvaluationPlan.Kind() != artifact.KindProfile || value.Code.Kind() != artifact.KindEvidence ||
		value.Environment.Kind() != artifact.KindEvidence || value.ResourceCharge.Kind() != artifact.KindEvidence ||
		!checked.Nonempty(value.Arms) {
		return errors.New("model recipe: invalid candidate materialization")
	}
	value.Arms = slices.Clone(value.Arms)
	slices.SortFunc(value.Arms, compareCandidateMaterializedArms)
	for index, arm := range value.Arms {
		if err := validateCandidateMaterializedArm(arm, CandidateComponentPlan{
			PeakResidentBytes: arm.PeakResidentBytes,
		}); err != nil || index > 0 && compareCandidateMaterializedArms(value.Arms[index-1], arm) == 0 {
			return errors.Join(errors.New("model recipe: invalid candidate materialization arm"), err)
		}
	}
	return nil
}

func compareCandidateMaterializedArms(left, right CandidateMaterializedArm) int {
	return cmp.Or(
		artifact.CompareID(left.Component, right.Component),
		artifact.CompareID(left.Ablation, right.Ablation),
		artifact.CompareID(left.Realization, right.Realization),
	)
}
