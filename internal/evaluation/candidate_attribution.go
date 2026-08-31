package evaluation

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
)

const (
	// CrossDomainEvaluationContractMediaType identifies a predeclared shared evaluator contract.
	CrossDomainEvaluationContractMediaType = "application/vnd.overgo.cross-domain-evaluation-contract+json"
	// CrossDomainEvaluationContractSchema identifies the evaluator contract wire schema.
	CrossDomainEvaluationContractSchema = "overgo/cross-domain-evaluation-contract/v1"
	// CrossDomainCandidateEvaluationMediaType identifies an immutable evaluator outcome.
	CrossDomainCandidateEvaluationMediaType = "application/vnd.overgo.cross-domain-candidate-evaluation+json"
	// CrossDomainCandidateEvaluationSchema identifies the evaluator outcome wire schema.
	CrossDomainCandidateEvaluationSchema = "overgo/cross-domain-candidate-evaluation/v1"

	minimumCandidateFitnessDimensions = 2
)

// CandidateDeltaDimension keeps causal domains separate instead of reducing
// model and system changes to one uninterpretable score.
type CandidateDeltaDimension string

const (
	// CandidateDeltaModel attributes a model-prototype change.
	CandidateDeltaModel CandidateDeltaDimension = "model"
	// CandidateDeltaComposition attributes a model-composition change.
	CandidateDeltaComposition CandidateDeltaDimension = "composition"
	// CandidateDeltaRouting attributes a routing-policy change.
	CandidateDeltaRouting CandidateDeltaDimension = "routing"
	// CandidateDeltaSteering attributes an inference-time steering change.
	CandidateDeltaSteering CandidateDeltaDimension = "steering"
	// CandidateDeltaData attributes a dataset-transform change.
	CandidateDeltaData CandidateDeltaDimension = "data"
	// CandidateDeltaRecipe attributes a mechanism or recipe change.
	CandidateDeltaRecipe CandidateDeltaDimension = "recipe"
	// CandidateDeltaSystem attributes a compiled system-code change.
	CandidateDeltaSystem CandidateDeltaDimension = "system"
)

// CandidateEvaluationRole names one mandatory controlled-comparison arm.
type CandidateEvaluationRole string

const (
	// CandidateEvaluationIncumbent identifies the production baseline arm.
	CandidateEvaluationIncumbent CandidateEvaluationRole = "incumbent"
	// CandidateEvaluationParent identifies one parent-specialist arm.
	CandidateEvaluationParent CandidateEvaluationRole = "parent"
	// CandidateEvaluationJoint identifies the complete candidate arm.
	CandidateEvaluationJoint CandidateEvaluationRole = "joint"
	// CandidateEvaluationAblation identifies one dropped-delta arm.
	CandidateEvaluationAblation CandidateEvaluationRole = "ablation"
)

// CandidateEvaluationVerdict is evidence only; it is deliberately not a
// lifecycle, alias, placement, or promotion transition.
type CandidateEvaluationVerdict string

const (
	// CandidateEvaluationImproved records a policy-satisfying improvement.
	CandidateEvaluationImproved CandidateEvaluationVerdict = "improved"
	// CandidateEvaluationRefused records an evidentiary or fitness refusal.
	CandidateEvaluationRefused CandidateEvaluationVerdict = "refused"
)

// CandidateFitnessRequirement declares a measurable directional threshold
// before any arm is observed. Metrics remain a vector and are never weighted.
type CandidateFitnessRequirement struct {
	Name               string              `json:"name"`
	Direction          runrecord.Direction `json:"direction"`
	MinimumImprovement float64             `json:"minimum_improvement"`
}

// CandidateAcceptedTradeoff is an explicit, authority-bound exception to
// Pareto non-regression. It is part of the pre-execution contract, never
// inferred from a candidate's observed result.
type CandidateAcceptedTradeoff struct {
	RegressedMetric    string      `json:"regressed_metric"`
	MaximumRegression  float64     `json:"maximum_regression"`
	ImprovedMetric     string      `json:"improved_metric"`
	MinimumImprovement float64     `json:"minimum_improvement"`
	Authority          artifact.ID `json:"authority"`
}

// CandidateAblationContract binds every compiler-declared omitted delta to
// the causal dimension its plugin owns.
type CandidateAblationContract struct {
	Ablation  artifact.ID             `json:"ablation"`
	Omitted   artifact.ID             `json:"omitted"`
	Dimension CandidateDeltaDimension `json:"dimension"`
}

// CrossDomainEvaluationContract freezes the exact driver choice, candidate
// compilation, promotion inputs, arm population, and vector fitness policy
// before measurements exist.
type CrossDomainEvaluationContract struct {
	Version             uint16                        `json:"version"`
	Decision            artifact.ID                   `json:"decision"`
	Candidate           artifact.ID                   `json:"candidate"`
	Admission           artifact.ID                   `json:"admission"`
	Trial               artifact.ID                   `json:"trial"`
	EvaluationPlan      artifact.ID                   `json:"evaluation_plan"`
	Incumbent           artifact.ID                   `json:"incumbent"`
	Parents             []artifact.ID                 `json:"parents"`
	Ablations           []CandidateAblationContract   `json:"ablations"`
	CandidateDimensions []CandidateDeltaDimension     `json:"candidate_dimensions"`
	PromotionSplit      artifact.ID                   `json:"promotion_split"`
	PromotionBudget     artifact.ID                   `json:"promotion_budget"`
	Evaluator           artifact.ID                   `json:"evaluator"`
	Inputs              artifact.ID                   `json:"inputs"`
	Requirements        []CandidateFitnessRequirement `json:"requirements"`
	AcceptedTradeoffs   []CandidateAcceptedTradeoff   `json:"accepted_tradeoffs,omitempty"`
	ID                  artifact.ID                   `json:"-"`
}

// CandidateEvaluationArm is one immutable measurement under the exact shared
// evaluator, inputs, promotion split, budget, coverage, and metric vector.
type CandidateEvaluationArm struct {
	Role       CandidateEvaluationRole   `json:"role"`
	Subject    artifact.ID               `json:"subject"`
	Ablation   artifact.ID               `json:"ablation,omitzero"`
	Omitted    artifact.ID               `json:"omitted,omitzero"`
	Dimensions []CandidateDeltaDimension `json:"dimensions,omitempty"`
	Evaluator  artifact.ID               `json:"evaluator"`
	Inputs     artifact.ID               `json:"inputs"`
	Split      artifact.ID               `json:"split"`
	Budget     artifact.ID               `json:"budget"`
	Evidence   artifact.ID               `json:"evidence"`
	Fitness    []runrecord.Metric        `json:"fitness"`
	Covered    uint64                    `json:"covered"`
	Total      uint64                    `json:"total"`
}

// CandidateAblationContribution records the joint-minus-dropped-delta vector.
// It is attribution evidence, not a scalar contribution score.
type CandidateAblationContribution struct {
	Ablation artifact.ID        `json:"ablation"`
	Omitted  artifact.ID        `json:"omitted"`
	Delta    []runrecord.Metric `json:"delta"`
}

// CrossDomainCandidateEvaluation is the immutable shared-evaluator fact. A
// refusal is retained as an outcome so missing arms, leakage, and regressions
// cannot disappear as transient errors.
type CrossDomainCandidateEvaluation struct {
	Version             uint16                          `json:"version"`
	Contract            CrossDomainEvaluationContract   `json:"contract"`
	Arms                []CandidateEvaluationArm        `json:"arms"`
	Contributions       []CandidateAblationContribution `json:"contributions,omitempty"`
	Verdict             CandidateEvaluationVerdict      `json:"verdict"`
	Reasons             []string                        `json:"reasons,omitempty"`
	AcceptedAuthorities []artifact.ID                   `json:"accepted_authorities,omitempty"`
	ID                  artifact.ID                     `json:"-"`
}

var crossDomainEvaluationContractCodec = artifact.JSONDocumentCodec(
	"cross-domain evaluation contract", artifact.KindProfile,
	CrossDomainEvaluationContractMediaType, CrossDomainEvaluationContractSchema,
	canonicalizeCrossDomainEvaluationContract,
	func(value CrossDomainEvaluationContract) artifact.ID { return value.ID },
	func(value *CrossDomainEvaluationContract, id artifact.ID) { value.ID = id },
	cloneCrossDomainEvaluationContract,
)

var crossDomainCandidateEvaluationCodec = artifact.JSONDocumentCodec(
	"cross-domain candidate evaluation", artifact.KindEvidence,
	CrossDomainCandidateEvaluationMediaType, CrossDomainCandidateEvaluationSchema,
	canonicalizeCrossDomainCandidateEvaluation,
	func(value CrossDomainCandidateEvaluation) artifact.ID { return value.ID },
	func(value *CrossDomainCandidateEvaluation, id artifact.ID) { value.ID = id },
	cloneCrossDomainCandidateEvaluation,
)

// NewCrossDomainEvaluationContract derives the complete arm population and
// causal dimensions from the exact compiler output selected by the driver.
func NewCrossDomainEvaluationContract(
	decision runrecord.DriverDecision,
	compiled modelrecipe.CandidateCompilation,
	evaluator, inputs artifact.ID,
	requirements []CandidateFitnessRequirement,
	tradeoffs []CandidateAcceptedTradeoff,
) (CrossDomainEvaluationContract, error) {
	if _, err := decision.Content(); err != nil {
		return CrossDomainEvaluationContract{}, err
	}
	if err := compiled.ValidateIdentity(); err != nil {
		return CrossDomainEvaluationContract{}, err
	}
	trial, plan := compiled.Trial, compiled.EvaluationPlan
	if decision.Stop != runrecord.DriverStopContinue ||
		decision.Action != runrecord.DriverActionRealize && decision.Action != runrecord.DriverActionEvaluate &&
			decision.Action != runrecord.DriverActionAblate || decision.Subject != trial.Candidate {
		return CrossDomainEvaluationContract{}, errors.New("evaluation: driver decision does not select this candidate spine")
	}
	selected := false
	for _, option := range decision.Options {
		if option.Candidate == trial.Candidate && option.Admission == trial.Admission {
			selected = true
			break
		}
	}
	if !selected {
		return CrossDomainEvaluationContract{}, errors.New("evaluation: driver decision omits the candidate admission")
	}
	dimensions := make([]CandidateDeltaDimension, len(compiled.Components))
	bySpecification := make(map[artifact.ID]CandidateDeltaDimension, len(compiled.Components))
	for index, component := range compiled.Components {
		dimension, err := candidateDimension(component.Domain)
		if err != nil {
			return CrossDomainEvaluationContract{}, err
		}
		dimensions[index] = dimension
		bySpecification[component.Specification] = dimension
	}
	dimensions = canonicalCandidateDimensions(dimensions)
	ablative := make([]CandidateAblationContract, len(compiled.Ablations))
	for index, ablation := range compiled.Ablations {
		ablative[index] = CandidateAblationContract{
			Ablation: ablation.ID, Omitted: ablation.Omitted,
			Dimension: bySpecification[ablation.Specification],
		}
	}
	return crossDomainEvaluationContractCodec.New(CrossDomainEvaluationContract{
		Version:  artifact.InitialDocumentVersion,
		Decision: decision.ID, Candidate: trial.Candidate, Admission: trial.Admission,
		Trial: trial.ID, EvaluationPlan: plan.ID, Incumbent: decision.Incumbent,
		Parents: []artifact.ID{trial.Parent}, Ablations: ablative, CandidateDimensions: dimensions,
		PromotionSplit: plan.PromotionSplit, PromotionBudget: plan.PromotionBudget,
		Evaluator: evaluator, Inputs: inputs,
		Requirements: slices.Clone(requirements), AcceptedTradeoffs: slices.Clone(tradeoffs),
	})
}

// EvaluateCrossDomainCandidate evaluates a completed arm set. Evidentiary
// failure returns a content-addressed refusal; only malformed authorities
// return an error.
func EvaluateCrossDomainCandidate(
	contract CrossDomainEvaluationContract,
	decision runrecord.DriverDecision,
	compiled modelrecipe.CandidateCompilation,
	arms []CandidateEvaluationArm,
) (CrossDomainCandidateEvaluation, error) {
	expected, err := NewCrossDomainEvaluationContract(
		decision, compiled, contract.Evaluator, contract.Inputs,
		contract.Requirements, contract.AcceptedTradeoffs,
	)
	if err != nil || expected.ID != contract.ID {
		return CrossDomainCandidateEvaluation{}, errors.Join(
			errors.New("evaluation: evaluation contract differs from exact authorities"), err,
		)
	}
	arms = cloneCandidateEvaluationArms(arms)
	for index := range arms {
		if err := canonicalizeCandidateEvaluationArm(&arms[index]); err != nil {
			return CrossDomainCandidateEvaluation{}, err
		}
	}
	slices.SortFunc(arms, compareCandidateEvaluationArms)
	reasons, accepted := evaluateCandidateArms(contract, arms)
	contributions := candidateAblationContributions(contract, arms)
	verdict := CandidateEvaluationImproved
	if len(reasons) != 0 {
		verdict = CandidateEvaluationRefused
	}
	return crossDomainCandidateEvaluationCodec.New(CrossDomainCandidateEvaluation{
		Version: artifact.InitialDocumentVersion, Contract: contract, Arms: arms,
		Contributions: contributions, Verdict: verdict, Reasons: reasons,
		AcceptedAuthorities: accepted,
	})
}

// Content returns the immutable pre-execution evaluator contract.
func (value CrossDomainEvaluationContract) Content() (artifact.Content, error) {
	return crossDomainEvaluationContractCodec.Content(value)
}

// Lineage binds the contract to every input authority available before measurement.
func (value CrossDomainEvaluationContract) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Decision, value.Candidate, value.Admission, value.Trial, value.EvaluationPlan,
		value.Incumbent, value.PromotionSplit, value.PromotionBudget, value.Evaluator, value.Inputs,
	}
	parents = append(parents, value.Parents...)
	for _, ablation := range value.Ablations {
		parents = append(parents, ablation.Ablation, ablation.Omitted)
	}
	for _, tradeoff := range value.AcceptedTradeoffs {
		parents = append(parents, tradeoff.Authority)
	}
	return artifact.DependencyLineage(value.ID, uniqueCandidateEvaluationIDs(parents)...)
}

// Content returns the immutable evaluation outcome.
func (value CrossDomainCandidateEvaluation) Content() (artifact.Content, error) {
	return crossDomainCandidateEvaluationCodec.Content(value)
}

// Lineage binds the outcome to the frozen contract and exact arm evidence.
func (value CrossDomainCandidateEvaluation) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Contract.ID}
	for _, arm := range value.Arms {
		parents = append(parents, arm.Subject, arm.Ablation, arm.Omitted, arm.Evidence)
	}
	parents = append(parents, value.AcceptedAuthorities...)
	return artifact.DependencyLineage(value.ID, uniqueCandidateEvaluationIDs(parents)...)
}

func evaluateCandidateArms(
	contract CrossDomainEvaluationContract,
	arms []CandidateEvaluationArm,
) ([]string, []artifact.ID) {
	var reasons []string
	byRole := make(map[CandidateEvaluationRole][]CandidateEvaluationArm)
	for _, arm := range arms {
		byRole[arm.Role] = append(byRole[arm.Role], arm)
		if arm.Evaluator != contract.Evaluator || arm.Inputs != contract.Inputs ||
			arm.Split != contract.PromotionSplit || arm.Budget != contract.PromotionBudget {
			reasons = append(reasons, "arm differs from shared evaluator inputs, promotion split, or budget")
		}
		if arm.Covered != arm.Total {
			reasons = append(reasons, "arm coverage is incomplete")
		}
	}
	incumbent := matchingArm(byRole[CandidateEvaluationIncumbent], contract.Incumbent, artifact.ID{})
	joint := matchingArm(byRole[CandidateEvaluationJoint], contract.Candidate, artifact.ID{})
	if incumbent == nil {
		reasons = append(reasons, "production incumbent arm is missing")
	}
	if joint == nil {
		reasons = append(reasons, "joint candidate arm is missing")
	}
	for _, parent := range contract.Parents {
		if matchingArm(byRole[CandidateEvaluationParent], parent, artifact.ID{}) == nil {
			reasons = append(reasons, "parent specialist arm is missing: "+parent.String())
		}
	}
	for _, ablation := range contract.Ablations {
		arm := matchingArm(byRole[CandidateEvaluationAblation], artifact.ID{}, ablation.Ablation)
		if arm == nil || arm.Omitted != ablation.Omitted ||
			!slices.Equal(arm.Dimensions, []CandidateDeltaDimension{ablation.Dimension}) {
			reasons = append(reasons, "declared dropped-delta ablation arm is missing: "+ablation.Ablation.String())
		}
	}
	wantCount := 2 + len(contract.Parents) + len(contract.Ablations)
	if len(arms) != wantCount {
		reasons = append(reasons, "evaluation arm population differs from the predeclared contract")
	}
	var accepted []artifact.ID
	if incumbent != nil && joint != nil {
		fitnessReasons, authorities := evaluateCandidateFitness(contract, *incumbent, *joint)
		reasons = append(reasons, fitnessReasons...)
		accepted = append(accepted, authorities...)
	}
	slices.Sort(reasons)
	reasons = slices.Compact(reasons)
	slices.SortFunc(accepted, artifact.CompareID)
	return reasons, slices.Compact(accepted)
}

func evaluateCandidateFitness(
	contract CrossDomainEvaluationContract,
	baseline, candidate CandidateEvaluationArm,
) ([]string, []artifact.ID) {
	if !slices.Equal(candidate.Dimensions, contract.CandidateDimensions) {
		return []string{"joint candidate causal dimensions differ from the compiled plugins"}, nil
	}
	baselineMetrics := metricsByName(baseline.Fitness)
	candidateMetrics := metricsByName(candidate.Fitness)
	var reasons []string
	var accepted []artifact.ID
	strict := false
	for _, requirement := range contract.Requirements {
		before, baselineFound := baselineMetrics[requirement.Name]
		after, candidateFound := candidateMetrics[requirement.Name]
		if !baselineFound || !candidateFound || before.Direction != requirement.Direction ||
			after.Direction != requirement.Direction || before.Unit != after.Unit {
			reasons = append(reasons, "fitness vector differs for metric "+requirement.Name)
			continue
		}
		delta := directionalDelta(requirement.Direction, before.Value, after.Value)
		if delta >= requirement.MinimumImprovement {
			strict = true
		}
		if delta >= 0 {
			continue
		}
		tradeoff, found := acceptedCandidateTradeoff(contract, requirement.Name, -delta, baselineMetrics, candidateMetrics)
		if !found {
			reasons = append(reasons, "candidate regresses metric "+requirement.Name)
			continue
		}
		accepted = append(accepted, tradeoff.Authority)
	}
	if len(baseline.Fitness) != len(contract.Requirements) || len(candidate.Fitness) != len(contract.Requirements) {
		reasons = append(reasons, "fitness vector contains undeclared or missing metrics")
	}
	if !strict {
		reasons = append(reasons, "candidate has no predeclared measurable improvement")
	}
	return reasons, accepted
}

func acceptedCandidateTradeoff(
	contract CrossDomainEvaluationContract,
	regressed string,
	regression float64,
	baseline, candidate map[string]runrecord.Metric,
) (CandidateAcceptedTradeoff, bool) {
	for _, tradeoff := range contract.AcceptedTradeoffs {
		before, beforeFound := baseline[tradeoff.ImprovedMetric]
		after, afterFound := candidate[tradeoff.ImprovedMetric]
		if tradeoff.RegressedMetric == regressed && regression <= tradeoff.MaximumRegression &&
			beforeFound && afterFound && before.Direction == after.Direction && before.Unit == after.Unit &&
			directionalDelta(before.Direction, before.Value, after.Value) >= tradeoff.MinimumImprovement {
			return tradeoff, true
		}
	}
	return CandidateAcceptedTradeoff{}, false
}

func candidateAblationContributions(
	contract CrossDomainEvaluationContract,
	arms []CandidateEvaluationArm,
) []CandidateAblationContribution {
	var joint *CandidateEvaluationArm
	byAblation := make(map[artifact.ID]CandidateEvaluationArm)
	for index := range arms {
		if arms[index].Role == CandidateEvaluationJoint && arms[index].Subject == contract.Candidate {
			joint = &arms[index]
		}
		if arms[index].Role == CandidateEvaluationAblation {
			byAblation[arms[index].Ablation] = arms[index]
		}
	}
	if joint == nil {
		return nil
	}
	jointMetrics := metricsByName(joint.Fitness)
	result := make([]CandidateAblationContribution, 0, len(contract.Ablations))
	for _, declared := range contract.Ablations {
		arm, found := byAblation[declared.Ablation]
		if !found {
			continue
		}
		armMetrics := metricsByName(arm.Fitness)
		delta := make([]runrecord.Metric, 0, len(contract.Requirements))
		for _, requirement := range contract.Requirements {
			with, withFound := jointMetrics[requirement.Name]
			without, withoutFound := armMetrics[requirement.Name]
			if !withFound || !withoutFound || with.Direction != without.Direction || with.Unit != without.Unit {
				continue
			}
			delta = append(delta, runrecord.Metric{
				Name: requirement.Name, Unit: with.Unit, Direction: requirement.Direction,
				Value: directionalDelta(requirement.Direction, without.Value, with.Value),
			})
		}
		result = append(result, CandidateAblationContribution{
			Ablation: declared.Ablation, Omitted: declared.Omitted, Delta: delta,
		})
	}
	return result
}

func canonicalizeCrossDomainEvaluationContract(value *CrossDomainEvaluationContract) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Decision.Kind() != artifact.KindEvidence || value.Candidate.Kind() != artifact.KindRecipe ||
		value.Admission.Kind() != artifact.KindEvidence || value.Trial.Kind() != artifact.KindProfile ||
		value.EvaluationPlan.Kind() != artifact.KindProfile || value.Incumbent.Kind() != artifact.KindRecipe ||
		value.PromotionSplit.Kind() != artifact.KindDatasetShard || value.PromotionBudget.Kind() != artifact.KindEvidence ||
		value.Evaluator.Kind() != artifact.KindProfile || !value.Inputs.Valid() || len(value.Parents) == 0 ||
		len(value.Ablations) == 0 || len(value.CandidateDimensions) == 0 ||
		len(value.Requirements) < minimumCandidateFitnessDimensions {
		return errors.New("evaluation: invalid cross-domain evaluation contract")
	}
	value.Parents = uniqueCandidateEvaluationIDs(value.Parents)
	value.CandidateDimensions = canonicalCandidateDimensions(value.CandidateDimensions)
	slices.SortFunc(value.Ablations, func(left, right CandidateAblationContract) int {
		return artifact.CompareID(left.Ablation, right.Ablation)
	})
	for index, ablation := range value.Ablations {
		if ablation.Ablation.Kind() != artifact.KindProfile || !ablation.Omitted.Valid() || !validCandidateDimension(ablation.Dimension) ||
			index > 0 && ablation.Ablation == value.Ablations[index-1].Ablation {
			return errors.New("evaluation: invalid candidate ablation contract")
		}
	}
	value.Requirements = slices.Clone(value.Requirements)
	slices.SortFunc(value.Requirements, func(left, right CandidateFitnessRequirement) int {
		return strings.Compare(left.Name, right.Name)
	})
	positive := false
	for index, requirement := range value.Requirements {
		if strings.TrimSpace(requirement.Name) != requirement.Name || requirement.Name == "" ||
			!finite(requirement.MinimumImprovement) || requirement.MinimumImprovement <= 0 ||
			requirement.Direction != runrecord.DirectionMinimize && requirement.Direction != runrecord.DirectionMaximize ||
			index > 0 && requirement.Name == value.Requirements[index-1].Name {
			return errors.New("evaluation: invalid candidate fitness requirement")
		}
		positive = true
	}
	if !positive {
		return errors.New("evaluation: evaluation contract has no measurable threshold")
	}
	value.AcceptedTradeoffs = slices.Clone(value.AcceptedTradeoffs)
	slices.SortFunc(value.AcceptedTradeoffs, func(left, right CandidateAcceptedTradeoff) int {
		if order := strings.Compare(left.RegressedMetric, right.RegressedMetric); order != 0 {
			return order
		}
		return artifact.CompareID(left.Authority, right.Authority)
	})
	for _, tradeoff := range value.AcceptedTradeoffs {
		if tradeoff.RegressedMetric == tradeoff.ImprovedMetric || tradeoff.Authority.Kind() != artifact.KindEvidence ||
			!finite(tradeoff.MaximumRegression) || tradeoff.MaximumRegression <= 0 ||
			!finite(tradeoff.MinimumImprovement) || tradeoff.MinimumImprovement <= 0 ||
			!hasFitnessRequirement(value.Requirements, tradeoff.RegressedMetric) ||
			!hasFitnessRequirement(value.Requirements, tradeoff.ImprovedMetric) {
			return errors.New("evaluation: invalid accepted candidate tradeoff")
		}
	}
	return nil
}

func canonicalizeCrossDomainCandidateEvaluation(value *CrossDomainCandidateEvaluation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		crossDomainEvaluationContractCodec.ValidateIdentity(value.Contract) != nil {
		return errors.New("evaluation: invalid cross-domain candidate evaluation")
	}
	value.Arms = cloneCandidateEvaluationArms(value.Arms)
	slices.SortFunc(value.Arms, compareCandidateEvaluationArms)
	for index := range value.Arms {
		if err := canonicalizeCandidateEvaluationArm(&value.Arms[index]); err != nil {
			return err
		}
	}
	value.Reasons = slices.Clone(value.Reasons)
	slices.Sort(value.Reasons)
	value.Reasons = slices.Compact(value.Reasons)
	value.AcceptedAuthorities = uniqueCandidateEvaluationIDs(value.AcceptedAuthorities)
	if value.Verdict == CandidateEvaluationImproved && len(value.Reasons) != 0 ||
		value.Verdict == CandidateEvaluationRefused && len(value.Reasons) == 0 ||
		value.Verdict != CandidateEvaluationImproved && value.Verdict != CandidateEvaluationRefused {
		return errors.New("evaluation: candidate evaluation verdict differs from reasons")
	}
	for index := range value.Contributions {
		contribution := &value.Contributions[index]
		if contribution.Ablation.Kind() != artifact.KindProfile || !contribution.Omitted.Valid() || len(contribution.Delta) == 0 {
			return errors.New("evaluation: invalid candidate ablation contribution")
		}
		if err := canonicalizeCandidateMetrics(&contribution.Delta); err != nil {
			return err
		}
	}
	slices.SortFunc(value.Contributions, func(left, right CandidateAblationContribution) int {
		return artifact.CompareID(left.Ablation, right.Ablation)
	})
	return nil
}

func canonicalizeCandidateEvaluationArm(value *CandidateEvaluationArm) error {
	if value == nil || !value.Subject.Valid() || value.Evaluator.Kind() != artifact.KindProfile ||
		!value.Inputs.Valid() || value.Split.Kind() != artifact.KindDatasetShard ||
		value.Budget.Kind() != artifact.KindEvidence || value.Evidence.Kind() != artifact.KindEvidence ||
		value.Total == 0 || value.Covered > value.Total ||
		len(value.Fitness) < minimumCandidateFitnessDimensions {
		return errors.New("evaluation: invalid candidate evaluation arm")
	}
	value.Dimensions = canonicalCandidateDimensions(value.Dimensions)
	switch value.Role {
	case CandidateEvaluationIncumbent, CandidateEvaluationParent:
		if value.Ablation.Valid() || value.Omitted.Valid() || len(value.Dimensions) != 0 {
			return errors.New("evaluation: baseline arm claims a candidate delta")
		}
	case CandidateEvaluationJoint:
		if value.Ablation.Valid() || value.Omitted.Valid() || len(value.Dimensions) == 0 {
			return errors.New("evaluation: invalid joint candidate arm")
		}
	case CandidateEvaluationAblation:
		if value.Ablation.Kind() != artifact.KindProfile || !value.Omitted.Valid() || len(value.Dimensions) != 1 {
			return errors.New("evaluation: invalid candidate ablation arm")
		}
	default:
		return errors.New("evaluation: invalid candidate evaluation role")
	}
	return canonicalizeCandidateMetrics(&value.Fitness)
}

func canonicalizeCandidateMetrics(values *[]runrecord.Metric) error {
	*values = slices.Clone(*values)
	slices.SortFunc(*values, func(left, right runrecord.Metric) int { return strings.Compare(left.Name, right.Name) })
	for index, metric := range *values {
		if strings.TrimSpace(metric.Name) != metric.Name || metric.Name == "" || !finite(metric.Value) ||
			metric.Direction != runrecord.DirectionMinimize && metric.Direction != runrecord.DirectionMaximize ||
			index > 0 && metric.Name == (*values)[index-1].Name {
			return errors.New("evaluation: invalid candidate fitness vector")
		}
	}
	return nil
}

func candidateDimension(domain modelrecipe.CandidateDomain) (CandidateDeltaDimension, error) {
	switch domain {
	case modelrecipe.CandidateModelPrototype:
		return CandidateDeltaModel, nil
	case modelrecipe.CandidateComposition:
		return CandidateDeltaComposition, nil
	case modelrecipe.CandidateRouting:
		return CandidateDeltaRouting, nil
	case modelrecipe.CandidateSteering:
		return CandidateDeltaSteering, nil
	case modelrecipe.CandidateDatasetTransform:
		return CandidateDeltaData, nil
	case modelrecipe.CandidateMechanism:
		return CandidateDeltaRecipe, nil
	case modelrecipe.CandidateCode:
		return CandidateDeltaSystem, nil
	default:
		return "", fmt.Errorf("evaluation: unsupported candidate evaluation domain %q", domain)
	}
}

func validCandidateDimension(value CandidateDeltaDimension) bool {
	return slices.Contains([]CandidateDeltaDimension{
		CandidateDeltaModel, CandidateDeltaComposition, CandidateDeltaRouting, CandidateDeltaSteering,
		CandidateDeltaData, CandidateDeltaRecipe, CandidateDeltaSystem,
	}, value)
}

func canonicalCandidateDimensions(values []CandidateDeltaDimension) []CandidateDeltaDimension {
	values = slices.Clone(values)
	slices.Sort(values)
	return slices.Compact(values)
}

func matchingArm(arms []CandidateEvaluationArm, subject, ablation artifact.ID) *CandidateEvaluationArm {
	for index := range arms {
		if subject.Valid() && arms[index].Subject == subject || ablation.Valid() && arms[index].Ablation == ablation {
			return &arms[index]
		}
	}
	return nil
}

func metricsByName(values []runrecord.Metric) map[string]runrecord.Metric {
	result := make(map[string]runrecord.Metric, len(values))
	for _, value := range values {
		result[value.Name] = value
	}
	return result
}

func directionalDelta(direction runrecord.Direction, baseline, candidate float64) float64 {
	if direction == runrecord.DirectionMinimize {
		return baseline - candidate
	}
	return candidate - baseline
}

func hasFitnessRequirement(values []CandidateFitnessRequirement, name string) bool {
	return slices.ContainsFunc(values, func(value CandidateFitnessRequirement) bool { return value.Name == name })
}

func compareCandidateEvaluationArms(left, right CandidateEvaluationArm) int {
	if order := strings.Compare(string(left.Role), string(right.Role)); order != 0 {
		return order
	}
	if order := artifact.CompareID(left.Ablation, right.Ablation); order != 0 {
		return order
	}
	return artifact.CompareID(left.Subject, right.Subject)
}

func uniqueCandidateEvaluationIDs(values []artifact.ID) []artifact.ID {
	values = slices.DeleteFunc(slices.Clone(values), func(value artifact.ID) bool { return !value.Valid() })
	return uniqueArtifactIDs(values)
}

func cloneCrossDomainEvaluationContract(value CrossDomainEvaluationContract) CrossDomainEvaluationContract {
	value.Parents = slices.Clone(value.Parents)
	value.Ablations = slices.Clone(value.Ablations)
	value.CandidateDimensions = slices.Clone(value.CandidateDimensions)
	value.Requirements = slices.Clone(value.Requirements)
	value.AcceptedTradeoffs = slices.Clone(value.AcceptedTradeoffs)
	return value
}

func cloneCandidateEvaluationArms(values []CandidateEvaluationArm) []CandidateEvaluationArm {
	values = slices.Clone(values)
	for index := range values {
		values[index].Dimensions = slices.Clone(values[index].Dimensions)
		values[index].Fitness = slices.Clone(values[index].Fitness)
	}
	return values
}

func cloneCrossDomainCandidateEvaluation(value CrossDomainCandidateEvaluation) CrossDomainCandidateEvaluation {
	value.Contract = cloneCrossDomainEvaluationContract(value.Contract)
	value.Arms = cloneCandidateEvaluationArms(value.Arms)
	value.Contributions = slices.Clone(value.Contributions)
	for index := range value.Contributions {
		value.Contributions[index].Delta = slices.Clone(value.Contributions[index].Delta)
	}
	value.Reasons = slices.Clone(value.Reasons)
	value.AcceptedAuthorities = slices.Clone(value.AcceptedAuthorities)
	return value
}
