package modelrecipe

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
)

const (
	// CandidateComponentPlanMediaType identifies one common domain-plan envelope.
	CandidateComponentPlanMediaType = "application/vnd.overgo.candidate-component-plan+json"
	// CandidateComponentPlanSchema identifies the common domain-plan contract.
	CandidateComponentPlanSchema = "overgo/candidate-component-plan/v1"
	// CandidateAblationMediaType identifies one common drop-delta comparison arm.
	CandidateAblationMediaType = "application/vnd.overgo.candidate-ablation+json"
	// CandidateAblationSchema identifies the common ablation contract.
	CandidateAblationSchema = "overgo/candidate-ablation/v1"
	// CandidateEvaluationPlanMediaType identifies the common pre-execution evaluation intent.
	CandidateEvaluationPlanMediaType = "application/vnd.overgo.candidate-evaluation-plan+json"
	// CandidateEvaluationPlanSchema identifies the common pre-execution evaluation-plan contract.
	CandidateEvaluationPlanSchema = "overgo/candidate-evaluation-plan/v1"
	// CandidateTrialMediaType identifies one closed-world cross-domain trial.
	CandidateTrialMediaType = "application/vnd.overgo.candidate-trial+json"
	// CandidateTrialSchema identifies the closed-world trial contract.
	CandidateTrialSchema = "overgo/candidate-trial/v1"
)

// CandidateComponentPlan is the content-addressed common envelope around one
// domain result. Documents are newly derived immutable facts; Inputs are
// existing authorities and are never copied into the result.
type CandidateComponentPlan struct {
	Version           uint16          `json:"version"`
	Domain            CandidateDomain `json:"domain"`
	Specification     artifact.ID     `json:"specification"`
	Subject           artifact.ID     `json:"subject"`
	Realization       artifact.ID     `json:"realization"`
	ResourcePolicy    artifact.ID     `json:"resource_policy"`
	Inputs            []artifact.ID   `json:"inputs"`
	Documents         []artifact.ID   `json:"documents"`
	Ablations         []artifact.ID   `json:"ablations"`
	PeakResidentBytes uint64          `json:"peak_resident_bytes"`
	ArtifactBytes     uint64          `json:"artifact_bytes"`
	ID                artifact.ID     `json:"-"`
}

// CandidateAblation binds one omitted domain delta to its alternate realization.
// Only the common compiler can create this causal comparison authority.
type CandidateAblation struct {
	Version       uint16          `json:"version"`
	Candidate     artifact.ID     `json:"candidate"`
	Admission     artifact.ID     `json:"admission"`
	Domain        CandidateDomain `json:"domain"`
	Specification artifact.ID     `json:"specification"`
	Omitted       artifact.ID     `json:"omitted"`
	Realization   artifact.ID     `json:"realization"`
	ID            artifact.ID     `json:"-"`
}

// CandidateEvaluationPlan binds budgets, falsifier stop authority, causal
// references, component plans, and declared ablations before execution.
type CandidateEvaluationPlan struct {
	Version           uint16                    `json:"version"`
	Candidate         artifact.ID               `json:"candidate"`
	Admission         artifact.ID               `json:"admission"`
	Subject           artifact.ID               `json:"subject"`
	Parent            artifact.ID               `json:"parent"`
	Prediction        recipe.SteeringPrediction `json:"prediction"`
	CostUnit          string                    `json:"cost_unit"`
	Falsifier         artifact.ID               `json:"falsifier"`
	DevelopmentSplit  artifact.ID               `json:"development_split"`
	PromotionSplit    artifact.ID               `json:"promotion_split"`
	DevelopmentBudget artifact.ID               `json:"development_budget"`
	PromotionBudget   artifact.ID               `json:"promotion_budget"`
	ComponentPlans    []artifact.ID             `json:"component_plans"`
	Ablations         []artifact.ID             `json:"ablations"`
	CausalReferences  []CandidateReference      `json:"causal_references"`
	Code              artifact.ID               `json:"code"`
	Environment       artifact.ID               `json:"environment"`
	ID                artifact.ID               `json:"-"`
}

// CandidateTrial binds one admitted candidate to the exact common component
// plans and evaluation intent. It carries no mutable state or runtime authority.
type CandidateTrial struct {
	Version        uint16        `json:"version"`
	Candidate      artifact.ID   `json:"candidate"`
	Admission      artifact.ID   `json:"admission"`
	Subject        artifact.ID   `json:"subject"`
	Parent         artifact.ID   `json:"parent"`
	EvaluationPlan artifact.ID   `json:"evaluation_plan"`
	Components     []artifact.ID `json:"components"`
	Ablations      []artifact.ID `json:"ablations"`
	Code           artifact.ID   `json:"code"`
	Environment    artifact.ID   `json:"environment"`
	ID             artifact.ID   `json:"-"`
}

// CandidateCompilation is one uncommitted, deterministic closed-world trial.
// A later common materialization owner may publish these facts atomically.
type CandidateCompilation struct {
	Trial          CandidateTrial
	EvaluationPlan CandidateEvaluationPlan
	Components     []CandidateComponentPlan
	Ablations      []CandidateAblation
	Contents       []artifact.Content
	Lineage        []artifact.Lineage
}

// ValidateIdentity verifies the complete uncommitted compilation closure. It
// is intentionally read-only: downstream evaluators may consume the exact
// compiler result without gaining publication, execution, or lifecycle
// authority.
func (value CandidateCompilation) ValidateIdentity() error {
	if err := errors.Join(
		candidateTrialCodec.ValidateIdentity(value.Trial),
		candidateEvaluationPlanCodec.ValidateIdentity(value.EvaluationPlan),
	); err != nil {
		return err
	}
	componentIDs := make([]artifact.ID, len(value.Components))
	for index, component := range value.Components {
		if err := candidateComponentPlanCodec.ValidateIdentity(component); err != nil {
			return err
		}
		componentIDs[index] = component.ID
	}
	ablationIDs := make([]artifact.ID, len(value.Ablations))
	for index, ablation := range value.Ablations {
		if err := candidateAblationCodec.ValidateIdentity(ablation); err != nil {
			return err
		}
		ablationIDs[index] = ablation.ID
	}
	trial, plan := value.Trial, value.EvaluationPlan
	if trial.EvaluationPlan != plan.ID || trial.Candidate != plan.Candidate ||
		trial.Admission != plan.Admission || trial.Subject != plan.Subject ||
		trial.Parent != plan.Parent || trial.Code != plan.Code || trial.Environment != plan.Environment ||
		!slices.Equal(trial.Components, componentIDs) || !slices.Equal(plan.ComponentPlans, componentIDs) ||
		!slices.Equal(trial.Ablations, ablationIDs) || !slices.Equal(plan.Ablations, ablationIDs) {
		return errors.New("model recipe: candidate compilation closure differs")
	}
	planBySpecification := make(map[artifact.ID]CandidateComponentPlan, len(value.Components))
	var declaredAblations []artifact.ID
	for _, component := range value.Components {
		if component.Subject != trial.Subject || planBySpecification[component.Specification].ID.Valid() {
			return errors.New("model recipe: candidate compilation component differs")
		}
		planBySpecification[component.Specification] = component
		declaredAblations = append(declaredAblations, component.Ablations...)
	}
	declaredAblations, err := canonicalCandidateIDs(declaredAblations, artifact.KindProfile)
	if err != nil || !slices.Equal(declaredAblations, ablationIDs) {
		return errors.Join(errors.New("model recipe: candidate compilation ablation union differs"), err)
	}
	for _, ablation := range value.Ablations {
		component, found := planBySpecification[ablation.Specification]
		if !found || ablation.Candidate != trial.Candidate || ablation.Admission != trial.Admission ||
			ablation.Domain != component.Domain || !slices.Contains(component.Ablations, ablation.ID) {
			return errors.New("model recipe: candidate compilation ablation differs")
		}
	}
	return nil
}

var candidateComponentPlanCodec = artifact.JSONDocumentCodec(
	"candidate component plan", artifact.KindProfile,
	CandidateComponentPlanMediaType, CandidateComponentPlanSchema,
	canonicalizeCandidateComponentPlan,
	func(value CandidateComponentPlan) artifact.ID { return value.ID },
	func(value *CandidateComponentPlan, id artifact.ID) { value.ID = id },
	cloneCandidateComponentPlan,
)

var candidateAblationCodec = artifact.JSONDocumentCodec(
	"candidate ablation", artifact.KindProfile,
	CandidateAblationMediaType, CandidateAblationSchema,
	canonicalizeCandidateAblation,
	func(value CandidateAblation) artifact.ID { return value.ID },
	func(value *CandidateAblation, id artifact.ID) { value.ID = id }, nil,
)

var candidateEvaluationPlanCodec = artifact.JSONDocumentCodec(
	"candidate evaluation plan", artifact.KindProfile,
	CandidateEvaluationPlanMediaType, CandidateEvaluationPlanSchema,
	canonicalizeCandidateEvaluationPlan,
	func(value CandidateEvaluationPlan) artifact.ID { return value.ID },
	func(value *CandidateEvaluationPlan, id artifact.ID) { value.ID = id },
	cloneCandidateEvaluationPlan,
)

var candidateTrialCodec = artifact.JSONDocumentCodec(
	"candidate trial", artifact.KindProfile,
	CandidateTrialMediaType, CandidateTrialSchema,
	canonicalizeCandidateTrial,
	func(value CandidateTrial) artifact.ID { return value.ID },
	func(value *CandidateTrial, id artifact.ID) { value.ID = id },
	cloneCandidateTrial,
)

// CompileAdmittedCandidate replays the persisted admission before dispatch,
// compiles every declared component through one matching Go plugin, and derives
// common component, ablation, evaluation, and trial identities without execution.
func CompileAdmittedCandidate(
	ctx context.Context,
	reader artifact.Reader,
	candidateID, admissionID artifact.ID,
	plugins ...CandidateDomainPlugin,
) (CandidateCompilation, error) {
	if ctx == nil || reader == nil || candidateID.Kind() != artifact.KindRecipe ||
		admissionID.Kind() != artifact.KindEvidence {
		return CandidateCompilation{}, errors.New("model recipe: candidate compilation authority is absent")
	}
	candidate, err := RequireCandidate(ctx, reader, candidateID)
	if err != nil {
		return CandidateCompilation{}, err
	}
	byDomain, adapters, err := compilePluginSet(plugins)
	if err != nil {
		return CandidateCompilation{}, err
	}
	admission, err := runrecord.RequireReplayedCandidateAdmission(
		ctx, reader, admissionID, candidate, adapters...,
	)
	if err != nil || admission.Candidate != candidate.ID() {
		return CandidateCompilation{}, errors.Join(
			errors.New("model recipe: candidate compilation admission does not replay"), err,
		)
	}
	facts := candidateCompileFacts(candidate, admission)
	if err := validateCandidateEvaluators(ctx, reader, facts, candidate.Spec().Components, byDomain); err != nil {
		return CandidateCompilation{}, err
	}
	components, ablations, pluginContents, pluginLineage, err := compileCandidateComponents(
		ctx, reader, facts, candidate.Spec().Components, byDomain,
	)
	if err != nil {
		return CandidateCompilation{}, err
	}
	evaluationPlan, err := newCandidateEvaluationPlan(facts, components, ablations)
	if err != nil {
		return CandidateCompilation{}, err
	}
	trial, err := newCandidateTrial(facts, evaluationPlan.ID, components, ablations)
	if err != nil {
		return CandidateCompilation{}, err
	}
	evaluationContent, err := candidateEvaluationPlanCodec.Content(evaluationPlan)
	if err != nil {
		return CandidateCompilation{}, err
	}
	trialContent, err := candidateTrialCodec.Content(trial)
	if err != nil {
		return CandidateCompilation{}, err
	}
	contents := append([]artifact.Content{evaluationContent, trialContent}, pluginContents...)
	lineage := append(evaluationPlan.Lineage(), trial.Lineage()...)
	lineage = append(lineage, pluginLineage...)
	contents, lineage, err = canonicalCompilationFacts(contents, lineage)
	if err != nil {
		return CandidateCompilation{}, err
	}
	compiled := CandidateCompilation{
		Trial: trial, EvaluationPlan: evaluationPlan,
		Components: slices.Clone(components), Ablations: slices.Clone(ablations),
		Contents: cloneCandidateContents(contents), Lineage: slices.Clone(lineage),
	}
	if err := requireStoredCandidateCompilationIfPresent(ctx, reader, compiled); err != nil {
		return CandidateCompilation{}, err
	}
	return compiled, nil
}

// RequireStoredCandidateCompilation replays candidate compilation and then
// requires every derived trial, plan, ablation, and plugin document to exist
// with the exact bytes and lineage computed by the current Go owners.
func RequireStoredCandidateCompilation(
	ctx context.Context,
	reader artifact.Reader,
	candidateID, admissionID artifact.ID,
	plugins ...CandidateDomainPlugin,
) (CandidateCompilation, error) {
	compiled, err := CompileAdmittedCandidate(ctx, reader, candidateID, admissionID, plugins...)
	if err != nil {
		return CandidateCompilation{}, err
	}
	if _, found, err := reader.Artifact(ctx, compiled.Trial.ID); err != nil {
		return CandidateCompilation{}, err
	} else if !found {
		return CandidateCompilation{}, errors.New("model recipe: stored candidate compilation is absent")
	}
	return compiled, nil
}

func requireStoredCandidateCompilationIfPresent(
	ctx context.Context,
	reader artifact.Reader,
	compiled CandidateCompilation,
) error {
	if _, found, err := reader.Artifact(ctx, compiled.Trial.ID); err != nil || !found {
		return err
	}
	trial, err := candidateTrialCodec.RequireExactLineage(
		ctx, reader, compiled.Trial.ID, CandidateTrial.Lineage,
	)
	if err != nil {
		return err
	}
	if err := validateStoredCandidateTrial(ctx, reader, trial); err != nil {
		return err
	}
	return requireStoredCandidateCompilation(ctx, reader, compiled)
}

func requireStoredCandidateCompilation(
	ctx context.Context,
	reader artifact.Reader,
	compiled CandidateCompilation,
) error {
	expectedLineage := make(map[artifact.ID][]artifact.Lineage, len(compiled.Contents))
	for _, edge := range compiled.Lineage {
		expectedLineage[edge.Child] = append(expectedLineage[edge.Child], edge)
	}
	for _, expected := range compiled.Contents {
		stored, found, err := artifact.ReadContent(ctx, reader, expected.Descriptor.ID)
		if err != nil || !found || stored.Descriptor != expected.Descriptor ||
			!bytes.Equal(stored.Data, expected.Data) {
			return errors.Join(
				fmt.Errorf("model recipe: stored candidate document differs: %s", expected.Descriptor.ID), err,
			)
		}
		parents, err := reader.Parents(ctx, expected.Descriptor.ID)
		if err != nil {
			return err
		}
		slices.SortFunc(parents, compareCandidateLineage)
		want := slices.Clone(expectedLineage[expected.Descriptor.ID])
		slices.SortFunc(want, compareCandidateLineage)
		if !slices.Equal(parents, want) {
			return fmt.Errorf("model recipe: stored candidate document lineage differs: %s", expected.Descriptor.ID)
		}
	}
	return nil
}

// Lineage binds the common plan to every existing and newly derived authority.
func (value CandidateComponentPlan) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Specification, value.Subject, value.Realization, value.ResourcePolicy}
	parents = append(parents, value.Inputs...)
	parents = append(parents, value.Documents...)
	parents = append(parents, value.Ablations...)
	return artifact.DependencyLineage(value.ID, uniqueCandidateIDs(parents)...)
}

// Lineage binds an ablation to the admitted candidate, omitted delta, and alternate realization.
func (value CandidateAblation) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, uniqueCandidateIDs([]artifact.ID{
		value.Candidate, value.Admission, value.Specification, value.Omitted, value.Realization,
	})...)
}

// Lineage binds evaluation to the exact causal, budget, stop, and plan authorities.
func (value CandidateEvaluationPlan) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Candidate, value.Admission, value.Subject, value.Parent, value.Falsifier,
		value.DevelopmentSplit, value.PromotionSplit, value.DevelopmentBudget,
		value.PromotionBudget, value.Code, value.Environment,
	}
	parents = append(parents, value.ComponentPlans...)
	parents = append(parents, value.Ablations...)
	for _, reference := range value.CausalReferences {
		parents = append(parents, reference.Subject, reference.Evidence)
	}
	return artifact.DependencyLineage(value.ID, uniqueCandidateIDs(parents)...)
}

// Lineage binds the trial to its candidate, plans, ablations, and evaluation intent.
func (value CandidateTrial) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Candidate, value.Admission, value.Subject, value.Parent,
		value.EvaluationPlan, value.Code, value.Environment,
	}
	parents = append(parents, value.Components...)
	parents = append(parents, value.Ablations...)
	return artifact.DependencyLineage(value.ID, uniqueCandidateIDs(parents)...)
}

func compilePluginSet(
	plugins []CandidateDomainPlugin,
) (map[CandidateDomain]CandidateDomainPlugin, []runrecord.CandidateComponentAdmissionAdapter, error) {
	byDomain := make(map[CandidateDomain]CandidateDomainPlugin, len(plugins))
	adapters := CandidateAdmissionAdapters()
	admissionDomains := make(map[string]bool, len(adapters)+len(plugins))
	for _, adapter := range adapters {
		admissionDomains[adapter.CandidateDomain()] = true
	}
	for _, plugin := range plugins {
		if plugin == nil || strings.TrimSpace(plugin.CandidateDomain()) != plugin.CandidateDomain() ||
			plugin.CandidateDomain() == "" {
			return nil, nil, errors.New("model recipe: invalid candidate domain plugin")
		}
		domain := CandidateDomain(plugin.CandidateDomain())
		if _, duplicate := byDomain[domain]; duplicate {
			return nil, nil, fmt.Errorf("model recipe: duplicate candidate domain plugin %q", domain)
		}
		adapter := plugin.AdmissionAdapter()
		evaluator := plugin.EvaluatorPlugin()
		if adapter == nil || adapter.CandidateDomain() != plugin.CandidateDomain() || evaluator == nil ||
			strings.TrimSpace(evaluator.CandidateEvaluator()) != evaluator.CandidateEvaluator() ||
			evaluator.CandidateEvaluator() == "" {
			return nil, nil, fmt.Errorf("model recipe: candidate domain %q admission adapter differs", domain)
		}
		byDomain[domain] = plugin
		if !admissionDomains[adapter.CandidateDomain()] {
			adapters = append(adapters, adapter)
			admissionDomains[adapter.CandidateDomain()] = true
		}
	}
	return byDomain, adapters, nil
}

func validateCandidateEvaluators(
	ctx context.Context,
	reader artifact.Reader,
	facts CandidateCompileFacts,
	declared []CandidateComponent,
	plugins map[CandidateDomain]CandidateDomainPlugin,
) error {
	validated := make(map[string]bool, len(declared))
	for _, component := range declared {
		plugin, present := plugins[component.Domain]
		if !present {
			return fmt.Errorf("model recipe: candidate domain %q has no compiled Go evaluator plugin", component.Domain)
		}
		evaluator := plugin.EvaluatorPlugin()
		if validated[evaluator.CandidateEvaluator()] {
			continue
		}
		if err := evaluator.ValidateCandidateEvaluation(ctx, reader, cloneCandidateCompileFacts(facts)); err != nil {
			return fmt.Errorf("model recipe: validate candidate evaluator %q: %w", evaluator.CandidateEvaluator(), err)
		}
		validated[evaluator.CandidateEvaluator()] = true
	}
	return nil
}

func compileCandidateComponents(
	ctx context.Context,
	reader artifact.Reader,
	facts CandidateCompileFacts,
	declared []CandidateComponent,
	plugins map[CandidateDomain]CandidateDomainPlugin,
) ([]CandidateComponentPlan, []CandidateAblation, []artifact.Content, []artifact.Lineage, error) {
	if !checked.Nonzero(len(declared)) {
		return nil, nil, nil, nil, errors.New("model recipe: candidate has no declared components")
	}
	components := make([]CandidateComponentPlan, tensor.FirstOffset, len(declared))
	var ablations []CandidateAblation
	var contents []artifact.Content
	var lineage []artifact.Lineage
	for _, component := range declared {
		plugin, present := plugins[component.Domain]
		if !present {
			return nil, nil, nil, nil, fmt.Errorf(
				"model recipe: candidate domain %q has no compiled Go realization plugin", component.Domain,
			)
		}
		result, err := plugin.CompileCandidateComponent(ctx, reader, cloneCandidateCompileFacts(facts), component)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("model recipe: compile candidate %s component: %w", component.Domain, err)
		}
		plan, commonAblations, componentContents, componentLineage, err := prepareCandidateComponent(
			ctx, reader, facts, component, result,
		)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		components = append(components, plan)
		ablations = append(ablations, commonAblations...)
		contents = append(contents, componentContents...)
		lineage = append(lineage, componentLineage...)
	}
	return components, ablations, contents, lineage, nil
}

func prepareCandidateComponent(
	ctx context.Context,
	reader artifact.Reader,
	facts CandidateCompileFacts,
	component CandidateComponent,
	result CandidateComponentCompilation,
) (CandidateComponentPlan, []CandidateAblation, []artifact.Content, []artifact.Lineage, error) {
	result, err := validateCandidateComponentCompilation(ctx, reader, facts, component, result)
	if err != nil {
		return CandidateComponentPlan{}, nil, nil, nil, err
	}
	ablations := make([]CandidateAblation, len(result.Ablations))
	contents := cloneCandidateContents(result.Contents)
	lineage := slices.Clone(result.Lineage)
	for index, declared := range result.Ablations {
		ablations[index], err = candidateAblationCodec.New(CandidateAblation{
			Version: artifact.InitialDocumentVersion, Candidate: facts.Candidate, Admission: facts.Admission,
			Domain: component.Domain, Specification: component.Specification,
			Omitted: declared.Omitted, Realization: declared.Realization,
		})
		if err != nil {
			return CandidateComponentPlan{}, nil, nil, nil, err
		}
		content, contentErr := candidateAblationCodec.Content(ablations[index])
		if contentErr != nil {
			return CandidateComponentPlan{}, nil, nil, nil, contentErr
		}
		contents = append(contents, content)
		lineage = append(lineage, ablations[index].Lineage()...)
		result.Plan.Ablations = append(result.Plan.Ablations, ablations[index].ID)
	}
	result.Plan.Version = artifact.InitialDocumentVersion
	result.Plan.ID = artifact.ID{}
	plan, err := candidateComponentPlanCodec.New(result.Plan)
	if err != nil {
		return CandidateComponentPlan{}, nil, nil, nil, err
	}
	planContent, err := candidateComponentPlanCodec.Content(plan)
	if err != nil {
		return CandidateComponentPlan{}, nil, nil, nil, err
	}
	contents = append(contents, planContent)
	lineage = append(lineage, plan.Lineage()...)
	return plan, ablations, contents, lineage, nil
}

func validateCandidateComponentCompilation(
	ctx context.Context,
	reader artifact.Reader,
	facts CandidateCompileFacts,
	component CandidateComponent,
	result CandidateComponentCompilation,
) (CandidateComponentCompilation, error) {
	plan := &result.Plan
	plan.Version = artifact.InitialDocumentVersion
	plan.ID = artifact.ID{}
	plan.Ablations = nil
	if err := canonicalizeCandidateComponentPlanBase(plan); err != nil ||
		plan.Domain != component.Domain || plan.Specification != component.Specification ||
		plan.Subject != facts.Subject || !slices.Contains(plan.Inputs, facts.Parent) {
		return CandidateComponentCompilation{}, errors.Join(
			errors.New("model recipe: candidate plugin result differs from common authority"), err,
		)
	}
	for _, id := range append(slices.Clone(plan.Inputs), plan.Subject) {
		if _, found, err := reader.Artifact(ctx, id); err != nil || !found {
			return CandidateComponentCompilation{}, errors.Join(
				fmt.Errorf("model recipe: candidate plugin input authority is absent: %s", id), err,
			)
		}
	}
	documents := make(map[artifact.ID]bool, len(plan.Documents))
	for _, id := range plan.Documents {
		documents[id] = true
	}
	seen := make(map[artifact.ID]bool, len(result.Contents))
	result.Contents = cloneCandidateContents(result.Contents)
	for _, content := range result.Contents {
		id := content.Descriptor.ID
		if err := content.Validate(); err != nil || !documents[id] || seen[id] ||
			content.Descriptor.MediaType == "" || content.Descriptor.Schema == "" {
			return CandidateComponentCompilation{}, errors.Join(
				errors.New("model recipe: candidate plugin emitted invalid, undeclared, or duplicate content"), err,
			)
		}
		seen[id] = true
	}
	for id := range documents {
		if !seen[id] {
			return CandidateComponentCompilation{}, fmt.Errorf("model recipe: candidate plugin omitted document %s", id)
		}
	}
	allowedParents := make(map[artifact.ID]bool, len(plan.Inputs)+len(plan.Documents))
	for _, id := range plan.Inputs {
		allowedParents[id] = true
	}
	for _, id := range plan.Documents {
		allowedParents[id] = true
	}
	lineaged := make(map[artifact.ID]bool, len(plan.Documents))
	for _, edge := range result.Lineage {
		if err := edge.Validate(); err != nil || !documents[edge.Child] || !allowedParents[edge.Parent] {
			return CandidateComponentCompilation{}, errors.Join(
				errors.New("model recipe: candidate plugin emitted lineage outside its document closure"), err,
			)
		}
		lineaged[edge.Child] = true
	}
	if !lineaged[plan.Realization] {
		return CandidateComponentCompilation{}, errors.New("model recipe: candidate realization lacks exact derivation lineage")
	}
	if !checked.Nonzero(len(result.Ablations)) {
		return CandidateComponentCompilation{}, errors.New("model recipe: candidate component has no declared delta ablation")
	}
	seenOmitted := make(map[artifact.ID]bool, len(result.Ablations))
	seenRealization := map[artifact.ID]bool{plan.Realization: true, plan.ResourcePolicy: true}
	for _, ablation := range result.Ablations {
		if !ablation.Omitted.Valid() || !slices.Contains(plan.Inputs, ablation.Omitted) ||
			ablation.Realization.Kind() != artifact.KindProfile || !documents[ablation.Realization] ||
			seenOmitted[ablation.Omitted] || seenRealization[ablation.Realization] || !lineaged[ablation.Realization] {
			return CandidateComponentCompilation{}, errors.New("model recipe: invalid or duplicate candidate domain ablation")
		}
		seenOmitted[ablation.Omitted] = true
		seenRealization[ablation.Realization] = true
	}
	slices.SortFunc(result.Ablations, func(left, right CandidateDomainAblation) int {
		if order := artifact.CompareID(left.Omitted, right.Omitted); checked.Nonzero(order) {
			return order
		}
		return artifact.CompareID(left.Realization, right.Realization)
	})
	canonicalContents, canonicalLineage, err := canonicalCompilationFacts(result.Contents, result.Lineage)
	if err != nil {
		return CandidateComponentCompilation{}, err
	}
	result.Contents, result.Lineage = canonicalContents, canonicalLineage
	return result, nil
}

func candidateCompileFacts(candidate Candidate, admission runrecord.CandidateAdmission) CandidateCompileFacts {
	spec := candidate.Spec()
	return CandidateCompileFacts{
		Candidate: candidate.ID(), Admission: admission.ID, Subject: spec.Subject, Parent: spec.Parent,
		Prediction: spec.Prediction, CostUnit: spec.CostUnit, Falsifier: spec.Falsifier,
		References: slices.Clone(spec.References), DevelopmentSplit: spec.DevelopmentSplit,
		PromotionSplit: spec.PromotionSplit, DevelopmentBudget: spec.DevelopmentBudget,
		PromotionBudget: spec.PromotionBudget, Code: spec.Code, Environment: spec.Environment,
	}
}

func newCandidateEvaluationPlan(
	facts CandidateCompileFacts,
	components []CandidateComponentPlan,
	ablations []CandidateAblation,
) (CandidateEvaluationPlan, error) {
	componentIDs := make([]artifact.ID, len(components))
	for index, component := range components {
		componentIDs[index] = component.ID
	}
	ablationIDs := make([]artifact.ID, len(ablations))
	for index, ablation := range ablations {
		ablationIDs[index] = ablation.ID
	}
	return candidateEvaluationPlanCodec.New(CandidateEvaluationPlan{
		Version:   artifact.InitialDocumentVersion,
		Candidate: facts.Candidate, Admission: facts.Admission, Subject: facts.Subject, Parent: facts.Parent,
		Prediction: facts.Prediction, CostUnit: facts.CostUnit, Falsifier: facts.Falsifier,
		DevelopmentSplit: facts.DevelopmentSplit, PromotionSplit: facts.PromotionSplit,
		DevelopmentBudget: facts.DevelopmentBudget, PromotionBudget: facts.PromotionBudget,
		ComponentPlans: componentIDs, Ablations: ablationIDs,
		CausalReferences: slices.Clone(facts.References), Code: facts.Code, Environment: facts.Environment,
	})
}

func newCandidateTrial(
	facts CandidateCompileFacts,
	evaluationPlan artifact.ID,
	components []CandidateComponentPlan,
	ablations []CandidateAblation,
) (CandidateTrial, error) {
	componentIDs := make([]artifact.ID, len(components))
	for index, component := range components {
		componentIDs[index] = component.ID
	}
	ablationIDs := make([]artifact.ID, len(ablations))
	for index, ablation := range ablations {
		ablationIDs[index] = ablation.ID
	}
	return candidateTrialCodec.New(CandidateTrial{
		Version:   artifact.InitialDocumentVersion,
		Candidate: facts.Candidate, Admission: facts.Admission, Subject: facts.Subject, Parent: facts.Parent,
		EvaluationPlan: evaluationPlan, Components: componentIDs, Ablations: ablationIDs,
		Code: facts.Code, Environment: facts.Environment,
	})
}

func validateStoredCandidateTrial(ctx context.Context, reader artifact.Reader, trial CandidateTrial) error {
	candidate, err := RequireCandidate(ctx, reader, trial.Candidate)
	if err != nil {
		return err
	}
	admission, err := runrecord.RequireCandidateAdmission(ctx, reader, trial.Admission)
	if err != nil || admission.Candidate != candidate.ID() {
		return errors.Join(errors.New("model recipe: stored trial admission differs"), err)
	}
	evaluation, err := candidateEvaluationPlanCodec.RequireExactLineage(
		ctx, reader, trial.EvaluationPlan, CandidateEvaluationPlan.Lineage,
	)
	if err != nil {
		return err
	}
	spec := candidate.Spec()
	if evaluation.Candidate != trial.Candidate || evaluation.Admission != trial.Admission ||
		evaluation.Subject != trial.Subject || evaluation.Parent != trial.Parent ||
		trial.Subject != spec.Subject || trial.Parent != spec.Parent || trial.Code != spec.Code ||
		trial.Environment != spec.Environment || evaluation.Prediction != spec.Prediction ||
		evaluation.CostUnit != spec.CostUnit || evaluation.Falsifier != spec.Falsifier ||
		evaluation.DevelopmentSplit != spec.DevelopmentSplit || evaluation.PromotionSplit != spec.PromotionSplit ||
		evaluation.DevelopmentBudget != spec.DevelopmentBudget || evaluation.PromotionBudget != spec.PromotionBudget ||
		evaluation.Code != spec.Code || evaluation.Environment != spec.Environment ||
		!slices.Equal(evaluation.CausalReferences, spec.References) ||
		!slices.Equal(evaluation.ComponentPlans, trial.Components) ||
		!slices.Equal(evaluation.Ablations, trial.Ablations) {
		return errors.New("model recipe: stored trial and evaluation closure differ")
	}
	plans := make([]CandidateComponentPlan, len(trial.Components))
	declared := make([]CandidateComponent, len(trial.Components))
	var planAblations []artifact.ID
	for index, id := range trial.Components {
		plans[index], err = candidateComponentPlanCodec.RequireExactLineage(
			ctx, reader, id, CandidateComponentPlan.Lineage,
		)
		if err != nil {
			return err
		}
		if plans[index].Subject != spec.Subject || !slices.Contains(plans[index].Inputs, spec.Parent) {
			return errors.New("model recipe: stored component differs from candidate subject or parent")
		}
		declared[index] = CandidateComponent{Domain: plans[index].Domain, Specification: plans[index].Specification}
		planAblations = append(planAblations, plans[index].Ablations...)
	}
	slices.SortFunc(declared, compareCandidateComponent)
	if !slices.Equal(declared, spec.Components) {
		return errors.New("model recipe: stored trial component set differs from candidate")
	}
	planAblations, err = canonicalCandidateIDs(planAblations, artifact.KindProfile)
	if err != nil || !slices.Equal(planAblations, trial.Ablations) {
		return errors.Join(errors.New("model recipe: stored trial ablation union differs"), err)
	}
	planBySpec := make(map[artifact.ID]CandidateComponentPlan, len(plans))
	for _, plan := range plans {
		planBySpec[plan.Specification] = plan
	}
	seenOmitted := make(map[artifact.ID]map[artifact.ID]bool, len(plans))
	seenRealization := make(map[artifact.ID]map[artifact.ID]bool, len(plans))
	for _, id := range trial.Ablations {
		ablation, loadErr := candidateAblationCodec.RequireExactLineage(
			ctx, reader, id, CandidateAblation.Lineage,
		)
		if loadErr != nil {
			return loadErr
		}
		plan, present := planBySpec[ablation.Specification]
		if !present {
			return errors.New("model recipe: stored trial ablation has no component authority")
		}
		if seenOmitted[plan.Specification] == nil {
			seenOmitted[plan.Specification] = make(map[artifact.ID]bool)
			seenRealization[plan.Specification] = map[artifact.ID]bool{
				plan.Realization: true, plan.ResourcePolicy: true,
			}
		}
		if ablation.Candidate != trial.Candidate || ablation.Admission != trial.Admission ||
			ablation.Domain != plan.Domain || !slices.Contains(plan.Inputs, ablation.Omitted) ||
			!slices.Contains(plan.Documents, ablation.Realization) ||
			seenOmitted[plan.Specification][ablation.Omitted] ||
			seenRealization[plan.Specification][ablation.Realization] {
			return errors.New("model recipe: stored trial ablation differs from component authority")
		}
		seenOmitted[plan.Specification][ablation.Omitted] = true
		seenRealization[plan.Specification][ablation.Realization] = true
	}
	return nil
}

func canonicalizeCandidateComponentPlan(value *CandidateComponentPlan) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !checked.Nonzero(len(value.Ablations)) {
		return errors.New("model recipe: invalid candidate component plan")
	}
	if err := canonicalizeCandidateComponentPlanBase(value); err != nil {
		return err
	}
	var err error
	value.Ablations, err = canonicalCandidateIDs(value.Ablations, artifact.KindProfile)
	return err
}

func canonicalizeCandidateComponentPlanBase(value *CandidateComponentPlan) error {
	if value == nil || !candidateSpecificationKind(value.Domain, value.Specification.Kind()) ||
		!value.Subject.Valid() ||
		value.Realization.Kind() != artifact.KindProfile || value.ResourcePolicy.Kind() != artifact.KindProfile ||
		value.Realization == value.ResourcePolicy || !checked.Nonzero(value.PeakResidentBytes) ||
		!checked.Nonzero(value.ArtifactBytes) || !checked.Nonzero(len(value.Inputs)) ||
		len(value.Documents) < tensor.PairedExtent {
		return errors.New("model recipe: invalid candidate component plan")
	}
	var err error
	for values, kind := range map[*[]artifact.ID]artifact.Kind{
		&value.Inputs: artifact.KindInvalid, &value.Documents: artifact.KindInvalid,
	} {
		*values, err = canonicalCandidateIDs(*values, kind)
		if err != nil {
			return err
		}
	}
	if !slices.Contains(value.Documents, value.Realization) || !slices.Contains(value.Documents, value.ResourcePolicy) {
		return errors.New("model recipe: component roles are outside produced documents")
	}
	for _, document := range value.Documents {
		if document == value.Specification || document == value.Subject || slices.Contains(value.Inputs, document) {
			return errors.New("model recipe: produced document aliases an existing authority")
		}
	}
	return nil
}

func canonicalizeCandidateAblation(value *CandidateAblation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Candidate.Kind() != artifact.KindRecipe || value.Admission.Kind() != artifact.KindEvidence ||
		!candidateSpecificationKind(value.Domain, value.Specification.Kind()) || !value.Omitted.Valid() ||
		value.Realization.Kind() != artifact.KindProfile || value.Omitted == value.Specification {
		return errors.New("model recipe: invalid candidate ablation")
	}
	return nil
}

func canonicalizeCandidateEvaluationPlan(value *CandidateEvaluationPlan) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Candidate.Kind() != artifact.KindRecipe || value.Admission.Kind() != artifact.KindEvidence ||
		!value.Subject.Valid() || !value.Parent.Valid() || value.Subject == value.Parent ||
		value.Falsifier.Kind() != artifact.KindRecipe || value.DevelopmentSplit.Kind() != artifact.KindDatasetShard ||
		value.PromotionSplit.Kind() != artifact.KindDatasetShard || value.DevelopmentSplit == value.PromotionSplit ||
		value.DevelopmentBudget.Kind() != artifact.KindEvidence || value.PromotionBudget.Kind() != artifact.KindEvidence ||
		value.Code.Kind() != artifact.KindEvidence || value.Environment.Kind() != artifact.KindEvidence ||
		strings.TrimSpace(value.CostUnit) == "" || !checked.Nonzero(len(value.ComponentPlans)) ||
		!checked.Nonzero(len(value.Ablations)) || !checked.Nonzero(len(value.CausalReferences)) {
		return errors.New("model recipe: invalid candidate evaluation plan")
	}
	if err := value.Prediction.Validate(); err != nil {
		return err
	}
	var err error
	value.ComponentPlans, err = canonicalCandidateIDs(value.ComponentPlans, artifact.KindProfile)
	if err != nil {
		return err
	}
	value.Ablations, err = canonicalCandidateIDs(value.Ablations, artifact.KindProfile)
	if err != nil {
		return err
	}
	value.CausalReferences = slices.Clone(value.CausalReferences)
	slices.SortFunc(value.CausalReferences, compareCandidateReference)
	for index, reference := range value.CausalReferences {
		if !reference.Role.valid() || !reference.Subject.Valid() || reference.Evidence.Kind() != artifact.KindEvidence ||
			index > tensor.FirstOffset && reference == value.CausalReferences[index-tensor.SingletonExtent] {
			return errors.New("model recipe: invalid candidate causal reference")
		}
	}
	return nil
}

func canonicalizeCandidateTrial(value *CandidateTrial) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Candidate.Kind() != artifact.KindRecipe || value.Admission.Kind() != artifact.KindEvidence ||
		!value.Subject.Valid() || !value.Parent.Valid() || value.Subject == value.Parent ||
		value.EvaluationPlan.Kind() != artifact.KindProfile || value.Code.Kind() != artifact.KindEvidence ||
		value.Environment.Kind() != artifact.KindEvidence || !checked.Nonzero(len(value.Components)) ||
		!checked.Nonzero(len(value.Ablations)) {
		return errors.New("model recipe: invalid candidate trial")
	}
	var err error
	value.Components, err = canonicalCandidateIDs(value.Components, artifact.KindProfile)
	if err != nil {
		return err
	}
	value.Ablations, err = canonicalCandidateIDs(value.Ablations, artifact.KindProfile)
	return err
}

func canonicalCompilationFacts(
	contents []artifact.Content,
	lineage []artifact.Lineage,
) ([]artifact.Content, []artifact.Lineage, error) {
	contents = cloneCandidateContents(contents)
	slices.SortFunc(contents, func(left, right artifact.Content) int {
		return artifact.CompareID(left.Descriptor.ID, right.Descriptor.ID)
	})
	compactedContents := contents[:tensor.FirstOffset]
	for _, content := range contents {
		if err := content.Validate(); err != nil {
			return nil, nil, err
		}
		if len(compactedContents) > tensor.FirstOffset &&
			compactedContents[len(compactedContents)-tensor.SingletonExtent].Descriptor.ID == content.Descriptor.ID {
			previous := compactedContents[len(compactedContents)-tensor.SingletonExtent]
			if previous.Descriptor != content.Descriptor || !bytes.Equal(previous.Data, content.Data) {
				return nil, nil, errors.New("model recipe: candidate compilation contains conflicting content")
			}
			continue
		}
		compactedContents = append(compactedContents, content)
	}
	lineage = slices.Clone(lineage)
	slices.SortFunc(lineage, compareCandidateLineage)
	lineage = slices.Compact(lineage)
	children := make(map[artifact.ID]bool, len(compactedContents))
	for _, content := range compactedContents {
		children[content.Descriptor.ID] = true
	}
	for _, edge := range lineage {
		if err := edge.Validate(); err != nil || !children[edge.Child] {
			return nil, nil, errors.Join(
				errors.New("model recipe: candidate compilation contains invalid lineage"), err,
			)
		}
	}
	return compactedContents, lineage, nil
}

func compareCandidateComponent(left, right CandidateComponent) int {
	if order := strings.Compare(string(left.Domain), string(right.Domain)); checked.Nonzero(order) {
		return order
	}
	return artifact.CompareID(left.Specification, right.Specification)
}

func compareCandidateReference(left, right CandidateReference) int {
	if order := strings.Compare(string(left.Role), string(right.Role)); checked.Nonzero(order) {
		return order
	}
	if order := artifact.CompareID(left.Subject, right.Subject); checked.Nonzero(order) {
		return order
	}
	return artifact.CompareID(left.Evidence, right.Evidence)
}

func compareCandidateLineage(left, right artifact.Lineage) int {
	if order := artifact.CompareID(left.Child, right.Child); checked.Nonzero(order) {
		return order
	}
	if order := artifact.CompareID(left.Parent, right.Parent); checked.Nonzero(order) {
		return order
	}
	return int(left.Relation) - int(right.Relation)
}

func canonicalCandidateIDs(values []artifact.ID, kind artifact.Kind) ([]artifact.ID, error) {
	result := slices.Clone(values)
	slices.SortFunc(result, artifact.CompareID)
	for index, id := range result {
		if !id.Valid() || kind != artifact.KindInvalid && id.Kind() != kind ||
			index > tensor.FirstOffset && result[index-tensor.SingletonExtent] == id {
			return nil, errors.New("model recipe: invalid or duplicate candidate plan authority")
		}
	}
	return result, nil
}

func cloneCandidateCompileFacts(value CandidateCompileFacts) CandidateCompileFacts {
	value.References = slices.Clone(value.References)
	return value
}

func cloneCandidateComponentPlan(value CandidateComponentPlan) CandidateComponentPlan {
	value.Inputs = slices.Clone(value.Inputs)
	value.Documents = slices.Clone(value.Documents)
	value.Ablations = slices.Clone(value.Ablations)
	return value
}

func cloneCandidateEvaluationPlan(value CandidateEvaluationPlan) CandidateEvaluationPlan {
	value.ComponentPlans = slices.Clone(value.ComponentPlans)
	value.Ablations = slices.Clone(value.Ablations)
	value.CausalReferences = slices.Clone(value.CausalReferences)
	return value
}

func cloneCandidateTrial(value CandidateTrial) CandidateTrial {
	value.Components = slices.Clone(value.Components)
	value.Ablations = slices.Clone(value.Ablations)
	return value
}

func cloneCandidateContents(values []artifact.Content) []artifact.Content {
	result := make([]artifact.Content, len(values))
	for index, value := range values {
		result[index] = value.Clone()
	}
	return result
}

func uniqueCandidateIDs(values []artifact.ID) []artifact.ID {
	result := slices.Clone(values)
	slices.SortFunc(result, artifact.CompareID)
	return slices.Compact(result)
}
