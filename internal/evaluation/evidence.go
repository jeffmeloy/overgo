package evaluation

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

const (
	evaluationEvidenceMedia  = "application/vnd.overgo.evaluation-evidence+json"
	evaluationEvidenceSchema = "overgo/evaluation-evidence/v1"
)

const (
	EvaluationEvidenceMediaType = evaluationEvidenceMedia
	EvaluationEvidenceSchema    = evaluationEvidenceSchema
)

type EvaluationEvidence struct {
	ID              artifact.ID             `json:"-"`
	Version         uint16                  `json:"version"`
	Plan            artifact.ID             `json:"plan"`
	Acceptance      artifact.ID             `json:"acceptance"`
	Evaluator       artifact.ID             `json:"evaluator"`
	Report          artifact.ID             `json:"report"`
	Run             artifact.ID             `json:"run"`
	Evaluation      artifact.ID             `json:"evaluation"`
	ModelDefinition artifact.ID             `json:"model_definition"`
	Recipe          artifact.ID             `json:"recipe"`
	Dataset         artifact.ID             `json:"dataset"`
	Split           artifact.ID             `json:"split"`
	Shards          []artifact.ID           `json:"shards,omitempty"`
	Environment     artifact.ID             `json:"environment"`
	CodeCommit      string                  `json:"code_commit"`
	Phases          []runrecord.PhaseMetric `json:"phases"`
	Metrics         []runrecord.Metric      `json:"metrics"`
	// ResourceObservation names the exact raw-chunk summary published in the
	// same semantic batch as an isolated run. Resources preserves measured
	// zero versus missing dimensions without forcing high-volume samples into
	// this evidence document.
	ResourceObservation artifact.ID                `json:"resource_observation,omitzero"`
	Resources           *runrecord.ResourceFitness `json:"resources,omitempty"`
	// Causal explains why the evaluation ran.
	Causal *runrecord.CausalContext `json:"causal,omitempty"`
}

var evaluationEvidenceCodec = artifact.JSONDocumentCodec(
	"evaluation evidence", artifact.KindEvidence, evaluationEvidenceMedia, evaluationEvidenceSchema,
	canonicalizeEvaluationEvidence, func(value EvaluationEvidence) artifact.ID { return value.ID },
	func(value *EvaluationEvidence, id artifact.ID) { value.ID = id }, cloneEvaluationEvidence,
)

// ParseEvaluationEvidence decodes canonical evaluation evidence.
func ParseEvaluationEvidence(content []byte) (EvaluationEvidence, error) {
	return evaluationEvidenceCodec.Parse(content)
}

// RequireEvaluationEvidence loads one canonical evidence document and proves
// that every typed plan, run, metric, report, and shard authority it cites is
// still the exact immutable authority recorded by the document.
func RequireEvaluationEvidence(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (EvaluationEvidence, error) {
	return evaluationEvidenceCodec.RequireVerified(ctx, reader, id, ValidateEvaluationEvidence)
}

// Content returns the native OvergoDB evidence document.
func (value EvaluationEvidence) Content() (artifact.Content, error) {
	return evaluationEvidenceCodec.Content(value)
}

// Lineage returns every immutable authority required to admit the evidence.
func (value EvaluationEvidence) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Plan, value.Acceptance, value.Evaluator, value.Report, value.Run, value.Evaluation,
		value.ModelDefinition, value.Recipe, value.Dataset, value.Split, value.Environment,
	}
	parents = append(parents, value.Shards...)
	if value.ResourceObservation.Valid() {
		parents = append(parents, value.ResourceObservation)
		parents = append(parents, value.Resources.Authorities()...)
	}
	return artifact.DependencyLineage(value.ID, uniqueArtifactIDs(parents)...)
}

func ValidateEvaluationEvidence(ctx context.Context, reader artifact.Reader, value EvaluationEvidence) error {
	if ctx == nil || reader == nil || evaluationEvidenceCodec.ValidateIdentity(value) != nil {
		return errors.New("evaluation: invalid stored evidence")
	}
	plan, err := loadEvidencePlan(ctx, reader, value.Plan)
	if err != nil || plan.body.ModelDefinition != value.ModelDefinition || plan.body.RuntimeRecipe != value.Recipe ||
		plan.body.Dataset != value.Dataset || plan.body.Split != value.Split || plan.body.Environment != value.Environment ||
		plan.body.CodeCommit != value.CodeCommit {
		return errors.Join(err, errors.New("evaluation: stored plan differs from evidence"))
	}
	policy, found, err := acceptancePolicyCodec.Read(ctx, reader, value.Acceptance)
	if err != nil || !found {
		return errors.Join(err, errors.New("evaluation: stored acceptance policy is absent"))
	}
	evaluator, found, err := evaluatorCodec.Read(ctx, reader, value.Evaluator)
	if err != nil || !found || evaluator.Plan != value.Plan || evaluator.Acceptance != value.Acceptance ||
		!slices.Equal(evaluator.Metrics, policy.Metrics) {
		return errors.Join(err, errors.New("evaluation: stored evaluator differs"))
	}
	run, err := runrecord.RequireRun(ctx, reader, value.Run)
	if err != nil {
		return err
	}
	record, err := runrecord.RequireEvaluation(ctx, reader, value.Evaluation)
	if err != nil {
		return err
	}
	if run.Outcome != runrecord.OutcomeSucceeded || run.Recipe != value.Recipe || run.Environment != value.Environment ||
		run.CodeCommit != value.CodeCommit || !slices.Contains(run.Inputs, value.Plan) || !slices.Contains(run.Outputs, value.Report) ||
		record.Recipe != value.Recipe || record.Run != value.Run || record.Dataset != value.Dataset || !policy.admits(record.Metrics) ||
		!slices.Equal(run.Phases, value.Phases) || !slices.Equal(record.Metrics, value.Metrics) {
		return errors.New("evaluation: stored run or metrics differ from evidence")
	}
	if value.ResourceObservation.Valid() {
		observation, err := runrecord.RequireObservationChunkSummary(ctx, reader, value.ResourceObservation)
		if err != nil || !reflect.DeepEqual(observation.Aggregate, *value.Resources) {
			return errors.Join(err, errors.New("evaluation: stored resource observation differs from evidence"))
		}
		if err := validateIsolatedResources(ctx, reader, plan, run, *value.Resources); err != nil {
			return err
		}
	}
	reportContent, found, err := artifact.ReadContent(ctx, reader, value.Report)
	if err != nil || !found {
		return errors.Join(err, errors.New("evaluation: stored report is absent"))
	}
	if err := reportContent.Validate(); err != nil {
		return err
	}
	shards, err := evaluationReportShards(ctx, reader, value.Report, value.Split)
	if err != nil || !slices.Equal(shards, value.Shards) {
		return errors.Join(err, errors.New("evaluation: stored shard lineage differs from evidence"))
	}
	return nil
}

func PublishEvaluationEvidence(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	acceptance AcceptancePolicy,
	evaluator Evaluator,
	report artifact.ID,
	run runrecord.Run,
	record runrecord.Evaluation,
	causal ...runrecord.CausalContext,
) (EvaluationEvidence, error) {
	evidence, err := newEvaluationEvidence(
		ctx, repository, plan, acceptance, evaluator, report, run, record, nil, causal...,
	)
	if err != nil {
		return EvaluationEvidence{}, err
	}
	batch := artifact.Batch{Key: "evaluation/evidence/" + evidence.ID.String()}
	if err := appendEvaluationEvidence(&batch, acceptance, evaluator, evidence); err != nil {
		return EvaluationEvidence{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return evidence, err
}

func newEvaluationEvidence(
	ctx context.Context,
	repository artifact.Reader,
	plan Plan,
	acceptance AcceptancePolicy,
	evaluator Evaluator,
	report artifact.ID,
	run runrecord.Run,
	record runrecord.Evaluation,
	observation *runrecord.ObservationChunkSummary,
	causal ...runrecord.CausalContext,
) (EvaluationEvidence, error) {
	if len(causal) > 1 || ctx == nil || repository == nil || report.Kind() != artifact.KindEvaluation ||
		acceptancePolicyCodec.ValidateIdentity(acceptance) != nil || evaluatorCodec.ValidateIdentity(evaluator) != nil ||
		evaluator.Plan != plan.identity || evaluator.Acceptance != acceptance.ID || !acceptance.admits(record.Metrics) ||
		run.ValidateIdentity() != nil || record.ValidateIdentity() != nil || run.Outcome != runrecord.OutcomeSucceeded ||
		run.Recipe != plan.body.RuntimeRecipe || run.Environment != plan.body.Environment || run.CodeCommit != plan.body.CodeCommit ||
		!slices.Contains(run.Inputs, plan.identity) || !slices.Contains(run.Outputs, report) ||
		record.Recipe != run.Recipe || record.Run != run.ID || record.Dataset != plan.body.Dataset {
		return EvaluationEvidence{}, errors.New("evaluation: evidence authorities differ")
	}
	content, found, err := artifact.ReadContent(ctx, repository, report)
	if err != nil || !found {
		return EvaluationEvidence{}, errors.Join(err, errors.New("evaluation: report content is absent"))
	}
	if err := content.Validate(); err != nil {
		return EvaluationEvidence{}, err
	}
	shards, err := evaluationReportShards(ctx, repository, report, plan.body.Split)
	if err != nil {
		return EvaluationEvidence{}, err
	}
	var causalBinding *runrecord.CausalContext
	if len(causal) == 1 {
		causalBinding = &causal[0]
	}
	value := EvaluationEvidence{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Acceptance: acceptance.ID, Evaluator: evaluator.ID,
		Report: report, Run: run.ID, Evaluation: record.ID,
		ModelDefinition: plan.body.ModelDefinition, Recipe: plan.body.RuntimeRecipe,
		Dataset: plan.body.Dataset, Split: plan.body.Split, Shards: shards,
		Environment: plan.body.Environment, CodeCommit: plan.body.CodeCommit,
		Phases: slices.Clone(run.Phases), Metrics: slices.Clone(record.Metrics),
		Causal: causalBinding,
	}
	if observation != nil {
		if err := observation.ValidateIdentity(); err != nil || observation.Aggregate.Scope.Attempt != run.ID {
			return EvaluationEvidence{}, errors.Join(err, errors.New("evaluation: resource observation differs from run"))
		}
		resources := observation.Aggregate
		value.ResourceObservation = observation.ID
		value.Resources = &resources
		if err := validateIsolatedResources(ctx, repository, plan, run, resources); err != nil {
			return EvaluationEvidence{}, err
		}
	}
	evidence, err := evaluationEvidenceCodec.New(value)
	if err != nil {
		return EvaluationEvidence{}, err
	}
	return evidence, nil
}

func appendEvaluationEvidence(
	batch *artifact.Batch,
	acceptance AcceptancePolicy,
	evaluator Evaluator,
	evidence EvaluationEvidence,
) error {
	if batch == nil {
		return errors.New("evaluation: evidence batch is absent")
	}
	policyContent, err := acceptance.Content()
	if err != nil {
		return err
	}
	evaluatorContent, err := evaluator.Content()
	if err != nil {
		return err
	}
	evidenceContent, err := evidence.Content()
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, policyContent, evaluatorContent, evidenceContent)
	batch.Lineage = append(batch.Lineage, evidence.Lineage()...)
	if err := runrecord.BindCausality(batch, evidence.ID, evidence.Causal); err != nil {
		return err
	}
	return batch.Validate()
}

func prepareIsolatedEvaluationEvidence(
	ctx context.Context,
	repository artifact.Reader,
	batch *artifact.Batch,
	plan Plan,
	acceptance AcceptancePolicy,
	evaluator Evaluator,
	report artifact.ID,
	run runrecord.Run,
	record runrecord.Evaluation,
	observation runrecord.ObservationChunkSummary,
) (EvaluationEvidence, error) {
	evidence, err := newEvaluationEvidence(
		ctx, repository, plan, acceptance, evaluator, report, run, record, &observation,
	)
	if err != nil {
		return EvaluationEvidence{}, err
	}
	if err := appendEvaluationEvidence(batch, acceptance, evaluator, evidence); err != nil {
		return EvaluationEvidence{}, err
	}
	return evidence, nil
}

func validateIsolatedResources(
	ctx context.Context,
	reader artifact.Reader,
	plan Plan,
	run runrecord.Run,
	resources runrecord.ResourceFitness,
) error {
	execution, found, err := artifact.ReadContent(ctx, reader, plan.body.Execution)
	if err != nil || !found {
		return errors.Join(err, errors.New("evaluation: isolated execution authority is absent"))
	}
	var policy ExecutionPolicy
	if err := strictjson.DecodeBytes(execution.Data, &policy); err != nil || policy.Lifecycle != LifecycleIsolated {
		return errors.Join(err, errors.New("evaluation: resource evidence is not process isolated"))
	}
	wall, wallObserved := resources.Measure(runrecord.ResourceWallNS)
	if err := resources.Validate(); err != nil || resources.Scope.Surface != runrecord.SurfaceEvaluation ||
		resources.Scope.Model.Kind() != artifact.KindModel || resources.Scope.Hardware != run.Environment ||
		resources.Scope.Workload != plan.identity || resources.Scope.Attempt != run.ID ||
		!wallObserved || wall != run.MeasuredNS {
		return errors.Join(err, errors.New("evaluation: isolated resource authorities differ"))
	}
	return nil
}

func evaluationReportShards(
	ctx context.Context,
	repository artifact.Reader,
	report, split artifact.ID,
) ([]artifact.ID, error) {
	queue, seen := []artifact.ID{report}, map[artifact.ID]struct{}{report: {}}
	shards := make([]artifact.ID, 0)
	for len(queue) != 0 {
		current := queue[0]
		queue = queue[1:]
		parents, err := repository.Parents(ctx, current)
		if err != nil {
			return nil, err
		}
		for _, edge := range parents {
			parent := edge.Parent
			if parent.Kind() == artifact.KindDatasetShard && parent != split {
				shards = append(shards, parent)
			}
			if parent.Kind() == artifact.KindEvaluation {
				if _, exists := seen[parent]; !exists {
					seen[parent] = struct{}{}
					queue = append(queue, parent)
				}
			}
		}
	}
	return uniqueArtifactIDs(shards), nil
}

func loadEvidencePlan(ctx context.Context, reader artifact.Reader, id artifact.ID) (Plan, error) {
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil || !found {
		return Plan{}, errors.Join(err, errors.New("evaluation: stored plan is absent"))
	}
	if err := evaluationPlanContract.ValidateContent(content, id); err != nil {
		return Plan{}, err
	}
	plan, err := ParsePlan(content.Data)
	if err != nil || plan.identity != id {
		return Plan{}, errors.Join(err, errors.New("evaluation: stored plan identity differs"))
	}
	if err := validateStoredPlanAuthorities(ctx, reader, plan); err != nil {
		return Plan{}, err
	}
	return plan, nil
}

func validateStoredPlanAuthorities(ctx context.Context, reader artifact.Reader, plan Plan) error {
	for _, id := range []artifact.ID{
		plan.body.ModelDefinition, plan.body.RuntimeRecipe, plan.body.Dataset, plan.body.Split,
		plan.body.CaseProfile, plan.body.Scorer, plan.body.Execution, plan.body.Environment,
	} {
		if _, found, err := reader.Artifact(ctx, id); err != nil || !found {
			return errors.Join(err, fmt.Errorf("evaluation: plan authority %s is absent", id))
		}
	}
	for id, contract := range map[artifact.ID]artifact.DocumentContract{
		plan.body.CaseProfile: caseProfileContract,
		plan.body.Scorer:      scorerProfileContract,
		plan.body.Execution:   executionContract,
	} {
		content, found, err := artifact.ReadContent(ctx, reader, id)
		if err != nil || !found {
			return errors.Join(err, fmt.Errorf("evaluation: plan authority content %s is absent", id))
		}
		if err := contract.ValidateContent(content, id); err != nil {
			return err
		}
	}
	execution, found, err := artifact.ReadContent(ctx, reader, plan.body.Execution)
	if err != nil || !found {
		return errors.Join(err, errors.New("evaluation: execution authority is absent"))
	}
	var policy ExecutionPolicy
	if err := strictjson.DecodeBytes(execution.Data, &policy); err != nil ||
		policy.Lifecycle != LifecycleIsolated && policy.Lifecycle != LifecycleResident {
		return errors.Join(err, errors.New("evaluation: execution authority is invalid"))
	}
	parents, err := reader.Parents(ctx, plan.identity)
	if err != nil {
		return err
	}
	for _, want := range plan.Lineage() {
		found := false
		for _, parent := range parents {
			if parent == want {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("evaluation: plan lineage omits authority %s", want.Parent)
		}
	}
	return nil
}

func canonicalizeEvaluationEvidence(value *EvaluationEvidence) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Plan.Kind() != artifact.KindProfile ||
		value.Acceptance.Kind() != artifact.KindProfile || value.Evaluator.Kind() != artifact.KindEvidence ||
		value.Report.Kind() != artifact.KindEvaluation ||
		value.Run.Kind() != artifact.KindRun || value.Evaluation.Kind() != artifact.KindEvaluation ||
		value.ModelDefinition.Kind() != artifact.KindModelDefinition || value.Recipe.Kind() != artifact.KindRecipe ||
		value.Dataset.Kind() != artifact.KindDataset || value.Split.Kind() != artifact.KindDatasetShard ||
		value.Environment.Kind() != artifact.KindEvidence || !validCommit(value.CodeCommit) ||
		len(value.Phases) == 0 || len(value.Metrics) == 0 {
		return errors.New("evaluation: invalid evidence bundle")
	}
	if value.ResourceObservation.Valid() != (value.Resources != nil) ||
		value.ResourceObservation.Valid() && value.ResourceObservation.Kind() != artifact.KindEvidence {
		return errors.New("evaluation: incomplete resource evidence")
	}
	if value.Resources != nil {
		resources, err := runrecord.NewResourceFitness(*value.Resources)
		if err != nil {
			return errors.Join(errors.New("evaluation: invalid resource evidence"), err)
		}
		value.Resources = &resources
	}
	value.Shards = uniqueArtifactIDs(value.Shards)
	for _, shard := range value.Shards {
		if shard.Kind() != artifact.KindDatasetShard || shard == value.Split {
			return errors.New("evaluation: invalid evidence shard")
		}
	}
	for _, phase := range value.Phases {
		if phase.Phase == "" || phase.DurationNS == 0 {
			return errors.New("evaluation: invalid evidence phase")
		}
	}
	for _, metric := range value.Metrics {
		if metric.Name == "" || !finite(metric.Value) || !validMetricDirection(metric.Direction) {
			return errors.New("evaluation: invalid evidence metric")
		}
	}
	if value.Causal != nil {
		if err := value.Causal.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func uniqueArtifactIDs(values []artifact.ID) []artifact.ID {
	result := slices.Clone(values)
	sort.Slice(result, func(i, j int) bool { return result[i].String() < result[j].String() })
	result = slices.Compact(result)
	return result
}

func cloneEvaluationEvidence(value EvaluationEvidence) EvaluationEvidence {
	value.Shards = slices.Clone(value.Shards)
	value.Phases = slices.Clone(value.Phases)
	value.Metrics = slices.Clone(value.Metrics)
	if value.Resources != nil {
		resources := *value.Resources
		resources.Measures = slices.Clone(resources.Measures)
		if resources.Interactions != nil {
			interactions := *resources.Interactions
			resources.Interactions = &interactions
		}
		value.Resources = &resources
	}
	if value.Causal != nil {
		cloned := *value.Causal
		cloned.Motivation = slices.Clone(value.Causal.Motivation)
		value.Causal = &cloned
	}
	return value
}
