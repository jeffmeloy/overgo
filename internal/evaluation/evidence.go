package evaluation

import (
	"context"
	"errors"
	"fmt"
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
}

var evaluationEvidenceCodec = artifact.JSONDocumentCodec(
	"evaluation evidence", artifact.KindEvidence, evaluationEvidenceMedia, evaluationEvidenceSchema,
	canonicalizeEvaluationEvidence, func(value EvaluationEvidence) artifact.ID { return value.ID },
	func(value *EvaluationEvidence, id artifact.ID) { value.ID = id }, cloneEvaluationEvidence,
)

func LoadEvaluationEvidence(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (EvaluationEvidence, bool, error) {
	content, found, err := reader.Content(ctx, id)
	if err != nil || !found || content.Descriptor.MediaType != evaluationEvidenceMedia || content.Descriptor.Schema != evaluationEvidenceSchema {
		return EvaluationEvidence{}, false, err
	}
	value, err := evaluationEvidenceCodec.Parse(content.Data)
	if err != nil || value.ID != id {
		return EvaluationEvidence{}, false, err
	}
	return value, true, nil
}

// Content returns the native RepoDB evidence document.
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
	run, err := loadEvidenceRun(ctx, reader, value.Run)
	if err != nil {
		return err
	}
	record, err := loadEvidenceRecord(ctx, reader, value.Evaluation)
	if err != nil {
		return err
	}
	if run.Outcome != runrecord.OutcomeSucceeded || run.Recipe != value.Recipe || run.Environment != value.Environment ||
		run.CodeCommit != value.CodeCommit || !slices.Contains(run.Inputs, value.Plan) || !slices.Contains(run.Outputs, value.Report) ||
		record.Recipe != value.Recipe || record.Run != value.Run || record.Dataset != value.Dataset || !policy.admits(record.Metrics) ||
		!slices.Equal(run.Phases, value.Phases) || !slices.Equal(record.Metrics, value.Metrics) {
		return errors.New("evaluation: stored run or metrics differ from evidence")
	}
	reportContent, found, err := reader.Content(ctx, value.Report)
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
) (EvaluationEvidence, error) {
	if ctx == nil || repository == nil || report.Kind() != artifact.KindEvaluation ||
		acceptancePolicyCodec.ValidateIdentity(acceptance) != nil || evaluatorCodec.ValidateIdentity(evaluator) != nil ||
		evaluator.Plan != plan.identity || evaluator.Acceptance != acceptance.ID || !acceptance.admits(record.Metrics) ||
		run.ValidateIdentity() != nil || record.ValidateIdentity() != nil || run.Outcome != runrecord.OutcomeSucceeded ||
		run.Recipe != plan.body.RuntimeRecipe || run.Environment != plan.body.Environment || run.CodeCommit != plan.body.CodeCommit ||
		!slices.Contains(run.Inputs, plan.identity) || !slices.Contains(run.Outputs, report) ||
		record.Recipe != run.Recipe || record.Run != run.ID || record.Dataset != plan.body.Dataset {
		return EvaluationEvidence{}, errors.New("evaluation: evidence authorities differ")
	}
	content, found, err := repository.Content(ctx, report)
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
	evidence, err := evaluationEvidenceCodec.New(EvaluationEvidence{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Acceptance: acceptance.ID, Evaluator: evaluator.ID,
		Report: report, Run: run.ID, Evaluation: record.ID,
		ModelDefinition: plan.body.ModelDefinition, Recipe: plan.body.RuntimeRecipe,
		Dataset: plan.body.Dataset, Split: plan.body.Split, Shards: shards,
		Environment: plan.body.Environment, CodeCommit: plan.body.CodeCommit,
		Phases: slices.Clone(run.Phases), Metrics: slices.Clone(record.Metrics),
	})
	if err != nil {
		return EvaluationEvidence{}, err
	}
	policyContent, err := acceptance.Content()
	if err != nil {
		return EvaluationEvidence{}, err
	}
	evaluatorContent, err := evaluator.Content()
	if err != nil {
		return EvaluationEvidence{}, err
	}
	evidenceContent, err := evidence.Content()
	if err != nil {
		return EvaluationEvidence{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/evidence/"+evidence.ID.String(), []artifact.Content{policyContent, evaluatorContent, evidenceContent},
		evidence.Lineage(), nil,
	)
	if err != nil {
		return EvaluationEvidence{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return evidence, err
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
	content, found, err := reader.Content(ctx, id)
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
		content, found, err := reader.Content(ctx, id)
		if err != nil || !found {
			return errors.Join(err, fmt.Errorf("evaluation: plan authority content %s is absent", id))
		}
		if err := contract.ValidateContent(content, id); err != nil {
			return err
		}
	}
	execution, found, err := reader.Content(ctx, plan.body.Execution)
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

func loadEvidenceRun(ctx context.Context, reader artifact.Reader, id artifact.ID) (runrecord.Run, error) {
	content, found, err := reader.Content(ctx, id)
	if err != nil || !found {
		return runrecord.Run{}, errors.Join(err, errors.New("evaluation: stored run is absent"))
	}
	return runrecord.ParseRun(content.Data)
}

func loadEvidenceRecord(ctx context.Context, reader artifact.Reader, id artifact.ID) (runrecord.Evaluation, error) {
	content, found, err := reader.Content(ctx, id)
	if err != nil || !found {
		return runrecord.Evaluation{}, errors.Join(err, errors.New("evaluation: stored metric record is absent"))
	}
	return runrecord.ParseEvaluation(content.Data)
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
	return value
}
