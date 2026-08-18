package evaluation

import (
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	evaluationEvidenceVersion uint16 = 1
	evaluationEvidenceMedia          = "application/vnd.overgo.evaluation-evidence+json"
	evaluationEvidenceSchema         = "overgo/evaluation-evidence/v1"
)

type EvaluationEvidence struct {
	ID              artifact.ID             `json:"-"`
	Version         uint16                  `json:"version"`
	Plan            artifact.ID             `json:"plan"`
	Acceptance      artifact.ID             `json:"acceptance"`
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

func PublishEvaluationEvidence(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	acceptance AcceptancePolicy,
	report artifact.ID,
	run runrecord.Run,
	record runrecord.Evaluation,
) (EvaluationEvidence, error) {
	if ctx == nil || repository == nil || report.Kind() != artifact.KindEvaluation ||
		acceptancePolicyCodec.ValidateIdentity(acceptance) != nil || !acceptance.admits(record.Metrics) ||
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
		Version: evaluationEvidenceVersion, Plan: plan.identity, Acceptance: acceptance.ID,
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
	evidenceContent, err := evaluationEvidenceCodec.Content(evidence)
	if err != nil {
		return EvaluationEvidence{}, err
	}
	parents := []artifact.ID{
		evidence.Plan, evidence.Acceptance, evidence.Report, evidence.Run, evidence.Evaluation,
		evidence.ModelDefinition, evidence.Recipe, evidence.Dataset, evidence.Split, evidence.Environment,
	}
	parents = append(parents, evidence.Shards...)
	batch, err := artifact.NewDocumentBatch(
		"evaluation/evidence/"+evidence.ID.String(), []artifact.Content{policyContent, evidenceContent},
		artifact.DependencyLineage(evidence.ID, uniqueArtifactIDs(parents)...), nil,
	)
	if err != nil {
		return EvaluationEvidence{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return evidence, err
}

func evaluationReportShards(
	ctx context.Context,
	repository artifact.Repository,
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

func canonicalizeEvaluationEvidence(value *EvaluationEvidence) error {
	if value == nil || value.Version != evaluationEvidenceVersion || value.Plan.Kind() != artifact.KindProfile ||
		value.Acceptance.Kind() != artifact.KindProfile || value.Report.Kind() != artifact.KindEvaluation ||
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
