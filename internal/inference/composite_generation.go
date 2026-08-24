package inference

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/composition"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

const (
	compositeGenerationOutputMediaType = "application/vnd.overgo.composite-generation-output+json"
	compositeGenerationOutputSchema    = "overgo/composite-generation-output/v1"
)

type compositeGenerationOutput struct {
	Version    uint16                             `json:"version"`
	Arm        composition.CompositeGenerationArm `json:"arm"`
	Seed       uint64                             `json:"seed"`
	Input      artifact.ID                        `json:"input"`
	Dimensions []uint64                           `json:"dimensions"`
	Values     []float32                          `json:"values"`
	ID         artifact.ID                        `json:"-"`
}

var compositeGenerationOutputCodec = artifact.JSONDocumentCodec(
	"composite generation output", artifact.KindOutput,
	compositeGenerationOutputMediaType, compositeGenerationOutputSchema,
	canonicalizeCompositeGenerationOutput,
	func(value compositeGenerationOutput) artifact.ID { return value.ID },
	func(value *compositeGenerationOutput, id artifact.ID) { value.ID = id },
	func(value compositeGenerationOutput) compositeGenerationOutput {
		value.Dimensions = slices.Clone(value.Dimensions)
		value.Values = slices.Clone(value.Values)
		return value
	},
)

// CompositeGenerationCase binds one held-out input and seed to exact tokenized
// source and target inputs. Tokens remain runtime values; the durable input ID
// is their immutable dataset authority.
type CompositeGenerationCase struct {
	Seed         uint64
	Input        artifact.ID
	SourceTokens []tokenizer.TokenID
	TargetTokens []tokenizer.TokenID
}

// CompositeGenerationMeasurement is observer-owned latency and memory evidence
// for exactly one arm invocation.
type CompositeGenerationMeasurement struct {
	StartedUnixNS int64
	MeasuredNS    uint64
	Usage         runrecord.ServingUsage
	Resources     runrecord.ServingResources
	Phases        []runrecord.PhaseMetric
	Hardware      []runrecord.ServingHardwareSample
}

// CompositeGenerationObserver executes one callback while collecting actual
// latency, memory, transfer, phase, and device-allocation facts.
type CompositeGenerationObserver interface {
	Observe(
		context.Context,
		composition.CompositeGenerationArm,
		func(context.Context) (reference.Value, error),
	) (reference.Value, CompositeGenerationMeasurement, error)
}

// CompositeGenerationMetricEvaluator scores one exact generated tensor.
type CompositeGenerationMetricEvaluator interface {
	Evaluate(
		context.Context,
		composition.CompositeGenerationArm,
		artifact.ID,
		reference.Value,
	) ([]runrecord.Metric, error)
}

// CompositeGenerationExecutionRequest supplies control recipes and immutable
// evaluation authorities. The composed recipe and execution plan always come
// from the already-open active runtime.
type CompositeGenerationExecutionRequest struct {
	Key                  string
	TargetBaselineRecipe artifact.ID
	SourceAblatedRecipe  artifact.ID
	BridgeAblatedRecipe  artifact.ID
	Dataset              artifact.ID
	HeldOutSplit         artifact.ID
	Evaluator            artifact.ID
	Environment          artifact.ID
	CodeCommit           string
	Cases                []CompositeGenerationCase
	Observer             CompositeGenerationObserver
	MetricEvaluator      CompositeGenerationMetricEvaluator
}

// CompositeGenerationExecutionResult returns the atomic OvergoDB commit and its
// complete four-arm evidence identity.
type CompositeGenerationExecutionResult struct {
	Evidence composition.CompositeGenerationEvidence
	Commit   artifact.CommitID
}

// GenerateEvidence executes every causal arm through the active composition
// sessions and atomically commits outputs, runs, evaluations, observations,
// the compiled plan, and their final evidence graph.
func (runtime *ProductionComposition) GenerateEvidence(
	ctx context.Context,
	repository artifact.Repository,
	request CompositeGenerationExecutionRequest,
) (CompositeGenerationExecutionResult, error) {
	if runtime == nil || ctx == nil || repository == nil || request.Observer == nil || request.MetricEvaluator == nil ||
		request.Key == "" || len(request.Cases) == 0 {
		return CompositeGenerationExecutionResult{}, errors.New("inference: composite generation execution authority is absent")
	}
	if err := ctx.Err(); err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	armRecipes := map[composition.CompositeGenerationArm]artifact.ID{
		composition.CompositeGenerationTargetBaseline: request.TargetBaselineRecipe,
		composition.CompositeGenerationComposed:       runtime.plan.CompositionRecipe,
		composition.CompositeGenerationSourceAblated:  request.SourceAblatedRecipe,
		composition.CompositeGenerationBridgeAblated:  request.BridgeAblatedRecipe,
	}
	if err := validateCompositeGenerationExecutionAuthorities(ctx, repository, runtime.plan, request, armRecipes); err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	sourceBefore, err := requireCompositeGenerationDescriptor(ctx, repository, runtime.plan.SourceModel)
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	targetBefore, err := requireCompositeGenerationDescriptor(ctx, repository, runtime.plan.TargetModel)
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	bridge, err := requireCompositeGenerationDescriptor(ctx, repository, runtime.plan.BridgeWeights[tensor.FirstOffset])
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	planContent, err := runtime.plan.Content()
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	contents := []artifact.Content{planContent}
	lineage := slices.Clone(runtime.plan.Lineage())
	trials := make([]composition.CompositeGenerationTrial, 0, len(request.Cases)*len(compositeGenerationArms))
	for _, generationCase := range request.Cases {
		if _, err := requireCompositeGenerationDescriptor(ctx, repository, generationCase.Input); err != nil {
			return CompositeGenerationExecutionResult{}, err
		}
		for _, arm := range compositeGenerationArms {
			output, measurement, observeErr := request.Observer.Observe(ctx, arm, func(executionContext context.Context) (reference.Value, error) {
				return runtime.ForwardArm(executionContext, arm, generationCase.SourceTokens, generationCase.TargetTokens)
			})
			if observeErr != nil {
				return CompositeGenerationExecutionResult{}, observeErr
			}
			if err := validateCompositeGenerationMeasurement(measurement); err != nil {
				return CompositeGenerationExecutionResult{}, err
			}
			outputDocument, err := compositeGenerationOutputCodec.New(compositeGenerationOutput{
				Version: artifact.InitialDocumentVersion, Arm: arm, Seed: generationCase.Seed, Input: generationCase.Input,
				Dimensions: output.Shape.Slice(), Values: slices.Clone(output.Data),
			})
			if err != nil {
				return CompositeGenerationExecutionResult{}, err
			}
			outputContent, err := compositeGenerationOutputCodec.Content(outputDocument)
			if err != nil {
				return CompositeGenerationExecutionResult{}, err
			}
			recipeID := armRecipes[arm]
			run, err := runrecord.NewBoundRun(
				recipeID, runrecord.OutcomeSucceeded, []artifact.ID{generationCase.Input},
				[]artifact.ID{outputDocument.ID}, "", request.CodeCommit, request.Environment,
				measurement.MeasuredNS, measurement.Phases,
			)
			if err != nil {
				return CompositeGenerationExecutionResult{}, err
			}
			metrics, err := request.MetricEvaluator.Evaluate(ctx, arm, generationCase.Input, output)
			if err != nil {
				return CompositeGenerationExecutionResult{}, err
			}
			evaluation, err := runrecord.NewEvaluation(recipeID, run.ID, request.Dataset, metrics)
			if err != nil {
				return CompositeGenerationExecutionResult{}, err
			}
			observation, err := runrecord.NewServingObservation(runrecord.ServingObservation{
				Model: runtime.plan.TargetModel, Recipe: recipeID, Environment: request.Environment,
				Run: run.ID, Task: runtime.plan.Task, Outcome: runrecord.OutcomeSucceeded,
				StartedUnixNS: measurement.StartedUnixNS, MeasuredNS: measurement.MeasuredNS,
				Usage: measurement.Usage, Resources: measurement.Resources,
				Phases: slices.Clone(measurement.Phases), Hardware: slices.Clone(measurement.Hardware),
			})
			if err != nil {
				return CompositeGenerationExecutionResult{}, err
			}
			for _, document := range []interface {
				Content() (artifact.Content, error)
			}{run, evaluation, observation} {
				content, contentErr := document.Content()
				if contentErr != nil {
					return CompositeGenerationExecutionResult{}, contentErr
				}
				contents = append(contents, content)
			}
			lineage = append(lineage, run.Lineage()...)
			lineage = append(lineage, evaluation.Lineage()...)
			lineage = append(lineage, observation.Lineage()...)
			lineage = append(lineage, artifact.Lineage{
				Child: outputDocument.ID, Parent: generationCase.Input, Relation: artifact.RelationDerivedFrom,
			})
			contents = append(contents, outputContent)
			trials = append(trials, composition.CompositeGenerationTrial{
				Seed: generationCase.Seed, Arm: arm, Input: generationCase.Input, Output: outputDocument.ID,
				Run: run.ID, Evaluation: evaluation.ID, Observation: observation.ID,
			})
		}
	}
	sourceAfter, err := requireCompositeGenerationDescriptor(ctx, repository, runtime.plan.SourceModel)
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	targetAfter, err := requireCompositeGenerationDescriptor(ctx, repository, runtime.plan.TargetModel)
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	evidence, err := (composition.CompositeGenerationEvidenceAuthority{}).New(composition.CompositeGenerationEvidence{
		SourceModel: composition.CompositeGenerationFrozenModel{Before: sourceBefore, After: sourceAfter},
		TargetModel: composition.CompositeGenerationFrozenModel{Before: targetBefore, After: targetAfter},
		Bridge:      bridge, ExecutionPlan: runtime.plan.ID,
		TargetBaselineRecipe: request.TargetBaselineRecipe, CompositionRecipe: runtime.plan.CompositionRecipe,
		SourceAblatedRecipe: request.SourceAblatedRecipe, BridgeAblatedRecipe: request.BridgeAblatedRecipe,
		Dataset: request.Dataset, HeldOutSplit: request.HeldOutSplit, Evaluator: request.Evaluator,
		Trials: trials,
	})
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	evidenceContent, err := evidence.Content()
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	contents = append(contents, evidenceContent)
	lineage = append(lineage, evidence.Lineage()...)
	batch, err := artifact.NewDocumentBatch(request.Key, contents, lineage, nil)
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	commit, err := artifact.CommitBatch(ctx, repository, batch)
	if err != nil {
		return CompositeGenerationExecutionResult{}, err
	}
	return CompositeGenerationExecutionResult{Evidence: evidence, Commit: commit}, nil
}

var compositeGenerationArms = [...]composition.CompositeGenerationArm{
	composition.CompositeGenerationTargetBaseline,
	composition.CompositeGenerationComposed,
	composition.CompositeGenerationSourceAblated,
	composition.CompositeGenerationBridgeAblated,
}

func validateCompositeGenerationExecutionAuthorities(
	ctx context.Context,
	repository artifact.Repository,
	plan composition.CompositionExecutionPlan,
	request CompositeGenerationExecutionRequest,
	armRecipes map[composition.CompositeGenerationArm]artifact.ID,
) error {
	if plan.ValidateIdentity() != nil || len(plan.BridgeWeights) != tensor.SingletonExtent ||
		request.Dataset.Kind() != artifact.KindDataset || request.HeldOutSplit.Kind() != artifact.KindDatasetShard ||
		request.Evaluator.Kind() != artifact.KindEvidence || request.Environment.Kind() != artifact.KindEvidence {
		return errors.New("inference: composite generation immutable authority is invalid")
	}
	seen := make(map[artifact.ID]struct{}, len(armRecipes))
	for _, arm := range compositeGenerationArms {
		id := armRecipes[arm]
		if id.Kind() != artifact.KindRecipe {
			return errors.New("inference: composite generation arm recipe is invalid")
		}
		seen[id] = struct{}{}
	}
	if len(seen) != len(armRecipes) {
		return errors.New("inference: composite generation arm recipes are not distinct")
	}
	authorities := []artifact.ID{
		plan.CompositionRecipe, request.Dataset, request.HeldOutSplit, request.Evaluator, request.Environment,
	}
	for _, arm := range compositeGenerationArms {
		authorities = append(authorities, armRecipes[arm])
	}
	for _, id := range authorities {
		if _, err := requireCompositeGenerationDescriptor(ctx, repository, id); err != nil {
			return err
		}
	}
	return nil
}

func requireCompositeGenerationDescriptor(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (artifact.Descriptor, error) {
	descriptor, found, err := reader.Artifact(ctx, id)
	if err != nil || !found || descriptor.ID != id || descriptor.Validate() != nil {
		return artifact.Descriptor{}, errors.Join(err, errors.New("inference: composite generation artifact is absent"))
	}
	return descriptor, nil
}

func validateCompositeGenerationMeasurement(value CompositeGenerationMeasurement) error {
	if value.StartedUnixNS <= 0 || !checked.Nonzero(value.MeasuredNS) ||
		!checked.Nonzero(value.Resources.PeakHostBytes) && !checked.Nonzero(value.Resources.PeakDeviceBytes) {
		return errors.New("inference: composite generation latency or memory evidence is absent")
	}
	return nil
}

func canonicalizeCompositeGenerationOutput(value *compositeGenerationOutput) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Input.Kind() != artifact.KindOutput || len(value.Dimensions) == 0 {
		return errors.New("inference: invalid composite generation output authority")
	}
	validArm := false
	for _, arm := range compositeGenerationArms {
		validArm = validArm || value.Arm == arm
	}
	shape, err := compositionTensor(value.Dimensions...)
	if !validArm || err != nil {
		return errors.Join(err, errors.New("inference: invalid composite generation output shape or arm"))
	}
	elements, err := shape.Elements()
	if err != nil || elements != uint64(len(value.Values)) {
		return errors.Join(err, errors.New("inference: composite generation output storage differs from shape"))
	}
	for _, scalar := range value.Values {
		if !checked.Finite32(scalar) {
			return errors.New("inference: composite generation output is non-finite")
		}
	}
	return nil
}
