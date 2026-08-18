package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

type evaluationSession interface {
	Evaluate(context.Context, string) error
	Close() error
}

type sessionOpener func(context.Context, manifest, modelRequest) (evaluationSession, error)

func executeModel(ctx context.Context, value manifest, request modelRequest, open sessionOpener) error {
	if ctx == nil || open == nil || len(request.Suites) == 0 {
		return errors.New("evaluate: incomplete model worker")
	}
	session, err := open(ctx, value, request)
	if err != nil {
		return err
	}
	for _, suite := range request.Suites {
		if err := session.Evaluate(ctx, suite); err != nil {
			return errors.Join(err, session.Close())
		}
	}
	return session.Close()
}

type nativeSession struct {
	store       *repodb.Store
	runner      *inference.Runner
	identity    modelrecipe.ProgramIdentity
	environment runrecord.Environment
	commit      string
}

func openEvaluationSession(ctx context.Context, value manifest, request modelRequest) (evaluationSession, error) {
	store, err := repodb.Open(value.Repository)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (evaluationSession, error) {
		return nil, errors.Join(cause, store.Close())
	}
	loaded, err := modelrecipe.ResolveActiveGGUF(ctx, store, request.Path)
	if err != nil {
		return fail(err)
	}
	identity, err := loaded.Identity()
	if err != nil {
		_ = loaded.Close()
		return fail(err)
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{DeviceOrdinal: value.Device})
	if err != nil {
		_ = loaded.Close()
		return fail(err)
	}
	environment, err := runrecord.CurrentEnvironment(fmt.Sprintf("cuda:%d", value.Device), "cuda")
	if err != nil {
		_ = runner.Close()
		return fail(err)
	}
	return &nativeSession{
		store: store, runner: runner, identity: identity,
		environment: environment, commit: value.CodeCommit,
	}, nil
}

func (s *nativeSession) Evaluate(ctx context.Context, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var envelope struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return fmt.Errorf("evaluate: decode suite %q: %w", path, err)
	}
	if envelope.Kind == evaluation.MultipleChoiceKind {
		return s.evaluateMultipleChoice(ctx, data)
	}
	if envelope.Kind == evaluation.GeneratedAnswerKind {
		return s.evaluateGeneratedAnswer(ctx, data)
	}
	if envelope.Kind == evaluation.MMLUProKind {
		return s.evaluateMMLUPro(ctx, data)
	}
	if envelope.Kind == evaluation.GroupedChoiceKind {
		return s.evaluateGroupedChoice(ctx, data)
	}
	if envelope.Kind == evaluation.ProbabilityMassKind {
		return s.evaluateProbabilityMass(ctx, data)
	}
	return s.evaluateExact(ctx, data)
}

func (s *nativeSession) evaluateProbabilityMass(ctx context.Context, data []byte) error {
	var suite evaluation.ProbabilityMassSuite
	if err := json.Unmarshal(data, &suite); err != nil {
		return fmt.Errorf("evaluate: decode probability-mass suite: %w", err)
	}
	compiled, err := evaluation.CompileProbabilityMass(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindProbabilityMass(compiled, s.authorities())
	if err != nil {
		return err
	}
	started := time.Now()
	report, evaluateErr := evaluation.EvaluateProbabilityMass(ctx, s.store, s.runner, compiled, plan)
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if evaluateErr != nil {
		return errors.Join(evaluateErr, s.publishFailure(ctx, plan, measured))
	}
	return s.publishSuccess(ctx, plan, report.ID, measured, []runrecord.Metric{{
		Name: "positive-probability-mass", Value: report.Mean, Direction: runrecord.DirectionMaximize,
	}})
}

func (s *nativeSession) evaluateGroupedChoice(ctx context.Context, data []byte) error {
	var suite evaluation.GroupedChoiceSuite
	if err := json.Unmarshal(data, &suite); err != nil {
		return fmt.Errorf("evaluate: decode grouped-choice suite: %w", err)
	}
	compiled, err := evaluation.CompileGroupedChoice(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindGroupedChoice(compiled, s.authorities())
	if err != nil {
		return err
	}
	started := time.Now()
	report, evaluateErr := evaluation.EvaluateGroupedChoice(ctx, s.store, s.runner, compiled, plan)
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if evaluateErr != nil {
		return errors.Join(evaluateErr, s.publishFailure(ctx, plan, measured))
	}
	metrics := choiceMetrics(report.Accuracy, report.Groups)
	return s.publishSuccess(ctx, plan, report.ID, measured, metrics)
}

func (s *nativeSession) evaluateMMLUPro(ctx context.Context, data []byte) error {
	var suite evaluation.MMLUProSuite
	if err := json.Unmarshal(data, &suite); err != nil {
		return fmt.Errorf("evaluate: decode MMLU-Pro suite: %w", err)
	}
	compiled, err := evaluation.CompileMMLUPro(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindMMLUPro(compiled, s.authorities())
	if err != nil {
		return err
	}
	started := time.Now()
	report, evaluateErr := evaluation.EvaluateMMLUPro(ctx, s.store, s.runner, compiled, plan)
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if evaluateErr != nil {
		return errors.Join(evaluateErr, s.publishFailure(ctx, plan, measured))
	}
	metrics := choiceMetrics(report.Accuracy, report.Categories)
	return s.publishSuccess(ctx, plan, report.ID, measured, metrics)
}

func choiceMetrics(accuracy float64, groups []evaluation.AccuracyGroup) []runrecord.Metric {
	metrics := make([]runrecord.Metric, 1, len(groups)+1)
	metrics[0] = runrecord.Metric{Name: "accuracy", Value: accuracy, Direction: runrecord.DirectionMaximize}
	for _, group := range groups {
		metrics = append(metrics, runrecord.Metric{
			Name: "accuracy/" + group.Name, Value: group.Accuracy, Direction: runrecord.DirectionMaximize,
		})
	}
	return metrics
}

func (s *nativeSession) evaluateGeneratedAnswer(ctx context.Context, data []byte) error {
	var suite evaluation.GeneratedAnswerSuite
	if err := json.Unmarshal(data, &suite); err != nil {
		return fmt.Errorf("evaluate: decode generated-answer suite: %w", err)
	}
	compiled, err := evaluation.CompileGeneratedAnswer(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindGeneratedAnswer(compiled, s.authorities())
	if err != nil {
		return err
	}
	started := time.Now()
	report, evaluateErr := evaluation.EvaluateGeneratedAnswer(ctx, s.store, s.runner, compiled, plan)
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if evaluateErr != nil {
		return errors.Join(evaluateErr, s.publishFailure(ctx, plan, measured))
	}
	return s.publishSuccess(
		ctx, plan, report.ID, measured,
		[]runrecord.Metric{{Name: "accuracy", Value: report.Accuracy, Direction: runrecord.DirectionMaximize}},
	)
}

func (s *nativeSession) evaluateExact(ctx context.Context, data []byte) error {
	var suite evaluation.ExactSuite
	if err := json.Unmarshal(data, &suite); err != nil {
		return fmt.Errorf("evaluate: decode exact suite: %w", err)
	}
	exact, err := evaluation.CompileExact(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindExact(exact, s.authorities())
	if err != nil {
		return err
	}
	started := time.Now()
	report, evaluateErr := evaluation.EvaluateExactSharded(ctx, s.store, s.runner, exact, plan, nil)
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if evaluateErr != nil {
		return errors.Join(evaluateErr, s.publishFailure(ctx, plan, measured))
	}
	return s.publishSuccess(
		ctx, plan, report, measured,
		[]runrecord.Metric{{Name: "exact-accuracy", Value: 1, Direction: runrecord.DirectionMaximize}},
	)
}

func (s *nativeSession) evaluateMultipleChoice(ctx context.Context, data []byte) error {
	var suite evaluation.MultipleChoiceSuite
	if err := json.Unmarshal(data, &suite); err != nil {
		return fmt.Errorf("evaluate: decode multiple-choice suite: %w", err)
	}
	compiled, err := evaluation.CompileMultipleChoice(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindMultipleChoice(compiled, s.authorities())
	if err != nil {
		return err
	}
	started := time.Now()
	report, evaluateErr := evaluation.EvaluateMultipleChoice(ctx, s.store, s.runner, compiled, plan)
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if evaluateErr != nil {
		return errors.Join(evaluateErr, s.publishFailure(ctx, plan, measured))
	}
	return s.publishSuccess(
		ctx, plan, report.ID, measured,
		[]runrecord.Metric{{Name: "accuracy", Value: report.Accuracy, Direction: runrecord.DirectionMaximize}},
	)
}

func (s *nativeSession) authorities() evaluation.ExactAuthorities {
	return evaluation.ExactAuthorities{
		ModelDefinition: s.identity.Definition, RuntimeRecipe: s.identity.Recipe,
		CodeCommit: s.commit, Environment: s.environment.ID,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	}
}

func (s *nativeSession) publishSuccess(
	ctx context.Context,
	plan evaluation.Plan,
	report artifact.ID,
	measured uint64,
	metrics []runrecord.Metric,
) error {
	run, err := runrecord.NewBoundRun(
		s.identity.Recipe, runrecord.OutcomeSucceeded,
		[]artifact.ID{plan.Identity()}, []artifact.ID{report}, "", s.commit,
		s.environment.ID, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		return err
	}
	record, err := runrecord.NewEvaluation(
		s.identity.Recipe, run.ID, plan.Dataset(), metrics,
	)
	if err != nil {
		return err
	}
	return s.publish(ctx, run, &record)
}

func (s *nativeSession) publishFailure(ctx context.Context, plan evaluation.Plan, measured uint64) error {
	run, err := runrecord.NewBoundRun(
		s.identity.Recipe, runrecord.OutcomeFailed,
		[]artifact.ID{plan.Identity()}, nil, "evaluation", s.commit,
		s.environment.ID, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		return err
	}
	return s.publish(ctx, run, nil)
}

func (s *nativeSession) publish(ctx context.Context, run runrecord.Run, record *runrecord.Evaluation) error {
	environmentContent, err := s.environment.Content()
	if err != nil {
		return err
	}
	runContent, err := run.Content()
	if err != nil {
		return err
	}
	contents := []artifact.Content{environmentContent, runContent}
	lineage := run.Lineage()
	if record != nil {
		recordContent, err := record.Content()
		if err != nil {
			return err
		}
		contents = append(contents, recordContent)
		lineage = append(lineage, record.Lineage()...)
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/run/"+run.ID.String(), contents, lineage, nil,
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, s.store, batch)
	return err
}

func (s *nativeSession) Close() error {
	if s == nil {
		return nil
	}
	var runnerErr error
	if s.runner != nil {
		runnerErr = s.runner.Close()
		s.runner = nil
	}
	storeErr := s.store.Close()
	s.store = nil
	return errors.Join(runnerErr, storeErr)
}
