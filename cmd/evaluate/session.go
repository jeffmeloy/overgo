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
	var suite evaluation.ExactSuite
	if err := json.Unmarshal(data, &suite); err != nil {
		return fmt.Errorf("evaluate: decode suite %q: %w", path, err)
	}
	exact, err := evaluation.CompileExact(suite)
	if err != nil {
		return err
	}
	plan, err := evaluation.BindExact(exact, evaluation.ExactAuthorities{
		ModelDefinition: s.identity.Definition,
		RuntimeRecipe:   s.identity.Recipe,
		CodeCommit:      s.commit,
		Environment:     s.environment.ID,
		Execution:       evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return err
	}
	started := time.Now()
	report, evaluateErr := evaluation.EvaluateExactSharded(ctx, s.store, s.runner, exact, plan, nil)
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if evaluateErr != nil {
		return errors.Join(evaluateErr, s.publishFailure(ctx, plan, measured))
	}
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
		s.identity.Recipe, run.ID, plan.Dataset(),
		[]runrecord.Metric{{Name: "exact-accuracy", Value: 1, Direction: runrecord.DirectionMaximize}},
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
