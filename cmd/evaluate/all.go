package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// servableModelPaths derives the evaluation targets from the store:
// every model whose active inference recipe is trusted and whose
// recorded bytes are present on disk. The listing is deterministic, so
// the parent and its workers agree on model indices without a shared
// manifest file.
func servableModelPaths(ctx context.Context, repository string, limit int) ([]string, error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	memo := discovery.LoadMemo(ctx, store)
	entries, err := discovery.ServableWithMemo(ctx, store, limit, memo)
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.Present && entry.Stale == "" && entry.Location != "" {
			paths = append(paths, entry.Location)
		}
	}
	if len(paths) == 0 {
		return nil, errors.New("evaluate: the store holds no servable local model")
	}
	return paths, nil
}

// runAllParent fans one worker process out per servable model, the
// same isolation the manifest path uses: a model that dies cannot take
// the remaining evaluations with it.
func runAllParent(ctx context.Context, repository string, device int, family string, limit int) error {
	models, err := servableModelPaths(ctx, repository, limit)
	if err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	var failures []error
	for index, model := range models {
		fmt.Printf("evaluating %d/%d: %s\n", index+1, len(models), model)
		arguments := []string{
			"-all", "-worker", "-repo", repository,
			"-device", strconv.Itoa(device), "-model-index", strconv.Itoa(index),
			"-catalog-limit", strconv.Itoa(limit),
		}
		if family != "" {
			arguments = append(arguments, "-family", family)
		}
		command := exec.CommandContext(ctx, executable, arguments...)
		command.Stdout, command.Stderr = os.Stdout, os.Stderr
		if err := command.Run(); err != nil {
			failures = append(failures, fmt.Errorf("model %q: %w", model, err))
		}
	}
	return errors.Join(failures...)
}

// runAllWorker evaluates one servable model against the store's
// derived suites and publishes the evidence through the campaign
// ledger -- the same session the manifest path opens, fed by suites
// compiled from the store instead of files.
func runAllWorker(ctx context.Context, repository string, device int, family string, limit, index int) error {
	models, err := servableModelPaths(ctx, repository, limit)
	if err != nil {
		return err
	}
	if index < 0 || index >= len(models) {
		return errors.New("evaluate: worker model index is invalid")
	}
	commit, err := runrecord.VerifyingCommit(".")
	if err != nil {
		return err
	}
	shared := manifest{Repository: repository, CodeCommit: commit, Device: device}
	session, err := openEvaluationSession(ctx, shared, modelRequest{Path: models[index], Suites: []string{"derived"}})
	if err != nil {
		return err
	}
	native, ok := session.(*nativeSession)
	if !ok {
		return errors.Join(errors.New("evaluate: derived evaluation needs the native session"), session.Close())
	}
	if err := native.EvaluateDerived(ctx, family); err != nil {
		return errors.Join(err, session.Close())
	}
	return session.Close()
}

// EvaluateDerived compiles the store's suites under this session's
// authorities and campaigns each one, optionally restricted to a
// single family source suffix for bounded smoke runs.
func (s *nativeSession) EvaluateDerived(ctx context.Context, family string) error {
	suites, skipped, err := evaluation.DeriveStoreSuites(ctx, s.store, s.campaign.Authorities())
	if err != nil {
		return err
	}
	for name, dropped := range skipped {
		fmt.Printf("suite %s: %d case(s) outside the exact vocabulary skipped\n", name, dropped)
	}
	selected := 0
	for _, suite := range suites {
		descriptor := suite.Descriptor()
		if family != "" && !strings.HasSuffix(descriptor.Source, "/"+family) {
			continue
		}
		selected++
		fmt.Printf("suite %s (%s, %d cases)\n", descriptor.Source, descriptor.Kind, descriptor.Cases)
		result, err := s.campaign.Evaluate(ctx, suite)
		if err != nil {
			return fmt.Errorf("evaluate: %s: %w", descriptor.Source, err)
		}
		for _, metric := range result.Metrics {
			fmt.Printf("  %s = %v %s\n", metric.Name, metric.Value, metric.Unit)
		}
	}
	if selected == 0 {
		return errors.New("evaluate: no derived suite matched the family filter")
	}
	return nil
}
