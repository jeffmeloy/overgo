package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
)

// servableModel is one evaluation target: its weights location and
// whether its declared domains admit text suites (undeclared models
// keep full coverage and count as text), which decides whether the
// long-form verification admits it.
type servableModel struct {
	path string
	text bool
}

// servableModels derives the evaluation targets from the store: every
// model whose active inference recipe is trusted and whose recorded
// bytes are present on disk. The listing is deterministic, so the
// parent and its workers agree on model indices without a shared
// manifest file.
func servableModels(ctx context.Context, repository string, limit int) ([]servableModel, error) {
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
	type sized struct {
		path  string
		bytes int64
		text  bool
	}
	models := make([]sized, 0, len(entries))
	for _, entry := range entries {
		if !entry.Present || entry.Stale != "" || entry.Location == "" {
			continue
		}
		info, err := os.Stat(entry.Location)
		if err != nil {
			return nil, err
		}
		domains, declared, err := evaluation.EvalDomains(ctx, store, entry.Model)
		if err != nil {
			return nil, err
		}
		models = append(models, sized{
			path: entry.Location, bytes: info.Size(),
			text: !declared || slices.Contains(domains, evaluation.DomainText),
		})
	}
	if len(models) == 0 {
		return nil, errors.New("evaluate: the store holds no servable local model")
	}
	// Smallest model first (owner rule 2026-09-01): the cheapest models
	// calibrate the pass and surface a broken suite in seconds, and the
	// largest model -- the one whose failure costs hours -- runs only
	// after every smaller model has scored. Recorded bytes on disk are
	// the size; ties fall back to the location so workers agree.
	slices.SortFunc(models, func(a, b sized) int {
		return cmp.Or(cmp.Compare(a.bytes, b.bytes), strings.Compare(a.path, b.path))
	})
	targets := make([]servableModel, len(models))
	for index, model := range models {
		targets[index] = servableModel{path: model.path, text: model.text}
	}
	return targets, nil
}

// errEvaluationSliceElapsed is the cause a worker's context carries when
// its share of the evaluation budget runs out, so budget exhaustion is
// distinguishable from every other cancellation.
var errEvaluationSliceElapsed = errors.New("evaluate: the model's evaluation-budget slice elapsed")

// runAllParent fans one worker process out per servable model, the
// same isolation the manifest path uses: a model that dies cannot take
// the remaining evaluations with it.
func runAllParent(ctx context.Context, repository string, device int, family string, limit int, chatProtocol bool, budget time.Duration) error {
	if budget <= 0 {
		return errors.New("evaluate: the evaluation budget must be positive")
	}
	// The claim precondition holds once, up front: every worker binds
	// its evidence to the verifying commit, so a dirty tree refuses the
	// pass here in one line instead of once per model after each load.
	if _, err := runrecord.VerifyingCommit("."); err != nil {
		return err
	}
	surface, err := longform.Surface(ctx, ".")
	if err != nil {
		return err
	}
	targets, err := servableModels(ctx, repository, limit)
	if err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	deadline := time.Now().Add(budget)
	var failures []error
	for index, target := range targets {
		model := target.path
		remaining := time.Until(deadline)
		if remaining <= 0 {
			failures = append(failures, fmt.Errorf(
				"model %q: unevaluated; the %s evaluation budget elapsed", model, budget))
			continue
		}
		// The long-form record admits a text model before a worker loads
		// it (owner rule 2026-09-04): a refused model is reported and the
		// pass moves on, so a slow runtime or one misreading a long
		// context is found in the minute its long-form run takes, not
		// after hours of scoring. The worker repeats the check as the
		// authority for a hand-launched pass.
		if target.text {
			if err := admitLongForm(ctx, repository, model, surface); err != nil {
				fmt.Printf("model %q: REFUSED: %v\n", model, err)
				failures = append(failures, fmt.Errorf("model %q: %w", model, err))
				continue
			}
		}
		// Each model gets an equal share of the budget still unspent, so a
		// single heavy model (the 27B over the 5761-case BBH suite) is
		// bounded to its slice rather than starving the models after it;
		// slices left unused by fast models widen the shares that follow.
		slice := remaining / time.Duration(len(targets)-index)
		fmt.Printf("evaluating %d/%d (%.1fh slice, %.1fh budget left): %s\n",
			index+1, len(targets), slice.Hours(), remaining.Hours(), model)
		// The worker receives the model path: the parent already listed
		// and hashed every servable file once, and a worker that listed
		// again would hash the whole catalog before loading one model.
		arguments := []string{
			"-all", "-worker", "-repo", repository,
			"-device", strconv.Itoa(device), "-model-path", model,
		}
		if family != "" {
			arguments = append(arguments, "-family", family)
		}
		if chatProtocol {
			arguments = append(arguments, "-chat-protocol")
		}
		// The remaining budget bounds this one worker: a single model that
		// would run past the ceiling is terminated and recorded as
		// budget-exceeded, so no model can consume the whole pass -- the
		// exact gap the 27B BBH run exposed, where a per-model floor cannot
		// preempt a model already inside its suite set.
		workerCtx, cancel := context.WithTimeoutCause(ctx, slice, errEvaluationSliceElapsed)
		receipt, runErr := processcontrol.Run(workerCtx, processcontrol.Command{
			Path: executable, Args: arguments, Stdout: os.Stdout, Stderr: os.Stderr,
		})
		cancel()
		switch {
		case errors.Is(context.Cause(workerCtx), errEvaluationSliceElapsed):
			// Budget exhaustion is a recorded outcome, not a failure: the suites the
			// worker committed before the slice elapsed are real recorded
			// evidence, and a model too heavy to finish in its share on this
			// hardware is a measured fact, not a broken pass. The pass fails
			// only on genuine evaluation errors, the same way the smoke lane
			// treats an unavailable model.
			fmt.Printf("model %q: BUDGET-EXCEEDED at its %.1fh slice; committed suites retained, remaining suites not scored\n",
				model, slice.Hours())
		case runErr != nil:
			failures = append(failures, fmt.Errorf("model %q: %w", model, runErr))
		case receipt.ExitCode != 0:
			failures = append(failures, fmt.Errorf("model %q: exit status %d", model, receipt.ExitCode))
		}
	}
	return errors.Join(failures...)
}

// runAllWorker evaluates one servable model against the store's
// derived suites and publishes the evidence through the campaign
// ledger -- the same session the manifest path opens, fed by suites
// compiled from the store instead of files. The budget bounds the
// worker's own pass: a worker launched by hand is the same bounded
// pass the parent runs, not an unbounded one (the 12B BBH chat pass
// ran 2h18m past a 2h ceiling the worker accepted and ignored).
func runAllWorker(ctx context.Context, repository string, device int, family, modelPath string, chatProtocol bool, budget time.Duration) error {
	if strings.TrimSpace(modelPath) == "" {
		return errors.New("evaluate: worker model path is required")
	}
	return runBudgetedPass(ctx, budget, modelPath, func(ctx context.Context) error {
		return runAllWorkerPass(ctx, repository, device, family, modelPath, chatProtocol)
	})
}

// runBudgetedPass runs one model's pass under the budget and turns
// budget exhaustion into the recorded outcome the parent reports: the
// suites committed before the ceiling are evidence, the rest are not
// scored, and the pass is not a failure.
func runBudgetedPass(ctx context.Context, budget time.Duration, model string, pass func(context.Context) error) error {
	if budget <= 0 {
		return errors.New("evaluate: the evaluation budget must be positive")
	}
	ctx, cancel := context.WithTimeoutCause(ctx, budget, errEvaluationSliceElapsed)
	defer cancel()
	err := pass(ctx)
	if errors.Is(context.Cause(ctx), errEvaluationSliceElapsed) {
		fmt.Printf("model %q: BUDGET-EXCEEDED at its %.1fh budget; committed suites retained, remaining suites not scored\n",
			model, budget.Hours())
		return nil
	}
	return err
}

// runAllWorkerPass opens the session and campaigns the derived suites
// under the already-bounded context.
func runAllWorkerPass(ctx context.Context, repository string, device int, family, modelPath string, chatProtocol bool) error {
	commit, err := runrecord.VerifyingCommit(".")
	if err != nil {
		return err
	}
	shared := manifest{Repository: repository, CodeCommit: commit, Device: device, ChatProtocol: chatProtocol}
	session, err := openEvaluationSession(ctx, shared, modelRequest{Path: modelPath, Suites: []string{"derived"}})
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

// admitLongForm reads the model's latest long-form record through a
// reader of its own: the check precedes the session, so no writer is
// open while it runs.
func admitLongForm(ctx context.Context, repository, modelPath, surface string) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	return errors.Join(longform.Admit(ctx, store, modelPath, surface), store.Close())
}

// declareEvalDomain binds a model's evaluation domains in the store,
// resolving the model identity through the servable listing so the
// declaration keys the same manifest every eval consumer reads.
func declareEvalDomain(ctx context.Context, repository, modelPath, domainsCSV string, limit int) error {
	domains := strings.Split(domainsCSV, ",")
	for index := range domains {
		domains[index] = strings.TrimSpace(domains[index])
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	memo := discovery.LoadMemo(ctx, store)
	entries, err := discovery.ServableWithMemo(ctx, store, limit, memo)
	if err != nil {
		return err
	}
	wanted := filepath.Clean(modelPath)
	for _, entry := range entries {
		if !strings.EqualFold(filepath.Clean(entry.Location), wanted) {
			continue
		}
		declaration, err := evaluation.DeclareEvalDomains(ctx, store, entry.Model, domains)
		if err != nil {
			return err
		}
		fmt.Printf("declared %s domains=%v declaration=%s\n", entry.Model, domains, declaration)
		return nil
	}
	return fmt.Errorf("evaluate: %q is not a servable model location", modelPath)
}

// EvaluateDerived compiles the store's suites under this session's
// authorities and campaigns each one, optionally restricted to a
// single family source suffix for bounded smoke runs.
func (s *nativeSession) EvaluateDerived(ctx context.Context, family string) error {
	// The model's input-shaping declaration lands in the store: the
	// runner derives it from the model's own metadata, and the store
	// records it as the authority every consumer reads.
	if prefix := s.runner.ScoringPrefix(); prefix != "" {
		if _, declared, _ := evaluation.LoadPromptTemplate(ctx, s.store, s.model); !declared {
			if _, err := evaluation.PublishPromptTemplate(ctx, s.store, evaluation.PromptTemplate{
				Model: s.model, Source: "tokenizer-metadata", ScoringPrefix: prefix,
			}); err != nil {
				return err
			}
			fmt.Printf("published prompt template: scoring prefix %q\n", prefix)
		}
	}
	// Only the requested family is read and compiled: the catalog's
	// other twenty thousand cases cost minutes per worker for a suite
	// that scores in seconds.
	suites, skipped, err := evaluation.DeriveStoreSuiteFamily(ctx, s.store, s.campaign.Authorities(), family)
	if err != nil {
		return err
	}
	// A declared eval domain routes suites: a DNA model never meets
	// English multiple choice. Undeclared models keep full coverage.
	domains, declared, err := evaluation.EvalDomains(ctx, s.store, s.model)
	if err != nil {
		return err
	}
	suites = evaluation.FilterSuitesForDomains(suites, domains, declared)
	if len(suites) == 0 {
		fmt.Printf("model domains %v admit no derived suite; nothing to evaluate\n", domains)
		return nil
	}
	// A text model meets the long-form admission here as well as in the
	// parent (owner rule 2026-09-04): a worker launched by hand is the
	// same bounded, admitted pass, and the record must pass on this
	// tree's inference surface.
	if !declared || slices.Contains(domains, evaluation.DomainText) {
		surface, err := longform.Surface(ctx, ".")
		if err != nil {
			return err
		}
		if err := longform.Admit(ctx, s.store, s.runner.ModelProperties().Path, surface); err != nil {
			return err
		}
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
		result, err := s.campaign.Evaluate(evaluation.WithProgress(ctx, printProgress), suite)
		if err != nil {
			return fmt.Errorf("evaluate: %s: %w", descriptor.Source, err)
		}
		for _, metric := range result.Metrics {
			fmt.Printf("  %s = %v %s\n", metric.Name, metric.Value, metric.Unit)
		}
	}
	return familyFilterOutcome(selected, declared, domains, family)
}

// familyFilterOutcome decides an empty family selection: a model whose
// declared domains admit other suites but not this family is excluded
// by its own declaration — the outcome is a named skip, exactly
// like the domain filter admitting nothing at all. Undeclared models
// keep full coverage, so a missed filter there names a family the
// catalog cannot serve.
func familyFilterOutcome(selected int, declared bool, domains []string, family string) error {
	if selected > 0 {
		return nil
	}
	if declared {
		fmt.Printf("model domains %v admit no %s suite; nothing to evaluate\n", domains, family)
		return nil
	}
	return errors.New("evaluate: no derived suite matched the family filter")
}
