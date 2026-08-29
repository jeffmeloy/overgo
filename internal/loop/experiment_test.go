package loop

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type strategyExperimentTestFixture struct {
	ctx          context.Context
	store        *overgodb.Store
	task         recipe.AgentTaskContract
	baseline     string
	candidates   []StrategyExperimentCandidate
	strategies   []Strategy
	attempts     []runrecord.AttemptRecord
	trajectories []runrecord.InteractionTrace
	evidence     []evaluation.EvaluationEvidence
	streams      []runrecord.ObservationStream
	evalStreams  []runrecord.ObservationStream
	causalRoot   runrecord.CausalContext
	provider     artifact.ID
}

type strategyExperimentFixtureOptions struct {
	descriptorOnlyWorker      bool
	descriptorOnlyModelRecipe bool
	mismatchedWorkerPrompt    bool
	mismatchedWorkerRecipe    bool
	mismatchedWorkerPolicies  bool
	mismatchedModelBinding    bool
}

func TestStrategyExperiment(t *testing.T) {
	fixture := newStrategyExperimentTestFixture(t)
	fitness := fixture.publishFitness(t, 1, 0)
	experiment, comparison, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline,
		[]StrategyExperimentCandidate{fixture.candidates[1], fixture.candidates[0]},
		[]artifact.ID{fitness.ID},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !experiment.ID.Valid() || len(experiment.Candidates) != 2 || comparison.Winner == nil ||
		*comparison.Winner != fixture.strategies[0].ID || len(comparison.Fitness) != 1 || comparison.Fitness[0] != fitness.ID ||
		!slices.IsSortedFunc(comparison.Ranked, artifact.CompareID) {
		t.Fatalf("comparison = %+v; experiment = %+v", comparison, experiment)
	}
}

func TestImprovementFitnessRejectsTransferredCost(t *testing.T) {
	fixture := newStrategyExperimentTestFixture(t)
	if _, err := runrecord.CompareResourceFitness(fixture.ctx, fixture.store, []runrecord.ResourceFitnessLane{{
		Name: runrecord.ResourceLaneAgent,
		RequiredMetrics: []runrecord.ResourceMetric{
			runrecord.ResourceWallNS, runrecord.ResourcePeakHostBytes,
			runrecord.ResourcePeakDeviceBytes, runrecord.ResourceCostUnits,
		},
		RequireInteractions: true,
		Baseline:            fixture.streams[0],
		Candidate:           fixture.streams[1],
	}}); err == nil {
		t.Fatal("resource regression produced a directed improvement proof")
	}
	_, comparison, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline, fixture.candidates, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if comparison.Winner != nil {
		t.Fatal("scalar attempt cost selected a winner without persisted joint fitness")
	}

	for _, test := range []struct {
		name    string
		options strategyExperimentFixtureOptions
	}{
		{name: "descriptor-only worker", options: strategyExperimentFixtureOptions{descriptorOnlyWorker: true}},
		{name: "descriptor-only model recipe", options: strategyExperimentFixtureOptions{descriptorOnlyModelRecipe: true}},
		{name: "worker prompt mismatch", options: strategyExperimentFixtureOptions{mismatchedWorkerPrompt: true}},
		{name: "worker model recipe mismatch", options: strategyExperimentFixtureOptions{mismatchedWorkerRecipe: true}},
		{name: "worker policies mismatch", options: strategyExperimentFixtureOptions{mismatchedWorkerPolicies: true}},
		{name: "model binding mismatch", options: strategyExperimentFixtureOptions{mismatchedModelBinding: true}},
	} {
		t.Run(test.name, func(t *testing.T) {
			adversarial := newStrategyExperimentTestFixtureWithOptions(t, test.options)
			if _, _, err := CompareStrategyExperiment(
				adversarial.ctx, adversarial.store, adversarial.task, adversarial.baseline,
				adversarial.candidates, nil,
			); err == nil {
				t.Fatal("untyped or semantically foreign strategy authority entered comparison")
			}
		})
	}
}

func TestStrategyComparisonRequiresIdentity(t *testing.T) {
	fixture := newStrategyExperimentTestFixture(t)
	_, comparison, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline, fixture.candidates, nil,
	)
	if err != nil {
		t.Fatalf("exact stored candidates refused: %v", err)
	}
	if comparison.Winner != nil || len(comparison.Evidence) != 2 {
		t.Fatalf("missing dominance proof selected a winner: %+v", comparison)
	}
	for _, evidence := range comparison.Evidence {
		candidate := fixture.candidateByStrategy(t, evidence.Strategy)
		attempt := fixture.attemptByID(t, candidate.Attempt)
		if evidence.Lease != candidate.Lease || evidence.Attempt != attempt.ID || evidence.Trajectory != attempt.Trajectory ||
			evidence.EvaluationEvidence != candidate.EvaluationEvidence {
			t.Fatalf("derived strategy evidence differs: %+v", evidence)
		}
	}

	foreignTask := fixture.task
	foreignTask.ID = artifact.ID{}
	foreignTask.Objective = "Compare a different exact task."
	foreignTask, err = recipe.NewAgentTaskContract(foreignTask)
	if err != nil {
		t.Fatal(err)
	}
	fixture.publishDocument(t, "identity-task/foreign", foreignTask.Content, foreignTask.Lineage())
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, foreignTask, fixture.baseline, fixture.candidates, nil,
	); err == nil {
		t.Fatal("well-formed candidates entered a foreign task comparison")
	}
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, strings.Repeat("b", 40), fixture.candidates, nil,
	); err == nil {
		t.Fatal("well-formed candidates entered a foreign baseline comparison")
	}

	foreignCommit := fixture.attempts[0]
	foreignCommit.ID = artifact.ID{}
	foreignCommit.CodeCommit = strings.Repeat("b", 40)
	foreignCommit, err = runrecord.NewAttemptRecord(foreignCommit)
	if err != nil {
		t.Fatal(err)
	}
	fixture.publishDocument(t, "identity-attempt/foreign-commit", foreignCommit.Content, foreignCommit.Lineage())
	foreignCommitCandidate := fixture.candidates[0]
	foreignCommitCandidate.Attempt = foreignCommit.ID
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline,
		[]StrategyExperimentCandidate{foreignCommitCandidate, fixture.candidates[1]}, nil,
	); err == nil || !strings.Contains(err.Error(), "attempt commit") {
		t.Fatalf("attempt for another commit was not refused at the baseline boundary: %v", err)
	}

	recordable := func(base runrecord.AttemptRecord, label string, strategy artifact.ID) artifact.ID {
		t.Helper()
		base.ID, base.Strategy, base.StrategyID = artifact.ID{}, label, strategy
		attempt, err := runrecord.NewAttemptRecord(base)
		if err != nil {
			t.Fatalf("recordable attempt refused: %v", err)
		}
		fixture.publishDocument(t, "identity-attempt/"+attempt.ID.String(), attempt.Content, attempt.Lineage())
		return attempt.ID
	}
	for name, attempt := range map[string]artifact.ID{
		"anonymous":  recordable(fixture.attempts[0], "", artifact.ID{}),
		"label-only": recordable(fixture.attempts[0], "display-label", artifact.ID{}),
		"mismatched": recordable(fixture.attempts[0], "", fixture.strategies[1].ID),
	} {
		invalid := fixture.candidates[0]
		invalid.Attempt = attempt
		if _, _, err := CompareStrategyExperiment(
			fixture.ctx, fixture.store, fixture.task, fixture.baseline,
			[]StrategyExperimentCandidate{invalid, fixture.candidates[1]}, nil,
		); err == nil {
			t.Fatalf("%s attempt entered comparison", name)
		}
	}

	for name, lease := range map[string]artifact.ID{
		"missing": {},
		"foreign": fixture.candidates[1].Lease,
	} {
		unbound := fixture.attempts[0]
		unbound.ID, unbound.WorkLease = artifact.ID{}, lease
		unbound, err = runrecord.NewAttemptRecord(unbound)
		if err != nil {
			t.Fatal(err)
		}
		fixture.publishDocument(t, "identity-attempt/work-lease-"+name, unbound.Content, unbound.Lineage())
		invalid := fixture.candidates[0]
		invalid.Attempt = unbound.ID
		if _, _, err := CompareStrategyExperiment(
			fixture.ctx, fixture.store, fixture.task, fixture.baseline,
			[]StrategyExperimentCandidate{invalid, fixture.candidates[1]}, nil,
		); err == nil || !strings.Contains(err.Error(), "attempt work lease") {
			t.Fatalf("%s attempt-to-lease binding entered comparison: %v", name, err)
		}
	}

	firstLease, found, err := plan.ReadWorkLease(fixture.ctx, fixture.store, fixture.candidates[0].Lease)
	if err != nil || !found {
		t.Fatalf("read first work lease: found=%v err=%v", found, err)
	}
	aliasLease, err := plan.NewWorkLease(plan.WorkLease{
		Task: "strategy-worktree-alias", Worktree: strings.TrimSuffix(firstLease.Worktree, "/") + "/.",
		Branch: "codex/strategy-worktree-alias", Role: "experiment", TargetHead: fixture.baseline,
		ConflictsWith: []string{}, Resources: plan.Resources{CPUThreads: 1, HostRAMGiB: 1},
		ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano),
	})
	if err != nil {
		t.Fatal(err)
	}
	encodedAliasLease, err := json.Marshal(aliasLease)
	if err != nil {
		t.Fatal(err)
	}
	aliasLease, err = plan.RecordWorkLease(fixture.ctx, fixture.store, encodedAliasLease)
	if err != nil {
		t.Fatal(err)
	}
	aliasedAttempt := fixture.attempts[1]
	aliasedAttempt.ID, aliasedAttempt.WorkLease = artifact.ID{}, aliasLease.ID
	aliasedAttempt, err = runrecord.NewAttemptRecord(aliasedAttempt)
	if err != nil {
		t.Fatal(err)
	}
	fixture.publishDocument(t, "identity-attempt/worktree-alias", aliasedAttempt.Content, aliasedAttempt.Lineage())
	aliasedCandidate := fixture.candidates[1]
	aliasedCandidate.Lease, aliasedCandidate.Attempt = aliasLease.ID, aliasedAttempt.ID
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline,
		[]StrategyExperimentCandidate{fixture.candidates[0], aliasedCandidate}, nil,
	); err == nil {
		t.Fatal("two spellings of one filesystem worktree entered an isolated comparison")
	}

	wrongStrategy := fixture.candidates[0]
	wrongStrategy.Strategy = fixture.strategies[1].ID
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline,
		[]StrategyExperimentCandidate{wrongStrategy, fixture.candidates[1]}, nil,
	); err == nil {
		t.Fatal("foreign stored strategy entered the candidate execution")
	}
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline,
		[]StrategyExperimentCandidate{fixture.candidates[0], fixture.candidates[0]}, nil,
	); err == nil {
		t.Fatal("one stored strategy was admitted as a cross-strategy comparison")
	}

	detached := fixture.attempts[0]
	detached.ID = artifact.ID{}
	detached.Trajectory = testutil.ArtifactID(t, artifact.KindEvidence, "detached-trajectory")
	detached, err = runrecord.NewAttemptRecord(detached)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(fixture.ctx, artifact.Batch{
		Key:       "identity-attempt/detached-authority",
		Artifacts: []artifact.Descriptor{{ID: detached.Trajectory}},
	}); err != nil {
		t.Fatal(err)
	}
	fixture.publishDocument(t, "identity-attempt/detached", detached.Content, detached.Lineage())
	detachedCandidate := fixture.candidates[0]
	detachedCandidate.Attempt = detached.ID
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline,
		[]StrategyExperimentCandidate{detachedCandidate, fixture.candidates[1]}, nil,
	); err == nil {
		t.Fatal("detached trajectory entered comparison")
	}

	foreignEvidence := fixture.candidates[0]
	foreignEvidence.EvaluationEvidence = fixture.candidates[1].EvaluationEvidence
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline,
		[]StrategyExperimentCandidate{foreignEvidence, fixture.candidates[1]}, nil,
	); err == nil {
		t.Fatal("well-formed evaluation for another trajectory entered comparison")
	}

	fitness := fixture.publishFitness(t, 1, 0)
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline, fixture.candidates,
		[]artifact.ID{fitness.ID, fitness.ID},
	); err == nil {
		t.Fatal("duplicate directed fitness entered comparison")
	}
	if _, _, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline, fixture.candidates,
		[]artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "missing-fitness")},
	); err == nil {
		t.Fatal("unpublished fitness entered comparison")
	}
}

func newStrategyExperimentTestFixture(t *testing.T) *strategyExperimentTestFixture {
	return newStrategyExperimentTestFixtureWithOptions(t, strategyExperimentFixtureOptions{})
}

func newStrategyExperimentTestFixtureWithOptions(
	t *testing.T,
	options strategyExperimentFixtureOptions,
) *strategyExperimentTestFixture {
	t.Helper()
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	model := id(artifact.KindModel, "strategy model")
	recipeModel := model
	if options.mismatchedModelBinding {
		recipeModel = id(artifact.KindModel, "foreign strategy model")
	}
	modelRecipeDefinition, err := modelrecipe.CapabilityDefinition(recipe.TaskGeneration, recipeModel)
	if err != nil {
		t.Fatal(err)
	}
	modelRecipe := modelRecipeDefinition.ID
	strategyModelRecipeDefinition := modelRecipeDefinition
	if options.mismatchedWorkerRecipe {
		strategyModelRecipeDefinition, err = modelrecipe.CapabilityDefinition(recipe.TaskForecast, recipeModel)
		if err != nil {
			t.Fatal(err)
		}
	}
	prompt := id(artifact.KindFile, "strategy prompt")
	policy := id(artifact.KindProfile, "strategy policy")
	catalog := id(artifact.KindProfile, "strategy catalog")
	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: "experiment-worker", Prompt: prompt, ModelRecipe: modelRecipe, Policies: []artifact.ID{policy},
	})
	if err != nil {
		t.Fatal(err)
	}
	task, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: worker.ID, Objective: "Compare exact stored strategies.", Scope: []string{"internal/loop"},
		AllowedEffects: []string{"workspace:mutation"},
		Acceptance: []recipe.AcceptanceCriterion{{
			Name: "tests", Scope: "internal/loop", Verifier: id(artifact.KindRecipe, "strategy verifier"),
		}},
		Verification: []artifact.ID{id(artifact.KindEvidence, "strategy gate")},
		Budget:       id(artifact.KindEvidence, "strategy budget"), PauseConditions: []string{"authority mismatch"},
	})
	if err != nil {
		t.Fatal(err)
	}
	strategyA, err := NewStrategy(worker, catalog, Config{MaxAttemptsPerStep: 2, MaxInvocations: 5})
	if err != nil {
		t.Fatal(err)
	}
	if options.mismatchedWorkerPrompt || options.mismatchedWorkerRecipe || options.mismatchedWorkerPolicies {
		strategyA.ID = artifact.ID{}
		if options.mismatchedWorkerPrompt {
			strategyA.Prompt = id(artifact.KindFile, "foreign strategy prompt")
		}
		if options.mismatchedWorkerRecipe {
			strategyA.ModelRecipe = strategyModelRecipeDefinition.ID
		}
		if options.mismatchedWorkerPolicies {
			strategyA.Policies = []artifact.ID{id(artifact.KindProfile, "foreign strategy policy")}
		}
		strategyA, err = strategyCodec.New(strategyA)
		if err != nil {
			t.Fatal(err)
		}
	}
	strategyB := strategyA
	strategyB.ID = artifact.ID{}
	strategyB.Loop.MaxInvocations++
	strategyB, err = strategyCodec.New(strategyB)
	if err != nil {
		t.Fatal(err)
	}
	fixture := &strategyExperimentTestFixture{
		ctx: ctx, store: store, task: task, baseline: strings.Repeat("a", 40),
		strategies: []Strategy{strategyA, strategyB},
	}
	leases := make([]plan.WorkLease, len(fixture.strategies))
	worktreeRoot := t.TempDir()
	for index := range fixture.strategies {
		name := string(rune('a' + index))
		worktree := filepath.Join(worktreeRoot, "strategy-"+name)
		if err := os.MkdirAll(worktree, 0o755); err != nil {
			t.Fatal(err)
		}
		lease, err := plan.NewWorkLease(plan.WorkLease{
			Task: "strategy-" + name, Worktree: filepath.ToSlash(worktree), Branch: "codex/strategy-" + name,
			Role: "experiment", TargetHead: fixture.baseline, ConflictsWith: []string{},
			Resources: plan.Resources{CPUThreads: 1, HostRAMGiB: 1},
			ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano),
		})
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(lease)
		if err != nil {
			t.Fatal(err)
		}
		leases[index], err = plan.RecordWorkLease(ctx, store, encoded)
		if err != nil || leases[index].ID != lease.ID {
			t.Fatalf("record lease = %s, want %s: %v", leases[index].ID, lease.ID, err)
		}
	}
	proposal := id(artifact.KindEvidence, "strategy causal proposal")
	motivation := id(artifact.KindEvidence, "strategy causal motivation")
	fixture.causalRoot, err = runrecord.NewCausalRoot(runrecord.TriggerControllerProposal, proposal, motivation)
	if err != nil {
		t.Fatal(err)
	}

	gateRecipe := id(artifact.KindRecipe, "strategy gate recipe")
	environmentRecord, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "strategy-fixture", OS: "test", Arch: "test", Device: "cpu",
		Backend: "fixture", Driver: "fixture",
	})
	if err != nil {
		t.Fatal(err)
	}
	environment := environmentRecord.ID
	providerImplementation := id(artifact.KindFile, "strategy cost provider implementation")
	providerSchema := id(artifact.KindProfile, "strategy cost provider schema")
	provider, err := (runrecord.CapabilityIdentity{
		Implementation: providerImplementation,
		Release:        "fixture",
		Transport: runrecord.CapabilityTransport{
			Kind: runrecord.CapabilityTransportBuiltin, Protocol: "cost-units/fixture",
		},
		Schema:   providerSchema,
		Platform: runrecord.CapabilityPlatform{OS: environmentRecord.OS, Arch: environmentRecord.Arch},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 1, MaxOutputBytes: 1, MaxConcurrent: 1, CPUThreads: 1, HostBytes: 1,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	fixture.provider = provider.ID
	evaluationRecipeDefinition, err := modelrecipe.CapabilityDefinition(recipe.TaskTabular, model)
	if err != nil {
		t.Fatal(err)
	}
	evaluationRecipe := evaluationRecipeDefinition.ID
	var modelDefinition artifact.ID
	static := []artifact.ID{
		prompt, strategyA.Prompt, policy, catalog, task.Budget, task.Verification[0], task.Acceptance[0].Verifier,
		model, recipeModel, proposal, motivation, gateRecipe, providerImplementation, providerSchema,
	}
	static = append(static, strategyA.Policies...)
	if options.descriptorOnlyWorker {
		static = append(static, worker.ID)
	}
	if options.descriptorOnlyModelRecipe {
		static = append(static, modelRecipe)
	}
	for index := range fixture.strategies {
		name := string(rune('a' + index))
		static = append(static, id(artifact.KindEvidence, "strategy request "+name))
	}
	descriptors := make([]artifact.Descriptor, 0, len(static))
	seen := map[artifact.ID]struct{}{}
	for _, value := range static {
		if _, duplicate := seen[value]; duplicate {
			continue
		}
		seen[value] = struct{}{}
		descriptors = append(descriptors, artifact.Descriptor{ID: value})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "loop/experiment/static", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	resolvedModelDefinition, err := modelrecipetest.PublishModelDefinition(
		ctx, store, "loop/experiment/model-definition", model,
	)
	if err != nil {
		t.Fatal(err)
	}
	modelDefinition = resolvedModelDefinition.Document.ID

	contents := make([]artifact.Content, 0, 6+3*len(fixture.strategies))
	lineage := slices.Clone(task.Lineage())
	if !options.descriptorOnlyModelRecipe {
		modelRecipeContent, err := modelRecipeDefinition.ArtifactContent()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, modelRecipeContent)
		lineage = append(lineage, artifact.DependencyLineage(modelRecipe, recipeModel)...)
		if strategyModelRecipeDefinition.ID != modelRecipeDefinition.ID {
			strategyModelRecipeContent, err := strategyModelRecipeDefinition.ArtifactContent()
			if err != nil {
				t.Fatal(err)
			}
			contents = append(contents, strategyModelRecipeContent)
			lineage = append(lineage, artifact.DependencyLineage(strategyModelRecipeDefinition.ID, recipeModel)...)
		}
	}
	evaluationRecipeContent, err := evaluationRecipeDefinition.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	contents = append(contents, evaluationRecipeContent)
	lineage = append(lineage, artifact.DependencyLineage(evaluationRecipe, model)...)
	if !options.descriptorOnlyWorker {
		workerContent, err := worker.ArtifactContent()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, workerContent)
		lineage = append(lineage, worker.Lineage()...)
	}
	environmentContent, err := environmentRecord.Content()
	if err != nil {
		t.Fatal(err)
	}
	providerContent, err := provider.Content()
	if err != nil {
		t.Fatal(err)
	}
	contents = append(contents, environmentContent, providerContent)
	lineage = append(lineage, provider.Lineage()...)
	taskContent, err := task.Content()
	if err != nil {
		t.Fatal(err)
	}
	contents = append(contents, taskContent)
	for index, strategy := range fixture.strategies {
		strategyContent, err := strategy.Content()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, strategyContent)
		lineage = append(lineage, strategy.Lineage()...)
		name := string(rune('a' + index))
		trace, err := runrecord.NewInteractionTrace(
			runrecord.Interaction{Recipe: strategy.ModelRecipe, Model: model}, id(artifact.KindEvidence, "strategy request "+name),
			[]runrecord.InteractionMessage{{Role: "user", Content: "perform the exact task"}}, nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		trace.TaskContract, trace.Terminal = task.ID, runrecord.OutcomeSucceeded
		trace, err = strategy.StampTrajectory(trace)
		if err != nil {
			t.Fatal(err)
		}
		var causal *runrecord.CausalContext
		if index == 0 {
			value := fixture.causalRoot
			causal = &value
		}
		wall := uint64(90 + index*10)
		gate, err := runrecord.NewGateRecord(
			gateRecipe, environment, fixture.baseline, runrecord.OutcomeSucceeded, "", wall,
			[]runrecord.GateStep{{Name: "tests", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: wall}},
		)
		if err != nil {
			t.Fatal(err)
		}
		attempt, err := strategy.StampAttempt(runrecord.AttemptRecord{
			PlanItem: "experiment", PlanStep: name, Result: gate.Result.ID,
			Recipe: gateRecipe, CodeCommit: fixture.baseline, Outcome: runrecord.OutcomeSucceeded,
			WallNS: wall, TaskContract: task.ID, WorkLease: leases[index].ID,
			Environment: environment, Trajectory: trace.ID,
			CostUnits: uint64(9 + index), Causal: causal,
		})
		if err != nil {
			t.Fatal(err)
		}
		traceContent, err := trace.Content()
		if err != nil {
			t.Fatal(err)
		}
		attemptContent, err := attempt.Content()
		if err != nil {
			t.Fatal(err)
		}
		gateContent, err := gate.Result.Content()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, traceContent, gateContent, attemptContent)
		lineage = append(lineage, trace.Lineage()...)
		lineage = append(lineage, gate.Result.Lineage()...)
		lineage = append(lineage, attempt.Lineage()...)
		fixture.trajectories = append(fixture.trajectories, trace)
		fixture.attempts = append(fixture.attempts, attempt)
	}
	documentBatch, err := artifact.NewDocumentBatch("loop/experiment/authorities", contents, lineage, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, documentBatch); err != nil {
		t.Fatal(err)
	}

	for index := range fixture.strategies {
		name := string(rune('a' + index))
		storedLease := leases[index]
		wall, peak, cost := uint64(90+index*10), uint64(90+index*10), uint64(9+index)
		work := runrecord.InteractionWork{SemanticTransitions: 1, ToolCalls: uint64(1 + index), Retries: uint64(index)}
		stream := fixture.publishObservation(t, "agent-"+name, runrecord.ResourceFitness{
			Scope: runrecord.ResourceScope{
				Surface: runrecord.SurfaceAgent, Model: model, Hardware: environment, Provider: fixture.provider, Workload: task.ID,
				Attempt: fixture.attempts[index].ID,
			},
			Measures: []runrecord.ResourceMeasure{
				{Metric: runrecord.ResourceWallNS, Value: wall},
				{Metric: runrecord.ResourcePeakHostBytes, Value: peak},
				{Metric: runrecord.ResourcePeakDeviceBytes, Value: 0},
				{Metric: runrecord.ResourceCostUnits, Value: cost},
			},
			Interactions: &work,
		}, wall)
		fixture.streams = append(fixture.streams, stream)
		quality := float64(2 - index)
		evidence, evaluationStream := fixture.publishEvaluation(
			t, "strategy-"+name, fixture.trajectories[index], quality,
			model, modelDefinition, evaluationRecipe, environment,
		)
		fixture.evidence = append(fixture.evidence, evidence)
		fixture.evalStreams = append(fixture.evalStreams, evaluationStream)
		fixture.candidates = append(fixture.candidates, StrategyExperimentCandidate{
			Strategy: fixture.strategies[index].ID, Lease: storedLease.ID,
			Attempt: fixture.attempts[index].ID, EvaluationEvidence: evidence.ID,
		})
	}
	return fixture
}

func (fixture *strategyExperimentTestFixture) publishFitness(t *testing.T, baseline, candidate int) evaluation.ImprovementFitness {
	t.Helper()
	requiredMetrics := []runrecord.ResourceMetric{
		runrecord.ResourceWallNS, runrecord.ResourcePeakHostBytes,
		runrecord.ResourcePeakDeviceBytes, runrecord.ResourceCostUnits,
	}
	resources, err := runrecord.CompareResourceFitness(fixture.ctx, fixture.store, []runrecord.ResourceFitnessLane{
		{
			Name: runrecord.ResourceLaneAgent, RequiredMetrics: requiredMetrics, RequireInteractions: true,
			Baseline: fixture.streams[baseline], Candidate: fixture.streams[candidate],
		},
		{
			Name: runrecord.ResourceLaneEvaluation, RequiredMetrics: requiredMetrics, RequireInteractions: true,
			Baseline: fixture.evalStreams[baseline], Candidate: fixture.evalStreams[candidate],
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	resourceBatch, err := resources.VerifiedBatch(fixture.ctx, fixture.store, "loop/experiment/resources/"+resources.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(fixture.ctx, fixture.store, resourceBatch); err != nil {
		t.Fatal(err)
	}
	requirements := []evaluation.CoverageRequirement{
		{Axis: evaluation.CoverageMeasurement, ExpectedSamples: 1, Metrics: []runrecord.ResourceMetric{
			runrecord.ResourceWallNS, runrecord.ResourcePeakHostBytes,
			runrecord.ResourcePeakDeviceBytes, runrecord.ResourceCostUnits,
		}},
		{Axis: evaluation.CoverageHardware, ExpectedSamples: 1, Metrics: []runrecord.ResourceMetric{
			runrecord.ResourcePeakHostBytes, runrecord.ResourcePeakDeviceBytes,
		}},
		{Axis: evaluation.CoverageEvaluation},
	}
	baselineEvidence, candidateEvidence := fixture.evidence[baseline], fixture.evidence[candidate]
	coverage, err := evaluation.NewEvidenceCoverageQuery(
		[]evaluation.CoverageUnit{
			{Attempt: baselineEvidence.Run, Terminal: baselineEvidence.Run,
				EvaluationEvidence: []artifact.ID{baselineEvidence.ID}, Required: requirements},
			{Attempt: candidateEvidence.Run, Terminal: candidateEvidence.Run,
				EvaluationEvidence: []artifact.ID{candidateEvidence.ID}, Required: requirements},
		},
		[]evaluation.CoveragePair{{Baseline: baselineEvidence.Run, Candidate: candidateEvidence.Run}},
		evaluation.CoverageBounds{
			MaxFacts: 16,
			Observation: runrecord.ObservationStreamBounds{
				MaxChunks: 8, MaxRawBytes: artifact.MaxContentBytes,
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	fitness, err := evaluation.PublishImprovementFitness(fixture.ctx, fixture.store, evaluation.ImprovementFitnessRequest{
		BaselineAttempt: fixture.attempts[baseline].ID, CandidateAttempt: fixture.attempts[candidate].ID,
		BaselineEvidence: baselineEvidence.ID, CandidateEvidence: candidateEvidence.ID,
		Resources: resources.ID, Coverage: coverage,
	})
	if err != nil {
		t.Fatal(err)
	}
	return fitness
}

func (fixture *strategyExperimentTestFixture) publishEvaluation(
	t *testing.T,
	name string,
	trajectory runrecord.InteractionTrace,
	quality float64,
	model, modelDefinition, evaluationRecipe, environment artifact.ID,
) (evaluation.EvaluationEvidence, runrecord.ObservationStream) {
	t.Helper()
	exact, err := evaluation.CompileExact(evaluation.ExactSuite{
		Schema: "loop-strategy/v1", Source: "loop strategy fixture",
		Cases: []evaluation.ExactCase{{
			Name: "exact", Prompt: "go", MaxTokens: 2, Text: "ok", PromptTokens: 2, GeneratedTokens: 2,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	base, err := evaluation.BindExact(exact, evaluation.ExactAuthorities{
		ModelDefinition: modelDefinition, RuntimeRecipe: evaluationRecipe,
		CodeCommit: fixture.baseline, Environment: environment,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleIsolated},
	})
	if err != nil {
		t.Fatal(err)
	}
	evaluationPlan, err := evaluation.BindAgentTrajectoryPlan(base, []artifact.ID{trajectory.ID}, nil)
	if err != nil {
		t.Fatal(err)
	}
	report, err := evaluation.EvaluateExactSharded(
		fixture.ctx, fixture.store, strategyExactGenerator{}, exact, evaluationPlan, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	metric := runrecord.Metric{Name: "quality", Value: quality, Direction: runrecord.DirectionMaximize}
	policy := evaluation.AcceptancePolicy{
		Version: artifact.InitialDocumentVersion,
		Metrics: []evaluation.MetricContract{{Name: metric.Name, Direction: metric.Direction}},
	}
	policy.ID, err = artifact.JSONID(artifact.KindProfile, policy)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := evaluation.NewEvaluator(evaluationPlan.Identity(), policy)
	if err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewBoundRun(
		evaluationRecipe, runrecord.OutcomeSucceeded, []artifact.ID{evaluationPlan.Identity()}, []artifact.ID{report}, "",
		fixture.baseline, environment, 10, []runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: 10}},
	)
	if err != nil {
		t.Fatal(err)
	}
	evaluationRecord, err := runrecord.NewEvaluation(evaluationRecipe, run.ID, evaluationPlan.Dataset(), []runrecord.Metric{metric})
	if err != nil {
		t.Fatal(err)
	}
	work := runrecord.InteractionWork{SemanticTransitions: 1}
	fitness, err := runrecord.NewResourceFitness(runrecord.ResourceFitness{
		Scope: runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation, Model: model, Hardware: environment, Provider: fixture.provider,
			Workload: evaluationPlan.Identity(), Attempt: run.ID,
		},
		Measures: []runrecord.ResourceMeasure{
			{Metric: runrecord.ResourceWallNS, Value: 10},
			{Metric: runrecord.ResourcePeakHostBytes, Value: 10},
			{Metric: runrecord.ResourcePeakDeviceBytes, Value: 0},
			{Metric: runrecord.ResourceCostUnits, Value: 1},
		},
		Interactions: &work,
	})
	if err != nil {
		t.Fatal(err)
	}
	chunk, err := runrecord.NewInitialObservationChunk(fitness, runrecord.ObservationSampleHardware, 10)
	if err != nil {
		t.Fatal(err)
	}
	runBatch, err := run.Batch("loop/experiment/evaluation/run/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.BindObservationChunk(fixture.ctx, fixture.store, &runBatch, chunk); err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(fixture.ctx, fixture.store, runBatch); err != nil {
		t.Fatal(err)
	}
	evaluationBatch, err := evaluationRecord.Batch("loop/experiment/evaluation/record/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(fixture.ctx, fixture.store, evaluationBatch); err != nil {
		t.Fatal(err)
	}
	evidence, err := evaluation.PublishEvaluationEvidence(
		fixture.ctx, fixture.store, evaluationPlan, policy, evaluator, report, run, evaluationRecord,
	)
	if err != nil {
		t.Fatal(err)
	}
	stream, found, err := runrecord.LoadObservationStream(
		fixture.ctx, fixture.store, run.ID,
		runrecord.ObservationStreamBounds{MaxChunks: 4, MaxRawBytes: artifact.MaxContentBytes},
	)
	if err != nil || !found {
		t.Fatalf("load evaluation observation stream: found=%v err=%v", found, err)
	}
	return evidence, stream
}

func (fixture *strategyExperimentTestFixture) publishObservation(
	t *testing.T,
	name string,
	value runrecord.ResourceFitness,
	elapsed uint64,
) runrecord.ObservationStream {
	t.Helper()
	fitness, err := runrecord.NewResourceFitness(value)
	if err != nil {
		t.Fatal(err)
	}
	chunk, err := runrecord.NewInitialObservationChunk(fitness, runrecord.ObservationSampleHardware, elapsed)
	if err != nil {
		t.Fatal(err)
	}
	batch := artifact.Batch{Key: "loop/experiment/observation/" + name}
	if _, err := runrecord.BindObservationChunk(fixture.ctx, fixture.store, &batch, chunk); err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(fixture.ctx, fixture.store, batch); err != nil {
		t.Fatal(err)
	}
	stream, found, err := runrecord.LoadObservationStream(
		fixture.ctx, fixture.store, fitness.Scope.Attempt,
		runrecord.ObservationStreamBounds{MaxChunks: 4, MaxRawBytes: artifact.MaxContentBytes},
	)
	if err != nil || !found {
		t.Fatalf("load observation stream: found=%v err=%v", found, err)
	}
	return stream
}

func (fixture *strategyExperimentTestFixture) publishDocument(
	t *testing.T,
	key string,
	content func() (artifact.Content, error),
	lineage []artifact.Lineage,
) {
	t.Helper()
	document, err := content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(key, []artifact.Content{document}, lineage, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(fixture.ctx, fixture.store, batch); err != nil {
		t.Fatal(err)
	}
}

func (fixture *strategyExperimentTestFixture) candidateByStrategy(t *testing.T, strategy artifact.ID) StrategyExperimentCandidate {
	t.Helper()
	for _, candidate := range fixture.candidates {
		if candidate.Strategy == strategy {
			return candidate
		}
	}
	t.Fatalf("strategy candidate %s is absent", strategy)
	return StrategyExperimentCandidate{}
}

func (fixture *strategyExperimentTestFixture) attemptByID(t *testing.T, id artifact.ID) runrecord.AttemptRecord {
	t.Helper()
	for _, attempt := range fixture.attempts {
		if attempt.ID == id {
			return attempt
		}
	}
	t.Fatalf("attempt %s is absent", id)
	return runrecord.AttemptRecord{}
}

type strategyExactGenerator struct{}

func (strategyExactGenerator) Generate(
	_ context.Context,
	_ string,
	options inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: 2})
	for index, piece := range []string{"o", "k"} {
		if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(index), Piece: piece}); err != nil {
			return nil, "", err
		}
	}
	return make([]tokenizer.TokenID, 4), "", nil
}
