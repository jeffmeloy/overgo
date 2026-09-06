package evaluation

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestImprovementFitnessRejectsTransferredCost(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	id := func(kind artifact.Kind, label string) artifact.ID {
		t.Helper()
		return testutil.ArtifactID(t, kind, "improvement fitness "+label)
	}
	exact, fixturePlan := ledgerFixture(t, "improvement fitness environment")
	model := id(artifact.KindModel, "model")
	prompt := id(artifact.KindFile, "agent prompt")
	policy := id(artifact.KindProfile, "agent policy")
	budget := id(artifact.KindEvidence, "budget")
	verifier := id(artifact.KindRecipe, "verifier")
	verification := id(artifact.KindEvidence, "verification")
	strategies := []artifact.ID{id(artifact.KindProfile, "baseline strategy"), id(artifact.KindProfile, "candidate strategy")}
	requests := []artifact.ID{id(artifact.KindEvidence, "baseline request"), id(artifact.KindEvidence, "candidate request")}
	dependencies := []artifact.ID{
		model, prompt, policy, budget, verifier, verification,
	}
	dependencies = append(dependencies, strategies...)
	dependencies = append(dependencies, requests...)
	descriptors := make([]artifact.Descriptor, len(dependencies))
	for index, dependency := range dependencies {
		descriptors[index] = artifact.Descriptor{ID: dependency}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "improvement/authorities", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	publishRecipe := func(name string, task recipe.Task) recipe.Definition {
		t.Helper()
		definition, err := modelrecipe.CapabilityDefinition(task, model)
		if err != nil {
			t.Fatal(err)
		}
		content, err := definition.ArtifactContent()
		if err != nil {
			t.Fatal(err)
		}
		parents := make([]artifact.ID, len(definition.Dependencies))
		for index := range definition.Dependencies {
			parents[index] = definition.Dependencies[index].Artifact
		}
		commitFitnessDocument(t, ctx, store, "improvement/recipe/"+name, content,
			artifact.DependencyLineage(definition.ID, parents...))
		return definition
	}
	strategyRecipe := publishRecipe("strategy", recipe.TaskGeneration)
	evaluationRecipe := publishRecipe("evaluation", recipe.TaskForecast)
	gateRecipe := publishRecipe("gate", recipe.TaskTabular)
	resolvedModelDefinition, err := modelrecipetest.PublishModelDefinition(
		ctx, store, "improvement/model-definition", model,
	)
	if err != nil {
		t.Fatal(err)
	}

	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "fitness-test", OS: "test", Arch: "test", Device: "cpu",
		Backend: "test", Driver: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	environmentBatch, err := environment.Batch("improvement/environment")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, environmentBatch); err != nil {
		t.Fatal(err)
	}
	basePlan, err := BindExact(exact, ExactAuthorities{
		ModelDefinition: resolvedModelDefinition.Document.ID,
		RuntimeRecipe:   evaluationRecipe.ID,
		CodeCommit:      fixturePlan.body.CodeCommit,
		Environment:     environment.ID,
		Execution:       ExecutionPolicy{Lifecycle: LifecycleIsolated},
	})
	if err != nil {
		t.Fatal(err)
	}
	providerImplementation := id(artifact.KindFile, "provider implementation")
	providerSchema := id(artifact.KindProfile, "provider schema")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "improvement/provider-parents",
		Artifacts: []artifact.Descriptor{
			{ID: providerImplementation},
			{ID: providerSchema},
		},
	}); err != nil {
		t.Fatal(err)
	}
	provider, err := (runrecord.CapabilityIdentity{
		Implementation: providerImplementation,
		Release:        "1.0.0",
		Transport: runrecord.CapabilityTransport{
			Kind: runrecord.CapabilityTransportBuiltin, Protocol: "cost-units/1",
		},
		Schema:   providerSchema,
		Platform: runrecord.CapabilityPlatform{OS: "test", Arch: "test"},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 1, MaxOutputBytes: 1, MaxConcurrent: 1, CPUThreads: 1, HostBytes: 1,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	providerContent, err := provider.Content()
	if err != nil {
		t.Fatal(err)
	}
	commitFitnessDocument(t, ctx, store, "improvement/provider", providerContent, provider.Lineage())

	manual, err := agenttool.NewManual(agenttool.Manual{
		Name: "measure", Description: "Records one exact tool measurement.", Effect: agenttool.EffectInspection,
		Transport:  agenttool.Transport{Kind: agenttool.TransportBuiltin, Protocol: "cost-units/1"},
		Capability: provider.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{manual}); err != nil {
		t.Fatal(err)
	}
	agent, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: "fitness-agent", Prompt: prompt, ModelRecipe: strategyRecipe.ID,
		ToolManuals: []artifact.ID{manual.ID}, Policies: []artifact.ID{policy},
	})
	if err != nil {
		t.Fatal(err)
	}
	agentContent, err := agent.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	commitFitnessDocument(t, ctx, store, "improvement/agent", agentContent, agent.Lineage())
	task, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: agent.ID, Objective: "Prove one exact improvement.", Scope: []string{"internal/evaluation"},
		AllowedEffects: []string{"workspace mutation"},
		Acceptance:     []recipe.AcceptanceCriterion{{Name: "tests", Scope: "internal/evaluation", Verifier: verifier}},
		Verification:   []artifact.ID{verification}, Budget: budget, PauseConditions: []string{"missing evidence"},
	})
	if err != nil {
		t.Fatal(err)
	}
	taskContent, err := task.Content()
	if err != nil {
		t.Fatal(err)
	}
	commitFitnessDocument(t, ctx, store, "improvement/task", taskContent, task.Lineage())

	type arm struct {
		name       string
		strategy   artifact.ID
		request    artifact.ID
		wall, cost uint64
		peakHost   uint64
		peakDevice uint64
		toolCalls  uint64
		trajectory runrecord.InteractionTrace
		attempt    runrecord.AttemptRecord
		plan       Plan
		evidence   EvaluationEvidence
		run        runrecord.Run
	}
	arms := []arm{
		{name: "baseline", strategy: strategies[0], request: requests[0], wall: 100, cost: 10, peakHost: 1_000, peakDevice: 500, toolCalls: 1},
		{name: "candidate", strategy: strategies[1], request: requests[1], wall: 90, cost: 8, peakHost: 900, peakDevice: 450},
	}
	for index := range arms {
		current := &arms[index]
		messages := []runrecord.InteractionMessage{{Role: "assistant", Content: "done"}}
		if current.toolCalls != 0 {
			messages = []runrecord.InteractionMessage{
				{Role: "assistant", ToolCalls: []runrecord.InteractionToolCall{{
					ID: "measure-1", Type: "function", Name: manual.Name, Manual: manual.ID, Arguments: `{}`,
				}}},
				{Role: "tool", Content: "measured", ToolCallID: "measure-1"},
				{Role: "assistant", Content: "done"},
			}
		}
		trace, err := runrecord.NewInteractionTrace(
			runrecord.Interaction{Recipe: strategyRecipe.ID, Model: model}, current.request, messages, nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		trace.TaskContract, trace.Strategy, trace.Terminal = task.ID, current.strategy, runrecord.OutcomeSucceeded
		if current.toolCalls != 0 {
			trace.ToolManuals = []artifact.ID{manual.ID}
		}
		current.trajectory, err = runrecord.NewAgentTrajectory(trace)
		if err != nil {
			t.Fatal(err)
		}
		trajectoryContent, err := current.trajectory.Content()
		if err != nil {
			t.Fatal(err)
		}
		commitFitnessDocument(t, ctx, store, "improvement/trajectory/"+current.name,
			trajectoryContent, current.trajectory.Lineage())

		gate, err := runrecord.NewGateRecord(
			gateRecipe.ID, basePlan.body.Environment, basePlan.body.CodeCommit,
			runrecord.OutcomeSucceeded, "", current.wall,
			[]runrecord.GateStep{{Name: "tests", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: current.wall}},
		)
		if err != nil {
			t.Fatal(err)
		}
		gateBatch, err := gate.Batch("improvement/gate/" + current.name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, gateBatch); err != nil {
			t.Fatal(err)
		}
		current.attempt, err = runrecord.NewAttemptRecord(runrecord.AttemptRecord{
			PlanItem: "resource-coverage", PlanStep: "fitness-integration", Result: gate.Result.ID,
			Recipe: gateRecipe.ID, CodeCommit: basePlan.body.CodeCommit,
			Outcome: runrecord.OutcomeSucceeded, WallNS: current.wall, CostUnits: current.cost,
			StrategyID: current.strategy, TaskContract: task.ID, Environment: basePlan.body.Environment,
			Trajectory: current.trajectory.ID,
		})
		if err != nil {
			t.Fatal(err)
		}
		attemptContent, err := current.attempt.Content()
		if err != nil {
			t.Fatal(err)
		}
		commitFitnessDocument(t, ctx, store, "improvement/attempt/"+current.name,
			attemptContent, current.attempt.Lineage())
		current.plan, err = BindAgentTrajectoryPlan(basePlan, []artifact.ID{current.trajectory.ID}, nil)
		if err != nil {
			t.Fatal(err)
		}
		current.run, current.evidence = publishImprovementEvaluation(t, ctx, store, exact, current.plan, current.wall, current.name)
	}

	agentStreams := make([]runrecord.ObservationStream, 2)
	evaluationStreams := make([]runrecord.ObservationStream, 2)
	toolStreams := make([]runrecord.ObservationStream, 2)
	for index := range arms {
		current := arms[index]
		agentStreams[index] = publishImprovementStream(t, ctx, store, current.name+"-agent", runrecord.ResourceScope{
			Surface: runrecord.SurfaceAgent, Model: model, Hardware: basePlan.body.Environment,
			Provider: provider.ID, Workload: task.ID, Attempt: current.attempt.ID,
		}, current.wall, current.cost, current.peakHost, current.peakDevice)
		evaluationStreams[index] = publishImprovementStream(t, ctx, store, current.name+"-evaluation", runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation, Model: model, Hardware: basePlan.body.Environment,
			Provider: provider.ID, Workload: current.plan.Identity(), Attempt: current.run.ID,
		}, current.wall, current.cost, current.peakHost, current.peakDevice)
		toolStreams[index] = publishImprovementStreamWithWork(t, ctx, store, current.name+"-tool", runrecord.ResourceScope{
			Surface: runrecord.SurfaceTool, Model: model, Hardware: basePlan.body.Environment,
			Provider: provider.ID, Workload: task.ID, Attempt: current.trajectory.ID,
		}, current.wall, current.cost, current.peakHost, current.peakDevice, runrecord.InteractionWork{
			SemanticTransitions: 1, ToolCalls: current.toolCalls,
		})
	}
	lanes := []runrecord.ResourceFitnessLane{
		improvementLane(runrecord.ResourceLaneAgent, agentStreams[0], agentStreams[1]),
		improvementLane(runrecord.ResourceLaneEvaluation, evaluationStreams[0], evaluationStreams[1]),
		improvementLane(runrecord.ResourceLaneTool, toolStreams[0], toolStreams[1]),
	}
	resources := publishImprovementResources(t, ctx, store, "genuine", lanes)
	requirements := []CoverageRequirement{
		{Axis: CoverageMeasurement, ExpectedSamples: 1, Metrics: slices.Clone(requiredImprovementResourceMetrics)},
		{Axis: CoverageHardware, ExpectedSamples: 1, Metrics: []runrecord.ResourceMetric{
			runrecord.ResourcePeakDeviceBytes, runrecord.ResourcePeakHostBytes,
		}},
		{Axis: CoverageEvaluation},
	}
	query, err := NewEvidenceCoverageQuery([]CoverageUnit{
		{Attempt: arms[0].run.ID, Terminal: arms[0].run.ID, EvaluationEvidence: []artifact.ID{arms[0].evidence.ID}, Required: requirements},
		{Attempt: arms[1].run.ID, Terminal: arms[1].run.ID, EvaluationEvidence: []artifact.ID{arms[1].evidence.ID}, Required: requirements},
	}, []CoveragePair{{Baseline: arms[0].run.ID, Candidate: arms[1].run.ID}}, coverageTestBounds(32, 2))
	if err != nil {
		t.Fatal(err)
	}
	request := ImprovementFitnessRequest{
		BaselineAttempt: arms[0].attempt.ID, CandidateAttempt: arms[1].attempt.ID,
		BaselineEvidence: arms[0].evidence.ID, CandidateEvidence: arms[1].evidence.ID,
		Coverage: query, Resources: resources.ID,
	}
	fitness, err := PublishImprovementFitness(ctx, store, request)
	if err != nil {
		t.Fatal(err)
	}
	stored, err := RequireImprovementFitness(ctx, store, fitness.ID)
	if err != nil || stored.ID != fitness.ID || !stored.Improved || !stored.Resource.Strict ||
		stored.Baseline.Attempt != arms[0].attempt.ID || stored.Candidate.Attempt != arms[1].attempt.ID {
		t.Fatalf("stored improvement fitness = (%+v, %v)", stored, err)
	}

	baselineAuthority, err := requireImprovementEndpoint(ctx, store, arms[0].attempt.ID, arms[0].evidence.ID)
	if err != nil {
		t.Fatal(err)
	}
	candidateAuthority, err := requireImprovementEndpoint(ctx, store, arms[1].attempt.ID, arms[1].evidence.ID)
	if err != nil {
		t.Fatal(err)
	}
	baselineTools, err := requireImprovementToolAuthority(ctx, store, baselineAuthority)
	if err != nil {
		t.Fatal(err)
	}
	candidateTools, err := requireImprovementToolAuthority(ctx, store, candidateAuthority)
	if err != nil {
		t.Fatal(err)
	}
	unadmittedManual, err := agenttool.NewManual(agenttool.Manual{
		Name: "foreign-measure", Description: "Records a foreign tool measurement.", Effect: agenttool.EffectInspection,
		Transport:  agenttool.Transport{Kind: agenttool.TransportBuiltin, Protocol: "cost-units/1"},
		Capability: provider.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{unadmittedManual}); err != nil {
		t.Fatal(err)
	}
	for _, refusal := range []struct {
		name   string
		mutate func(*improvementEndpointAuthority)
	}{
		{name: "missing manual", mutate: func(endpoint *improvementEndpointAuthority) {
			endpoint.trajectory.Events[0].Message.ToolCalls[0].Manual = artifact.ID{}
		}},
		{name: "manual name mismatch", mutate: func(endpoint *improvementEndpointAuthority) {
			endpoint.trajectory.Events[0].Message.ToolCalls[0].Name = "foreign"
		}},
		{name: "unadmitted manual", mutate: func(endpoint *improvementEndpointAuthority) {
			endpoint.trajectory.Events[0].Message.ToolCalls[0].Name = unadmittedManual.Name
			endpoint.trajectory.Events[0].Message.ToolCalls[0].Manual = unadmittedManual.ID
		}},
		{name: "arguments outside manual", mutate: func(endpoint *improvementEndpointAuthority) {
			endpoint.trajectory.Events[0].Message.ToolCalls[0].Arguments = `{"foreign":true}`
		}},
		{name: "missing result", mutate: func(endpoint *improvementEndpointAuthority) {
			endpoint.trajectory.Events = append(endpoint.trajectory.Events[:1], endpoint.trajectory.Events[2:]...)
		}},
		{name: "foreign result", mutate: func(endpoint *improvementEndpointAuthority) {
			endpoint.trajectory.Events[1].Message.ToolCallID = "foreign"
		}},
		{name: "duplicate result", mutate: func(endpoint *improvementEndpointAuthority) {
			endpoint.trajectory.Events = append(endpoint.trajectory.Events, endpoint.trajectory.Events[1])
		}},
	} {
		mutated := baselineAuthority
		mutated.trajectory.Events = slices.Clone(baselineAuthority.trajectory.Events)
		mutated.trajectory.Events[0].Message.ToolCalls = slices.Clone(
			baselineAuthority.trajectory.Events[0].Message.ToolCalls,
		)
		refusal.mutate(&mutated)
		if _, err := requireImprovementToolAuthority(ctx, store, mutated); err == nil {
			t.Fatalf("trajectory accepted %s", refusal.name)
		}
	}
	identityOnly := candidateAuthority
	identityOnly.trajectory.ToolManuals = []artifact.ID{manual.ID}
	if _, err := requireImprovementToolAuthority(ctx, store, identityOnly); err == nil {
		t.Fatal("trajectory accepted a tool identity without an exact call")
	}

	for _, counter := range []string{"calls", "failures"} {
		mismatched := resources
		mismatched.Lanes = slices.Clone(resources.Lanes)
		toolIndex := slices.IndexFunc(mismatched.Lanes, func(lane runrecord.ResourceFitnessLane) bool {
			return lane.Name == runrecord.ResourceLaneTool
		})
		work := *mismatched.Lanes[toolIndex].Baseline.Aggregate.Interactions
		if counter == "calls" {
			work.ToolCalls++
		} else {
			work.Failures++
		}
		mismatched.Lanes[toolIndex].Baseline.Aggregate.Interactions = &work
		if err := requireImprovementToolLane(
			ctx, store, mismatched, baselineAuthority, candidateAuthority, baselineTools, candidateTools,
		); err == nil {
			t.Fatalf("tool lane accepted mismatched %s", counter)
		}
	}

	foreignProvider, err := (runrecord.CapabilityIdentity{
		Implementation: providerImplementation, Release: "2.0.0",
		Transport: runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportBuiltin, Protocol: "cost-units/1"},
		Schema:    providerSchema, Platform: runrecord.CapabilityPlatform{OS: "foreign", Arch: "test"},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 1, MaxOutputBytes: 1, MaxConcurrent: 1, CPUThreads: 1, HostBytes: 1,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	foreignProviderContent, err := foreignProvider.Content()
	if err != nil {
		t.Fatal(err)
	}
	commitFitnessDocument(t, ctx, store, "improvement/foreign-provider", foreignProviderContent, foreignProvider.Lineage())
	foreignProviderStream := agentStreams[0]
	foreignProviderStream.Scope.Provider = foreignProvider.ID
	if err := requireImprovementProvider(ctx, store, foreignProviderStream, environment); err == nil {
		t.Fatal("resource stream accepted a provider for a foreign platform")
	}

	descriptorRecipe := id(artifact.KindRecipe, "descriptor recipe")
	descriptorEnvironment := id(artifact.KindEvidence, "descriptor environment")
	descriptorDataset := id(artifact.KindDataset, "descriptor dataset")
	descriptorSplit := id(artifact.KindDatasetShard, "descriptor split")
	descriptorModelDefinition := id(artifact.KindModelDefinition, "descriptor model definition")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "improvement/descriptor-only-authorities", Artifacts: []artifact.Descriptor{
		{ID: descriptorRecipe}, {ID: descriptorEnvironment}, {ID: descriptorDataset}, {ID: descriptorSplit},
		{ID: descriptorModelDefinition},
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := recipe.RequireDefinition(ctx, store, descriptorRecipe); err == nil {
		t.Fatal("improvement recipe accepted descriptor-only authority")
	}
	if _, err := runrecord.RequireEnvironment(ctx, store, descriptorEnvironment); err == nil {
		t.Fatal("improvement environment accepted descriptor-only authority")
	}
	if err := requireImprovementDatasetSplit(ctx, store, descriptorDataset, descriptorSplit); err == nil {
		t.Fatal("improvement dataset accepted descriptor-only authorities")
	}
	if err := modelrecipe.RequireModelDefinitionBinding(ctx, store, descriptorModelDefinition, model, descriptorRecipe); err == nil {
		t.Fatal("improvement accepted a descriptor-only model definition")
	}
	foreignModel := id(artifact.KindModel, "foreign model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "improvement/foreign-model", Artifacts: []artifact.Descriptor{{ID: foreignModel}},
	}); err != nil {
		t.Fatal(err)
	}
	foreignDefinition, err := modelrecipetest.PublishModelDefinition(
		ctx, store, "improvement/foreign-model-definition", foreignModel,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.RequireModelDefinitionBinding(ctx, store, foreignDefinition.Document.ID, model, descriptorRecipe); err == nil {
		t.Fatal("improvement accepted a model definition bound to a foreign model")
	}

	foreignCases := slices.Clone(exact.suite.Cases)
	foreignCases[0].Name += "-foreign"
	foreignDatasetContent, err := authorityContent(exactDatasetContract, foreignCases)
	if err != nil {
		t.Fatal(err)
	}
	foreignSplitContent, err := authorityContent(exactSplitContract, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: foreignDatasetContent.Descriptor.ID})
	if err != nil {
		t.Fatal(err)
	}
	foreignDatasetBatch, err := artifact.NewDocumentBatch(
		"improvement/foreign-dataset", []artifact.Content{foreignDatasetContent, foreignSplitContent},
		artifact.DependencyLineage(foreignSplitContent.Descriptor.ID, foreignDatasetContent.Descriptor.ID), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, foreignDatasetBatch); err != nil {
		t.Fatal(err)
	}
	if err := requireImprovementDatasetSplit(ctx, store, exact.dataset, foreignSplitContent.Descriptor.ID); err == nil {
		t.Fatal("improvement split accepted a different native dataset")
	}
	lineageStore, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = lineageStore.Close() })
	nativeContents, err := exact.contents()
	if err != nil {
		t.Fatal(err)
	}
	withoutLineage, err := artifact.NewDocumentBatch("improvement/native-without-lineage", nativeContents, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, lineageStore, withoutLineage); err != nil {
		t.Fatal(err)
	}
	if err := requireImprovementDatasetSplit(ctx, lineageStore, exact.dataset, exact.split); err == nil {
		t.Fatal("improvement dataset accepted a valid native split without dataset lineage")
	}

	requireForgedFitnessRejected := func(name string, mutate func(*ImprovementFitness)) {
		t.Helper()
		forged := cloneImprovementFitness(fitness)
		journalHead, journalSequence := store.Head()
		forged.Coverage.Head = journalHead
		forged.Coverage.Sequence = journalSequence
		mutate(&forged)
		forged.ID = artifact.ID{}
		forged, err = improvementFitnessCodec.New(forged)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := improvementFitnessCodec.Batch(
			"evaluation/improvement-fitness/"+forged.ID.String(), forged, forged.Lineage(), nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		batch.ExpectedHead = &journalHead
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		if _, err := RequireImprovementFitness(ctx, store, forged.ID); err == nil {
			t.Fatalf("stored fitness accepted forged projection %s", name)
		}
	}
	for _, refusal := range []struct {
		name   string
		mutate func(*ImprovementFitness)
	}{
		{name: "head", mutate: func(value *ImprovementFitness) { value.Coverage.Head[0] ^= 0xff }},
		{name: "sequence", mutate: func(value *ImprovementFitness) { value.Coverage.Sequence++ }},
		{name: "failure classifier version", mutate: func(value *ImprovementFitness) {
			value.Coverage.FailureClassifierVersion++
		}},
		{name: "causality projection version", mutate: func(value *ImprovementFitness) {
			value.Coverage.CausalityProjectionVersion++
		}},
	} {
		requireForgedFitnessRejected(refusal.name, refusal.mutate)
	}

	// The proof must be the exact commit immediately following the projection,
	// not merely a later document that cites an otherwise valid journal head.
	forged := cloneImprovementFitness(fitness)
	journalHead, journalSequence := store.Head()
	forged.Coverage.Head = journalHead
	forged.Coverage.Sequence = journalSequence
	forged.ID = artifact.ID{}
	forged, err = improvementFitnessCodec.New(forged)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := improvementFitnessCodec.Batch(
		"improvement/forged-projection/detached", forged, forged.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch.ExpectedHead = &journalHead
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireImprovementFitness(ctx, store, forged.ID); err == nil {
		t.Fatal("stored fitness accepted a detached publication commit")
	}

	// A canonical key at S+1 is not sufficient: that exact commit must first
	// introduce the fitness document. Otherwise an unrelated mutation can
	// reserve the expected key and attach the proof later under another key.
	forged = cloneImprovementFitness(fitness)
	journalHead, journalSequence = store.Head()
	forged.Coverage.Head = journalHead
	forged.Coverage.Sequence = journalSequence
	forged.ID = artifact.ID{}
	forged, err = improvementFitnessCodec.New(forged)
	if err != nil {
		t.Fatal(err)
	}
	forgedContent, err := improvementFitnessCodec.Content(forged)
	if err != nil {
		t.Fatal(err)
	}
	reserved := artifact.Batch{
		Key:          improvementFitnessCommitKey(forged.ID),
		ExpectedHead: &journalHead,
		Artifacts:    []artifact.Descriptor{forgedContent.Descriptor},
	}
	if _, err := artifact.CommitBatch(ctx, store, reserved); err != nil {
		t.Fatal(err)
	}
	lateBatch, err := improvementFitnessCodec.Batch(
		"improvement/forged-projection/late-content", forged, forged.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, lateBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireImprovementFitness(ctx, store, forged.ID); err == nil {
		t.Fatal("stored fitness accepted a publication key that did not introduce its content")
	}

	// The stored projection is only a claim. A structurally canonical document
	// with a larger pinned denominator must be rejected when Require replays the
	// exact streams and recomputes coverage instead of trusting axis counters.
	forged = cloneImprovementFitness(fitness)
	for unitIndex := range forged.CoverageQuery.Units {
		for requirementIndex := range forged.CoverageQuery.Units[unitIndex].Required {
			requirement := &forged.CoverageQuery.Units[unitIndex].Required[requirementIndex]
			if requirement.Axis == CoverageHardware {
				requirement.ExpectedSamples++
			}
		}
	}
	forgedQuery, err := NewEvidenceCoverageQuery(
		forged.CoverageQuery.Units, forged.CoverageQuery.Pairs, forged.CoverageQuery.Bounds,
	)
	if err != nil {
		t.Fatal(err)
	}
	forged.CoverageQuery = forgedQuery
	forged.Coverage.Query = forgedQuery.ID
	journalHead, journalSequence = store.Head()
	forged.Coverage.Head = journalHead
	forged.Coverage.Sequence = journalSequence
	forged.ID = artifact.ID{}
	forged, err = improvementFitnessCodec.New(forged)
	if err != nil {
		t.Fatal(err)
	}
	forgedBatch, err := improvementFitnessCodec.Batch(
		"evaluation/improvement-fitness/"+forged.ID.String(), forged, forged.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	forgedBatch.ExpectedHead = &journalHead
	if _, err := artifact.CommitBatch(ctx, store, forgedBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireImprovementFitness(ctx, store, forged.ID); err == nil {
		t.Fatal("stored coverage counters concealed degraded exact hardware coverage")
	}

	// Moving saved agent cost into evaluation work is still a regression. Extend
	// both streams with the same second-sample protocol, then add cost only to
	// the candidate. The comparison must fail on cost rather than denominator
	// mismatch, while the historical proof remains reproducible.
	baselineEvaluationHead := evaluationStreams[0].SummaryIDs[len(evaluationStreams[0].SummaryIDs)-1]
	baselineContinuation, err := runrecord.NewObservationChunk(
		evaluationStreams[0].Scope, baselineEvaluationHead,
		[]runrecord.ObservationSample{{
			Ordinal: 2, ElapsedNS: arms[0].wall + 1, Kind: runrecord.ObservationSampleHardware,
			Measures: []runrecord.ResourceMeasure{{Metric: runrecord.ResourceCostUnits, Value: 0}},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	baselineContinuationSummary := publishCoverageChunk(t, ctx, store, "improvement-baseline-cost-protocol", baselineContinuation)
	extendedBaselineEvaluation, err := runrecord.RequireObservationStream(
		ctx, store, arms[0].run.ID, baselineContinuationSummary.ID,
		runrecord.ObservationStreamBounds{MaxChunks: 2, MaxRawBytes: artifact.MaxContentBytes},
	)
	if err != nil {
		t.Fatal(err)
	}
	candidateEvaluationHead := evaluationStreams[1].SummaryIDs[len(evaluationStreams[1].SummaryIDs)-1]
	transferredChunk, err := runrecord.NewObservationChunk(
		evaluationStreams[1].Scope, candidateEvaluationHead,
		[]runrecord.ObservationSample{{
			Ordinal: 2, ElapsedNS: arms[1].wall + 1, Kind: runrecord.ObservationSampleHardware,
			Measures: []runrecord.ResourceMeasure{{Metric: runrecord.ResourceCostUnits, Value: 3}},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	transferredSummary := publishCoverageChunk(t, ctx, store, "improvement-transferred-cost", transferredChunk)
	transferredEvaluation, err := runrecord.RequireObservationStream(
		ctx, store, arms[1].run.ID, transferredSummary.ID,
		runrecord.ObservationStreamBounds{MaxChunks: 2, MaxRawBytes: artifact.MaxContentBytes},
	)
	if err != nil {
		t.Fatal(err)
	}
	transferredLanes := slices.Clone(lanes)
	transferredLanes[1] = improvementLane(runrecord.ResourceLaneEvaluation, extendedBaselineEvaluation, transferredEvaluation)
	if _, err := runrecord.CompareResourceFitness(ctx, store, transferredLanes); err == nil {
		t.Fatal("candidate transferred saved agent cost into evaluation work")
	}
	if historical, err := RequireImprovementFitness(ctx, store, fitness.ID); err != nil || historical.ID != fitness.ID {
		t.Fatalf("historical improvement after observation alias advance = (%+v, %v)", historical, err)
	}

	// A foreign lane can carry a strict reduction but has no typed authority
	// connecting it to either strategy. It must not manufacture improvement.
	foreignAttempts := []artifact.ID{id(artifact.KindEvidence, "foreign baseline"), id(artifact.KindEvidence, "foreign candidate")}
	foreignWorkload := id(artifact.KindProfile, "foreign workload")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "improvement/foreign-authorities", Artifacts: []artifact.Descriptor{
		{ID: foreignAttempts[0]}, {ID: foreignAttempts[1]}, {ID: foreignWorkload},
	}}); err != nil {
		t.Fatal(err)
	}
	foreignBaseline := publishImprovementStream(t, ctx, store, "foreign-baseline", runrecord.ResourceScope{
		Surface: runrecord.SurfaceWorkflow, Hardware: basePlan.body.Environment, Provider: provider.ID,
		Workload: foreignWorkload, Attempt: foreignAttempts[0],
	}, 20, 2, 200, 100)
	foreignCandidate := publishImprovementStream(t, ctx, store, "foreign-candidate", runrecord.ResourceScope{
		Surface: runrecord.SurfaceWorkflow, Hardware: basePlan.body.Environment, Provider: provider.ID,
		Workload: foreignWorkload, Attempt: foreignAttempts[1],
	}, 10, 1, 100, 50)
	foreignResources := publishImprovementResources(t, ctx, store, "foreign", append(lanes,
		improvementLane("workflow", foreignBaseline, foreignCandidate)))
	request.Resources = foreignResources.ID
	if _, err := PublishImprovementFitness(ctx, store, request); err == nil {
		t.Fatal("unrelated strict resource lane manufactured an improvement")
	}
}

func publishImprovementEvaluation(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
	exact ExactPlan,
	plan Plan,
	measured uint64,
	name string,
) (runrecord.Run, EvaluationEvidence) {
	t.Helper()
	report, err := EvaluateExactSharded(ctx, store, exactGenerator{pieces: []string{"o", "k"}}, exact, plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	metrics := []runrecord.Metric{{Name: exactMetricName, Value: 1, Direction: runrecord.DirectionMaximize}}
	policy, err := newAcceptancePolicy(metrics)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := NewEvaluator(plan.Identity(), policy)
	if err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewBoundRun(
		plan.body.RuntimeRecipe, runrecord.OutcomeSucceeded, []artifact.ID{plan.Identity()}, []artifact.ID{report}, "",
		plan.body.CodeCommit, plan.body.Environment, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := runrecord.NewEvaluation(plan.body.RuntimeRecipe, run.ID, plan.body.Dataset, metrics)
	if err != nil {
		t.Fatal(err)
	}
	ordered := []struct {
		name  string
		value interface {
			Batch(string) (artifact.Batch, error)
		}
	}{
		{name: "run", value: run},
		{name: "evaluation", value: evaluation},
	}
	for _, current := range ordered {
		batch, err := current.value.Batch("improvement/evaluation/" + name + "/" + current.name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	evidence, err := PublishEvaluationEvidence(ctx, store, plan, policy, evaluator, report, run, evaluation)
	if err != nil {
		t.Fatal(err)
	}
	return run, evidence
}

func publishImprovementStream(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
	name string,
	scope runrecord.ResourceScope,
	wall, cost, peakHost, peakDevice uint64,
) runrecord.ObservationStream {
	return publishImprovementStreamWithWork(t, ctx, store, name, scope, wall, cost, peakHost, peakDevice,
		runrecord.InteractionWork{SemanticTransitions: 1})
}

func publishImprovementStreamWithWork(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
	name string,
	scope runrecord.ResourceScope,
	wall, cost, peakHost, peakDevice uint64,
	work runrecord.InteractionWork,
) runrecord.ObservationStream {
	t.Helper()
	chunk, err := runrecord.NewObservationChunk(scope, artifact.ID{}, []runrecord.ObservationSample{{
		Ordinal: 1, ElapsedNS: wall, Kind: runrecord.ObservationSampleHardware,
		Measures: []runrecord.ResourceMeasure{
			{Metric: runrecord.ResourceWallNS, Value: wall},
			{Metric: runrecord.ResourceCostUnits, Value: cost},
			{Metric: runrecord.ResourcePeakHostBytes, Value: peakHost},
			{Metric: runrecord.ResourcePeakDeviceBytes, Value: peakDevice},
		},
		Interactions: &work,
	}})
	if err != nil {
		t.Fatal(err)
	}
	summary := publishCoverageChunk(t, ctx, store, "improvement-"+name, chunk)
	stream, err := runrecord.RequireObservationStream(ctx, store, scope.Attempt, summary.ID,
		runrecord.ObservationStreamBounds{MaxChunks: 1, MaxRawBytes: artifact.MaxContentBytes})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func improvementLane(name string, baseline, candidate runrecord.ObservationStream) runrecord.ResourceFitnessLane {
	return runrecord.ResourceFitnessLane{
		Name: name, RequiredMetrics: slices.Clone(requiredImprovementResourceMetrics), RequireInteractions: true,
		Baseline: baseline, Candidate: candidate,
	}
}

func publishImprovementResources(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
	name string,
	lanes []runrecord.ResourceFitnessLane,
) runrecord.ResourceFitnessComparison {
	t.Helper()
	comparison, err := runrecord.CompareResourceFitness(ctx, store, lanes)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := comparison.VerifiedBatch(ctx, store, "improvement/resources/"+name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	return comparison
}

func commitFitnessDocument(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
	key string,
	content artifact.Content,
	lineage []artifact.Lineage,
) {
	t.Helper()
	batch, err := artifact.NewDocumentBatch(key, []artifact.Content{content}, lineage, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
}
