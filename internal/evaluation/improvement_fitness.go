package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

const (
	// ImprovementFitnessMediaType identifies a persisted, reproducible
	// capability/quality/resource/coverage improvement proof.
	ImprovementFitnessMediaType = "application/vnd.overgo.improvement-fitness+json"
	// ImprovementFitnessSchema identifies the first improvement proof schema.
	ImprovementFitnessSchema = "overgo/improvement-fitness/v1"
	// ImprovementFitnessVersion is the current improvement proof version.
	ImprovementFitnessVersion uint16 = artifact.InitialDocumentVersion
)

// ImprovementFitnessEndpoint materializes the exact authority chain used for
// one side of an improvement comparison. Quality is copied from Evidence only
// after the complete evaluation bundle has been verified.
type ImprovementFitnessEndpoint struct {
	Strategy   artifact.ID        `json:"strategy"`
	Attempt    artifact.ID        `json:"attempt"`
	Trajectory artifact.ID        `json:"trajectory"`
	Evidence   artifact.ID        `json:"evidence"`
	Plan       artifact.ID        `json:"plan"`
	Run        artifact.ID        `json:"run"`
	Evaluation artifact.ID        `json:"evaluation"`
	Quality    []runrecord.Metric `json:"quality"`
}

// ImprovementFitnessDimension keeps non-regression distinct from a strict
// gain. Different dimensions are never scalarized into one score.
type ImprovementFitnessDimension struct {
	NoRegression bool `json:"no_regression"`
	Strict       bool `json:"strict"`
}

// ImprovementFitness is one immutable pairwise promotion proof. The complete
// coverage denominator and projection are embedded because CoverageQuery is a
// derived profile rather than an independently stored document.
type ImprovementFitness struct {
	Version       uint16                      `json:"version"`
	Baseline      ImprovementFitnessEndpoint  `json:"baseline"`
	Candidate     ImprovementFitnessEndpoint  `json:"candidate"`
	Resources     artifact.ID                 `json:"resources"`
	CoverageQuery CoverageQuery               `json:"coverage_query"`
	Coverage      CoverageProjection          `json:"coverage"`
	Capability    ImprovementFitnessDimension `json:"capability"`
	Quality       ImprovementFitnessDimension `json:"quality"`
	Resource      ImprovementFitnessDimension `json:"resource"`
	SourceChunks  []artifact.ID               `json:"source_chunks"`
	Improved      bool                        `json:"improved"`
	ID            artifact.ID                 `json:"-"`
}

// ImprovementFitnessRequest supplies only immutable authorities. Publication
// derives every verdict, stream head, metric vector, and source citation.
type ImprovementFitnessRequest struct {
	BaselineAttempt   artifact.ID
	CandidateAttempt  artifact.ID
	BaselineEvidence  artifact.ID
	CandidateEvidence artifact.ID
	Coverage          CoverageQuery
	Resources         artifact.ID
}

var improvementFitnessCodec = artifact.JSONDocumentCodec(
	"improvement fitness", artifact.KindEvidence,
	ImprovementFitnessMediaType, ImprovementFitnessSchema,
	canonicalizeImprovementFitness,
	func(value ImprovementFitness) artifact.ID { return value.ID },
	func(value *ImprovementFitness, id artifact.ID) { value.ID = id },
	cloneImprovementFitness,
)

var requiredImprovementResourceMetrics = []runrecord.ResourceMetric{
	runrecord.ResourceCostUnits,
	runrecord.ResourcePeakDeviceBytes,
	runrecord.ResourcePeakHostBytes,
	runrecord.ResourceWallNS,
}

var requiredImprovementCoverageAxes = [...]CoverageAxis{
	CoverageMeasurement,
	CoverageHardware,
	CoverageEvaluation,
}

type improvementEndpointAuthority struct {
	value       ImprovementFitnessEndpoint
	attempt     runrecord.AttemptRecord
	gate        runrecord.GateResult
	trajectory  runrecord.InteractionTrace
	evidence    EvaluationEvidence
	plan        Plan
	run         runrecord.Run
	task        recipe.AgentTaskContract
	agent       recipe.AgentDefinition
	environment runrecord.Environment
}

type improvementToolAuthority struct {
	calls      uint64
	failures   uint64
	capability artifact.ID
}

type improvementDatasetContractPair struct {
	dataset artifact.DocumentContract
	split   artifact.DocumentContract
}

var improvementDatasetContracts = [...]improvementDatasetContractPair{
	{dataset: exactDatasetContract, split: exactSplitContract},
	{dataset: generatedAnswerDatasetContract, split: generatedAnswerSplitContract},
	{dataset: instructionRulesDatasetContract, split: instructionRulesSplitContract},
	{dataset: multipleChoiceDatasetContract, split: multipleChoiceSplitContract},
	{dataset: probabilityMassDatasetContract, split: probabilityMassSplitContract},
	{dataset: sequenceScoringDatasetContract, split: sequenceScoringSplitContract},
	{dataset: structuredGeneratedDatasetContract, split: structuredGeneratedSplitContract},
}

type improvementAuthorities struct {
	baseline, candidate improvementEndpointAuthority
	resources           runrecord.ResourceFitnessComparison
	query               CoverageQuery
	agentLane           runrecord.ResourceFitnessLane
	evaluationLane      runrecord.ResourceFitnessLane
	capability          ImprovementFitnessDimension
	quality             ImprovementFitnessDimension
	resource            ImprovementFitnessDimension
}

type improvementFitnessReader interface {
	artifact.Reader
	Query(context.Context, overgodb.Query) (overgodb.QueryResult, error)
	ArtifactIntroduction(context.Context, artifact.ID) (overgodb.ArtifactIntroduction, bool, error)
}

// PublishImprovementFitness validates and atomically publishes one improvement
// proof against the exact journal head observed by its coverage projection.
func PublishImprovementFitness(
	ctx context.Context,
	store *overgodb.Store,
	request ImprovementFitnessRequest,
) (ImprovementFitness, error) {
	if ctx == nil || store == nil {
		return ImprovementFitness{}, errors.New("evaluation: improvement fitness store is absent")
	}
	authorities, err := requireImprovementAuthorities(ctx, store, request)
	if err != nil {
		return ImprovementFitness{}, err
	}
	coverage, err := authorities.query.project(ctx, store, improvementObservationCache(authorities.resources))
	if err != nil {
		return ImprovementFitness{}, err
	}
	if err := validateImprovementCoverage(authorities.query, coverage, authorities.evaluationLane); err != nil {
		return ImprovementFitness{}, err
	}
	value, err := newImprovementFitness(authorities, coverage)
	if err != nil {
		return ImprovementFitness{}, err
	}
	batch, err := improvementFitnessCodec.Batch(
		improvementFitnessCommitKey(value.ID), value, value.Lineage(), nil,
	)
	if err != nil {
		return ImprovementFitness{}, err
	}
	head := coverage.Head
	batch.ExpectedHead = &head
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return ImprovementFitness{}, err
	}
	return value, nil
}

// RequireImprovementFitness loads a persisted proof and re-verifies every
// immutable authority. Resource streams are replayed from their stored heads;
// current observation aliases are never consulted.
func RequireImprovementFitness(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (ImprovementFitness, error) {
	value, err := improvementFitnessCodec.Require(ctx, reader, id)
	if err != nil {
		return ImprovementFitness{}, err
	}
	repository, ok := reader.(improvementFitnessReader)
	if !ok {
		return ImprovementFitness{}, errors.New("evaluation: improvement fitness requires journal commit authority")
	}
	if err := requireCoverageProjectionCommit(ctx, repository, id, value.Coverage); err != nil {
		return ImprovementFitness{}, err
	}
	authorities, err := requireImprovementAuthorities(ctx, reader, ImprovementFitnessRequest{
		BaselineAttempt: value.Baseline.Attempt, CandidateAttempt: value.Candidate.Attempt,
		BaselineEvidence: value.Baseline.Evidence, CandidateEvidence: value.Candidate.Evidence,
		Coverage: value.CoverageQuery, Resources: value.Resources,
	})
	if err != nil {
		return ImprovementFitness{}, err
	}
	if err := validateImprovementCoverage(authorities.query, value.Coverage, authorities.evaluationLane); err != nil {
		return ImprovementFitness{}, err
	}
	expected, err := newImprovementFitness(authorities, value.Coverage)
	if err != nil || !reflect.DeepEqual(expected, value) {
		return ImprovementFitness{}, errors.Join(err, errors.New("evaluation: stored improvement fitness differs from exact authorities"))
	}
	return value, nil
}

func requireCoverageProjectionCommit(
	ctx context.Context,
	repository improvementFitnessReader,
	fitness artifact.ID,
	projection CoverageProjection,
) error {
	if projection.Sequence == ^uint64(0) {
		return errors.New("evaluation: improvement coverage journal authority differs")
	}
	nextSequence := projection.Sequence + 1
	expectedSequences := [...]uint64{projection.Sequence, nextSequence}
	result, err := repository.Query(ctx, overgodb.Query{
		FromSequence: projection.Sequence,
		ToSequence:   nextSequence,
		MaxResults:   len(expectedSequences),
		Projection:   overgodb.ProjectCommits,
	})
	if err != nil {
		return err
	}
	if result.Truncated || len(result.Commits) != len(expectedSequences) ||
		result.Commits[0].Sequence != expectedSequences[0] || result.Commits[0].ID != projection.Head ||
		result.Commits[1].Sequence != expectedSequences[1] ||
		result.Commits[1].Key != improvementFitnessCommitKey(fitness) {
		return errors.New("evaluation: improvement coverage journal authority differs")
	}
	introduction, found, err := repository.ArtifactIntroduction(ctx, fitness)
	if err != nil {
		return err
	}
	if !found || introduction.Artifact != fitness || introduction.Sequence != nextSequence ||
		introduction.Commit != result.Commits[1].ID {
		return errors.New("evaluation: improvement fitness was not introduced by its coverage publication commit")
	}
	return nil
}

func improvementFitnessCommitKey(id artifact.ID) string {
	return "evaluation/improvement-fitness/" + id.String()
}

// Lineage binds the proof directly to both execution chains, its resource
// comparison, and every immutable raw observation chunk it cites.
func (value ImprovementFitness) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Resources}
	for _, endpoint := range []ImprovementFitnessEndpoint{value.Baseline, value.Candidate} {
		parents = append(parents, endpoint.Strategy, endpoint.Attempt, endpoint.Trajectory,
			endpoint.Evidence, endpoint.Plan, endpoint.Run, endpoint.Evaluation)
	}
	parents = append(parents, value.SourceChunks...)
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

func requireImprovementAuthorities(
	ctx context.Context,
	reader artifact.Reader,
	request ImprovementFitnessRequest,
) (improvementAuthorities, error) {
	if ctx == nil || reader == nil || request.BaselineAttempt == request.CandidateAttempt ||
		request.BaselineEvidence == request.CandidateEvidence || request.Resources.Kind() != artifact.KindEvidence {
		return improvementAuthorities{}, errors.New("evaluation: invalid improvement fitness request")
	}
	baseline, err := requireImprovementEndpoint(ctx, reader, request.BaselineAttempt, request.BaselineEvidence)
	if err != nil {
		return improvementAuthorities{}, err
	}
	candidate, err := requireImprovementEndpoint(ctx, reader, request.CandidateAttempt, request.CandidateEvidence)
	if err != nil {
		return improvementAuthorities{}, err
	}
	if baseline.attempt.StrategyID == candidate.attempt.StrategyID ||
		baseline.attempt.TaskContract != candidate.attempt.TaskContract ||
		baseline.attempt.Environment != candidate.attempt.Environment ||
		baseline.evidence.Acceptance != candidate.evidence.Acceptance ||
		baseline.evidence.Dataset != candidate.evidence.Dataset || baseline.evidence.Split != candidate.evidence.Split ||
		baseline.plan.body.CaseProfile != candidate.plan.body.CaseProfile ||
		baseline.plan.body.Scorer != candidate.plan.body.Scorer || baseline.plan.body.Execution != candidate.plan.body.Execution {
		return improvementAuthorities{}, errors.New("evaluation: improvement endpoints use different task, environment, or quality authority")
	}
	baselineTools, err := requireImprovementToolAuthority(ctx, reader, baseline)
	if err != nil {
		return improvementAuthorities{}, err
	}
	candidateTools, err := requireImprovementToolAuthority(ctx, reader, candidate)
	if err != nil {
		return improvementAuthorities{}, err
	}
	resources, err := runrecord.RequireResourceFitnessComparison(ctx, reader, request.Resources)
	if err != nil {
		return improvementAuthorities{}, err
	}
	requiresTool := baselineTools.calls != 0 || candidateTools.calls != 0
	if err := requireImprovementLaneSet(resources, requiresTool); err != nil {
		return improvementAuthorities{}, err
	}
	agentLane, err := requireImprovementLane(resources, runrecord.ResourceLaneAgent, runrecord.SurfaceAgent,
		baseline.attempt.ID, candidate.attempt.ID)
	if err != nil {
		return improvementAuthorities{}, err
	}
	evaluationLane, err := requireImprovementLane(resources, runrecord.ResourceLaneEvaluation, runrecord.SurfaceEvaluation,
		baseline.evidence.Run, candidate.evidence.Run)
	if err != nil {
		return improvementAuthorities{}, err
	}
	if err := requireTypedAgentResources(ctx, reader, agentLane.Baseline, baseline); err != nil {
		return improvementAuthorities{}, err
	}
	if err := requireTypedAgentResources(ctx, reader, agentLane.Candidate, candidate); err != nil {
		return improvementAuthorities{}, err
	}
	if err := requireTypedEvaluationResources(ctx, reader, evaluationLane.Baseline, baseline); err != nil {
		return improvementAuthorities{}, err
	}
	if err := requireTypedEvaluationResources(ctx, reader, evaluationLane.Candidate, candidate); err != nil {
		return improvementAuthorities{}, err
	}
	if err := requireImprovementToolLane(ctx, reader, resources, baseline, candidate, baselineTools, candidateTools); err != nil {
		return improvementAuthorities{}, err
	}
	query, err := pinImprovementCoverage(request.Coverage, baseline, candidate, evaluationLane)
	if err != nil {
		return improvementAuthorities{}, err
	}
	capability := ImprovementFitnessDimension{
		NoRegression: candidate.attempt.Outcome == runrecord.OutcomeSucceeded,
		Strict:       baseline.attempt.Outcome != runrecord.OutcomeSucceeded && candidate.attempt.Outcome == runrecord.OutcomeSucceeded,
	}
	if !capability.NoRegression {
		return improvementAuthorities{}, errors.New("evaluation: candidate capability regresses")
	}
	qualityNoRegression, qualityStrict := orderedMetricRelation(candidate.evidence.Metrics, baseline.evidence.Metrics)
	if !qualityNoRegression {
		return improvementAuthorities{}, errors.New("evaluation: candidate quality regresses or is incomparable")
	}
	quality := ImprovementFitnessDimension{NoRegression: true, Strict: qualityStrict}
	resource := ImprovementFitnessDimension{NoRegression: true, Strict: resources.StrictImprovement}
	if !capability.Strict && !quality.Strict && !resource.Strict {
		return improvementAuthorities{}, errors.New("evaluation: candidate records no strict improvement")
	}
	return improvementAuthorities{
		baseline: baseline, candidate: candidate, resources: resources, query: query,
		agentLane: agentLane, evaluationLane: evaluationLane,
		capability: capability, quality: quality, resource: resource,
	}, nil
}

func requireImprovementEndpoint(
	ctx context.Context,
	reader artifact.Reader,
	attemptID, evidenceID artifact.ID,
) (improvementEndpointAuthority, error) {
	attempt, err := runrecord.RequireAttemptRecord(ctx, reader, attemptID)
	if err != nil || !attempt.StrategyID.Valid() || !attempt.TaskContract.Valid() || !attempt.Environment.Valid() ||
		!attempt.Trajectory.Valid() {
		return improvementEndpointAuthority{}, errors.Join(err, errors.New("evaluation: improvement attempt lacks exact authorities"))
	}
	gate, err := runrecord.RequireGateResult(ctx, reader, attempt.Result)
	if err != nil || gate.Recipe != attempt.Recipe || gate.Environment != attempt.Environment ||
		gate.CodeCommit != attempt.CodeCommit || gate.Outcome != attempt.Outcome || gate.Failure != attempt.Failure {
		return improvementEndpointAuthority{}, errors.Join(err, errors.New("evaluation: gate result differs from improvement attempt"))
	}
	trajectory, err := runrecord.RequireInteractionTrace(ctx, reader, attempt.Trajectory)
	if err != nil || trajectory.TaskContract != attempt.TaskContract ||
		trajectory.Strategy != attempt.StrategyID || trajectory.Terminal != attempt.Outcome {
		return improvementEndpointAuthority{}, errors.Join(err, errors.New("evaluation: trajectory differs from improvement attempt"))
	}
	task, err := recipe.RequireAgentTaskContract(ctx, reader, attempt.TaskContract)
	if err != nil {
		return improvementEndpointAuthority{}, err
	}
	agent, err := recipe.RequireAgentDefinition(ctx, reader, task.Agent)
	if err != nil {
		return improvementEndpointAuthority{}, err
	}
	strategyRecipe, err := recipe.RequireDefinition(ctx, reader, trajectory.Recipe)
	if err != nil || agent.ModelRecipe != trajectory.Recipe || strategyRecipe.Model != trajectory.Model {
		return improvementEndpointAuthority{}, errors.Join(err, errors.New("evaluation: trajectory recipe differs from typed agent and model"))
	}
	environment, err := runrecord.RequireEnvironment(ctx, reader, attempt.Environment)
	if err != nil {
		return improvementEndpointAuthority{}, err
	}
	evidence, err := RequireEvaluationEvidence(ctx, reader, evidenceID)
	if err != nil {
		return improvementEndpointAuthority{}, err
	}
	evaluationRecipe, err := recipe.RequireDefinition(ctx, reader, evidence.Recipe)
	if err != nil || evaluationRecipe.Model != trajectory.Model {
		return improvementEndpointAuthority{}, errors.Join(err, errors.New("evaluation: evaluation recipe differs from trajectory model"))
	}
	if err := modelrecipe.RequireModelDefinitionBinding(ctx, reader, evidence.ModelDefinition, trajectory.Model, evidence.Recipe); err != nil {
		return improvementEndpointAuthority{}, err
	}
	if err := requireImprovementDatasetSplit(ctx, reader, evidence.Dataset, evidence.Split); err != nil {
		return improvementEndpointAuthority{}, err
	}
	plan, err := loadEvidencePlan(ctx, reader, evidence.Plan)
	if err != nil {
		return improvementEndpointAuthority{}, err
	}
	planAgent, found := plan.AgentTrajectoryAuthorities()
	if !found || len(planAgent.Trajectories) != 1 || planAgent.Trajectories[0] != attempt.Trajectory {
		return improvementEndpointAuthority{}, errors.New("evaluation: evidence plan differs from the exact improvement trajectory")
	}
	run, err := runrecord.RequireRun(ctx, reader, evidence.Run)
	if err != nil || run.Outcome != runrecord.OutcomeSucceeded || evidence.CodeCommit != attempt.CodeCommit ||
		evidence.Environment != attempt.Environment {
		return improvementEndpointAuthority{}, errors.Join(err, errors.New("evaluation: evaluation run differs from improvement attempt"))
	}
	return improvementEndpointAuthority{
		value: ImprovementFitnessEndpoint{
			Strategy: attempt.StrategyID, Attempt: attempt.ID, Trajectory: attempt.Trajectory,
			Evidence: evidence.ID, Plan: evidence.Plan, Run: evidence.Run, Evaluation: evidence.Evaluation,
			Quality: slices.Clone(evidence.Metrics),
		},
		attempt: attempt, gate: gate, trajectory: trajectory, evidence: evidence, plan: plan, run: run,
		task: task, agent: agent, environment: environment,
	}, nil
}

func requireImprovementLane(
	comparison runrecord.ResourceFitnessComparison,
	name string,
	surface runrecord.InteractionSurface,
	baselineAttempt, candidateAttempt artifact.ID,
) (runrecord.ResourceFitnessLane, error) {
	index := slices.IndexFunc(comparison.Lanes, func(lane runrecord.ResourceFitnessLane) bool { return lane.Name == name })
	if index < 0 {
		return runrecord.ResourceFitnessLane{}, fmt.Errorf("evaluation: resource comparison omits required %s lane", name)
	}
	lane := comparison.Lanes[index]
	if lane.Baseline.Scope.Surface != surface || lane.Candidate.Scope.Surface != surface ||
		lane.Baseline.Scope.Attempt != baselineAttempt || lane.Candidate.Scope.Attempt != candidateAttempt ||
		!lane.RequireInteractions {
		return runrecord.ResourceFitnessLane{}, fmt.Errorf("evaluation: resource comparison %s lane authority differs", name)
	}
	for _, metric := range requiredImprovementResourceMetrics {
		if !slices.Contains(lane.RequiredMetrics, metric) {
			return runrecord.ResourceFitnessLane{}, fmt.Errorf("evaluation: resource comparison %s lane omits core metric %q", name, metric)
		}
	}
	return lane, nil
}

func requireTypedAgentResources(
	ctx context.Context,
	reader artifact.Reader,
	stream runrecord.ObservationStream,
	endpoint improvementEndpointAuthority,
) error {
	attempt := endpoint.attempt
	wall, wallObserved := stream.Aggregate.Measure(runrecord.ResourceWallNS)
	cost, costObserved := stream.Aggregate.Measure(runrecord.ResourceCostUnits)
	if stream.Scope.Surface != runrecord.SurfaceAgent || stream.Scope.Attempt != attempt.ID ||
		stream.Scope.Workload != attempt.TaskContract || stream.Scope.Hardware != attempt.Environment ||
		stream.Scope.Model != endpoint.trajectory.Model ||
		!wallObserved || wall != attempt.WallNS ||
		!costObserved || cost != attempt.CostUnits {
		return errors.New("evaluation: agent resource aggregate differs from typed attempt")
	}
	return requireImprovementProvider(ctx, reader, stream, endpoint.environment)
}

func requireTypedEvaluationResources(
	ctx context.Context,
	reader artifact.Reader,
	stream runrecord.ObservationStream,
	endpoint improvementEndpointAuthority,
) error {
	wall, observed := stream.Aggregate.Measure(runrecord.ResourceWallNS)
	if stream.Scope.Attempt != endpoint.evidence.Run || stream.Scope.Surface != runrecord.SurfaceEvaluation ||
		stream.Scope.Workload != endpoint.evidence.Plan || stream.Scope.Hardware != endpoint.evidence.Environment ||
		stream.Scope.Model != endpoint.trajectory.Model ||
		!observed || wall != endpoint.run.MeasuredNS {
		return errors.New("evaluation: evaluation resource aggregate differs from typed evidence run")
	}
	return requireImprovementProvider(ctx, reader, stream, endpoint.environment)
}

func requireImprovementToolLane(
	ctx context.Context,
	reader artifact.Reader,
	comparison runrecord.ResourceFitnessComparison,
	baseline, candidate improvementEndpointAuthority,
	baselineTools, candidateTools improvementToolAuthority,
) error {
	toolIndex := slices.IndexFunc(comparison.Lanes, func(lane runrecord.ResourceFitnessLane) bool {
		return lane.Name == runrecord.ResourceLaneTool
	})
	requiresTool := baselineTools.calls != 0 || candidateTools.calls != 0
	if !requiresTool && toolIndex < 0 {
		return nil
	}
	if !requiresTool {
		return errors.New("evaluation: resource comparison has an unrelated tool lane")
	}
	lane, err := requireImprovementLane(
		comparison, runrecord.ResourceLaneTool, runrecord.SurfaceTool,
		baseline.trajectory.ID, candidate.trajectory.ID,
	)
	if err != nil {
		return err
	}
	for _, arm := range []struct {
		stream   runrecord.ObservationStream
		endpoint improvementEndpointAuthority
		tools    improvementToolAuthority
	}{{lane.Baseline, baseline, baselineTools}, {lane.Candidate, candidate, candidateTools}} {
		if arm.stream.Scope.Workload != arm.endpoint.trajectory.TaskContract ||
			arm.stream.Scope.Hardware != arm.endpoint.attempt.Environment ||
			arm.stream.Scope.Model != arm.endpoint.trajectory.Model {
			return errors.New("evaluation: tool resource lane scope differs from trajectory authority")
		}
		if err := requireImprovementProvider(ctx, reader, arm.stream, arm.endpoint.environment); err != nil {
			return err
		}
		if arm.tools.capability.Valid() && arm.stream.Scope.Provider != arm.tools.capability {
			return errors.New("evaluation: tool resource provider differs from called manuals")
		}
		work := arm.stream.Aggregate.Interactions
		if work == nil || work.ToolCalls != arm.tools.calls || work.Failures != arm.tools.failures {
			return errors.New("evaluation: tool resource aggregate differs from exact trajectory calls")
		}
	}
	return nil
}

func requireImprovementToolAuthority(
	ctx context.Context,
	reader artifact.Reader,
	endpoint improvementEndpointAuthority,
) (improvementToolAuthority, error) {
	var authority improvementToolAuthority
	calledManuals := make([]artifact.ID, 0)
	calls, failures, err := endpoint.trajectory.ToolExchanges()
	if err != nil {
		return improvementToolAuthority{}, err
	}
	authority.failures = failures
	declaresTools := len(calls) != 0 || len(endpoint.trajectory.ToolActions) != 0 ||
		len(endpoint.trajectory.ToolManuals) != 0 || len(endpoint.trajectory.InvocationEffects) != 0
	for _, call := range calls {
		manual, err := agenttool.RequireManual(ctx, reader, call.Manual)
		if err != nil || manual.Name != call.Name || !slices.Contains(endpoint.agent.ToolManuals, manual.ID) {
			return improvementToolAuthority{}, errors.Join(err, errors.New("evaluation: trajectory tool call lacks an admitted exact manual"))
		}
		if _, err := agenttool.DeriveInvocationEffect(manual, json.RawMessage(call.Arguments), nil); err != nil {
			return improvementToolAuthority{}, errors.Join(err, errors.New("evaluation: trajectory tool arguments differ from exact manual"))
		}
		if !manual.Capability.Valid() || authority.capability.Valid() && authority.capability != manual.Capability {
			return improvementToolAuthority{}, errors.New("evaluation: trajectory tool calls use different or absent capabilities")
		}
		authority.capability = manual.Capability
		authority.calls++
		calledManuals = append(calledManuals, manual.ID)
	}
	if authority.calls == 0 {
		if declaresTools {
			return improvementToolAuthority{}, errors.New("evaluation: trajectory declares tool work without exact calls")
		}
		return improvementToolAuthority{}, nil
	}
	if len(endpoint.trajectory.ToolManuals) != 0 {
		slices.SortFunc(calledManuals, artifact.CompareID)
		calledManuals = slices.Compact(calledManuals)
		if !slices.Equal(calledManuals, endpoint.trajectory.ToolManuals) {
			return improvementToolAuthority{}, errors.New("evaluation: trajectory tool manual set differs from exact calls")
		}
	}
	return authority, nil
}

func requireImprovementProvider(
	ctx context.Context,
	reader artifact.Reader,
	stream runrecord.ObservationStream,
	environment runrecord.Environment,
) error {
	provider, err := runrecord.RequireCapabilityIdentity(ctx, reader, stream.Scope.Provider)
	if err != nil || provider.Platform.OS != environment.OS || provider.Platform.Arch != environment.Arch {
		return errors.Join(err, errors.New("evaluation: resource provider differs from typed environment"))
	}
	return nil
}

func requireImprovementDatasetSplit(
	ctx context.Context,
	reader artifact.Reader,
	dataset, split artifact.ID,
) error {
	datasetContent, found, err := artifact.ReadContent(ctx, reader, dataset)
	if err != nil || !found {
		return errors.Join(err, errors.New("evaluation: improvement dataset content is absent"))
	}
	splitContent, found, err := artifact.ReadContent(ctx, reader, split)
	if err != nil || !found {
		return errors.Join(err, errors.New("evaluation: improvement split content is absent"))
	}
	registered := false
	for _, pair := range improvementDatasetContracts {
		if pair.dataset.ValidateContent(datasetContent, dataset) == nil && pair.split.ValidateContent(splitContent, split) == nil {
			registered = true
			break
		}
	}
	if !registered {
		return errors.New("evaluation: improvement dataset and split do not share a registered native contract")
	}
	var binding struct {
		Dataset artifact.ID `json:"dataset"`
	}
	if err := strictjson.DecodeBytes(splitContent.Data, &binding); err != nil || binding.Dataset != dataset {
		return errors.Join(err, errors.New("evaluation: improvement split differs from its exact dataset"))
	}
	parents, err := reader.Parents(ctx, split)
	if err != nil {
		return err
	}
	if !slices.ContainsFunc(parents, func(edge artifact.Lineage) bool {
		return edge.Child == split && edge.Parent == dataset && edge.Relation == artifact.RelationDependsOn
	}) {
		return errors.New("evaluation: improvement split lacks exact dataset lineage")
	}
	return nil
}

func requireImprovementLaneSet(comparison runrecord.ResourceFitnessComparison, requiresTool bool) error {
	want := []string{runrecord.ResourceLaneAgent, runrecord.ResourceLaneEvaluation}
	if requiresTool {
		want = append(want, runrecord.ResourceLaneTool)
	}
	if len(comparison.Lanes) != len(want) {
		return errors.New("evaluation: resource comparison contains an unrelated or missing lane")
	}
	for _, name := range want {
		if !slices.ContainsFunc(comparison.Lanes, func(lane runrecord.ResourceFitnessLane) bool { return lane.Name == name }) {
			return errors.New("evaluation: resource comparison contains an unrelated or missing lane")
		}
	}
	return nil
}

func pinImprovementCoverage(
	query CoverageQuery,
	baseline, candidate improvementEndpointAuthority,
	evaluationLane runrecord.ResourceFitnessLane,
) (CoverageQuery, error) {
	canonical, err := NewEvidenceCoverageQuery(query.Units, query.Pairs, query.Bounds)
	if err != nil || query.ID.Valid() && query.ID != canonical.ID {
		return CoverageQuery{}, errors.Join(err, errors.New("evaluation: improvement coverage query is not canonical"))
	}
	endpoints := []struct {
		authority improvementEndpointAuthority
		stream    runrecord.ObservationStream
	}{
		{authority: baseline, stream: evaluationLane.Baseline},
		{authority: candidate, stream: evaluationLane.Candidate},
	}
	expectedPairs := []CoveragePair{{Baseline: baseline.evidence.Run, Candidate: candidate.evidence.Run}}
	if len(canonical.Units) != len(endpoints) || !slices.Equal(canonical.Pairs, expectedPairs) {
		return CoverageQuery{}, errors.New("evaluation: improvement coverage must contain one exact ordered pair")
	}
	measurementSamples, hardwareSamples, err := improvementCoverageDenominators(evaluationLane.Baseline)
	if err != nil {
		return CoverageQuery{}, err
	}
	for _, endpoint := range endpoints {
		unitIndex := slices.IndexFunc(canonical.Units, func(unit CoverageUnit) bool {
			return unit.Attempt == endpoint.authority.evidence.Run
		})
		if unitIndex < 0 {
			return CoverageQuery{}, errors.New("evaluation: improvement coverage omits an evaluation run")
		}
		unit := &canonical.Units[unitIndex]
		head := endpoint.stream.SummaryIDs[len(endpoint.stream.SummaryIDs)-1]
		if unit.Terminal != endpoint.authority.evidence.Run ||
			!slices.Equal(unit.EvaluationEvidence, []artifact.ID{endpoint.authority.evidence.ID}) ||
			len(unit.FailureObservations) != 0 || unit.ObservationHead.Valid() && unit.ObservationHead != head {
			return CoverageQuery{}, errors.New("evaluation: improvement coverage unit differs from exact evidence")
		}
		unit.ObservationHead = head
		for index := range unit.Required {
			switch unit.Required[index].Axis {
			case CoverageMeasurement:
				unit.Required[index].ExpectedSamples = measurementSamples
			case CoverageHardware:
				unit.Required[index].ExpectedSamples = hardwareSamples
			}
		}
		if err := requireImprovementCoverageContract(unit.Required); err != nil {
			return CoverageQuery{}, err
		}
	}
	return NewEvidenceCoverageQuery(canonical.Units, canonical.Pairs, canonical.Bounds)
}

func improvementCoverageDenominators(stream runrecord.ObservationStream) (uint32, uint32, error) {
	minimum := func(counts []runrecord.ObservationMetricSamples, metrics []runrecord.ResourceMetric) uint32 {
		result := observationMetricSamples(counts, metrics[0])
		for _, metric := range metrics[1:] {
			count := observationMetricSamples(counts, metric)
			result = min(result, count)
		}
		return result
	}
	measurement := minimum(stream.Coverage.Metrics, requiredImprovementResourceMetrics)
	hardware := minimum(stream.Coverage.HardwareMetrics, []runrecord.ResourceMetric{
		runrecord.ResourcePeakDeviceBytes, runrecord.ResourcePeakHostBytes,
	})
	if measurement == 0 || hardware == 0 {
		return 0, 0, errors.New("evaluation: baseline stream cannot define complete improvement coverage")
	}
	return measurement, hardware, nil
}

func requireImprovementCoverageContract(requirements []CoverageRequirement) error {
	if len(requirements) != len(requiredImprovementCoverageAxes) {
		return errors.New("evaluation: improvement coverage requires exactly measurement, hardware, and evaluation")
	}
	measurement := slices.IndexFunc(requirements, func(requirement CoverageRequirement) bool {
		return requirement.Axis == CoverageMeasurement
	})
	evaluation := slices.IndexFunc(requirements, func(requirement CoverageRequirement) bool {
		return requirement.Axis == CoverageEvaluation
	})
	hardware := slices.IndexFunc(requirements, func(requirement CoverageRequirement) bool {
		return requirement.Axis == CoverageHardware
	})
	if measurement < 0 || evaluation < 0 || hardware < 0 {
		return errors.New("evaluation: improvement coverage omits measurement, hardware, or evaluation")
	}
	for _, metric := range requiredImprovementResourceMetrics {
		if !slices.Contains(requirements[measurement].Metrics, metric) {
			return fmt.Errorf("evaluation: improvement coverage omits core metric %q", metric)
		}
	}
	for _, metric := range []runrecord.ResourceMetric{
		runrecord.ResourcePeakDeviceBytes, runrecord.ResourcePeakHostBytes,
	} {
		if !slices.Contains(requirements[hardware].Metrics, metric) {
			return fmt.Errorf("evaluation: improvement hardware coverage omits core metric %q", metric)
		}
	}
	return nil
}

func validateImprovementCoverage(
	query CoverageQuery,
	projection CoverageProjection,
	evaluationLane runrecord.ResourceFitnessLane,
) error {
	expectedUnits := uint64(len(query.Units))
	expectedPairs := uint64(len(query.Pairs))
	if projection.Version != EvidenceCoverageProjectionVersion || projection.Query != query.ID ||
		projection.FailureClassifierVersion != executionfailure.ClassifierVersion ||
		projection.CausalityProjectionVersion != overgodb.CausalityProjectionVersion ||
		!projection.Head.Valid() || projection.Sequence == 0 || projection.Units != expectedUnits ||
		projection.Failures != 0 || projection.FailuresClassified != 0 || projection.FailuresUnclassified != 0 ||
		projection.Costed != expectedUnits || projection.Uncosted != 0 || projection.Recovered != 0 ||
		projection.Pairs != expectedPairs || projection.Paired != expectedPairs || projection.Unpaired != 0 {
		return errors.New("evaluation: improvement coverage is incomplete")
	}
	if projection.Causal != (AxisCoverage{}) {
		return errors.New("evaluation: improvement coverage records a non-required causal axis")
	}
	for _, axis := range requiredImprovementCoverageAxes {
		counts := projection.axis(axis)
		if counts.Observed != expectedUnits || counts.Missing != 0 || counts.Degraded != 0 {
			return fmt.Errorf("evaluation: improvement coverage axis %q is incomplete", axis)
		}
	}
	for _, endpoint := range []struct {
		attempt artifact.ID
		stream  runrecord.ObservationStream
	}{
		{attempt: evaluationLane.Baseline.Scope.Attempt, stream: evaluationLane.Baseline},
		{attempt: evaluationLane.Candidate.Scope.Attempt, stream: evaluationLane.Candidate},
	} {
		unitIndex := slices.IndexFunc(query.Units, func(unit CoverageUnit) bool { return unit.Attempt == endpoint.attempt })
		if unitIndex < 0 {
			return errors.New("evaluation: improvement coverage lacks a resource endpoint")
		}
		unit := query.Units[unitIndex]
		if unit.ObservationHead != endpoint.stream.SummaryIDs[len(endpoint.stream.SummaryIDs)-1] {
			return errors.New("evaluation: improvement coverage stream head differs from resources")
		}
		for _, requirement := range unit.Required {
			switch requirement.Axis {
			case CoverageMeasurement:
				if measurementCoverage(endpoint.stream, true, requirement) != coverageObserved {
					return errors.New("evaluation: improvement measurement coverage differs from exact stream")
				}
			case CoverageHardware:
				if hardwareCoverage(endpoint.stream, true, requirement) != coverageObserved {
					return errors.New("evaluation: improvement hardware coverage differs from exact stream")
				}
			case CoverageEvaluation:
				if len(unit.EvaluationEvidence) == 0 {
					return errors.New("evaluation: improvement evaluation coverage differs from exact evidence")
				}
			default:
				return errors.New("evaluation: improvement coverage has a foreign axis")
			}
		}
	}
	wantChunks := append(slices.Clone(evaluationLane.Baseline.ChunkIDs), evaluationLane.Candidate.ChunkIDs...)
	slices.SortFunc(wantChunks, artifact.CompareID)
	wantChunks = slices.Compact(wantChunks)
	if !slices.Equal(projection.SourceChunks, wantChunks) {
		return errors.New("evaluation: improvement coverage source chunks differ from evaluation resources")
	}
	return nil
}

func newImprovementFitness(
	authorities improvementAuthorities,
	coverage CoverageProjection,
) (ImprovementFitness, error) {
	sources := append(slices.Clone(coverage.SourceChunks), authorities.resources.SourceChunks()...)
	slices.SortFunc(sources, artifact.CompareID)
	sources = slices.Compact(sources)
	return improvementFitnessCodec.New(ImprovementFitness{
		Version:  ImprovementFitnessVersion,
		Baseline: authorities.baseline.value, Candidate: authorities.candidate.value,
		Resources: authorities.resources.ID, CoverageQuery: authorities.query, Coverage: coverage,
		Capability: authorities.capability, Quality: authorities.quality, Resource: authorities.resource,
		SourceChunks: sources, Improved: true,
	})
}

func improvementObservationCache(comparison runrecord.ResourceFitnessComparison) coverageObservationCache {
	cache := make(coverageObservationCache, len(comparison.Lanes)+len(comparison.Lanes))
	for _, lane := range comparison.Lanes {
		for _, stream := range []runrecord.ObservationStream{lane.Baseline, lane.Candidate} {
			head := stream.SummaryIDs[len(stream.SummaryIDs)-1]
			cache[coverageObservationKey{attempt: stream.Scope.Attempt, head: head}] = stream
		}
	}
	return cache
}

func canonicalizeImprovementFitness(value *ImprovementFitness) error {
	if value == nil || value.Version != ImprovementFitnessVersion || value.Resources.Kind() != artifact.KindEvidence ||
		!value.Coverage.Head.Valid() || value.Coverage.Sequence == 0 || !value.Improved {
		return errors.New("evaluation: invalid improvement fitness")
	}
	canonicalQuery, err := NewEvidenceCoverageQuery(
		value.CoverageQuery.Units, value.CoverageQuery.Pairs, value.CoverageQuery.Bounds,
	)
	if err != nil || value.CoverageQuery.ID.Valid() && value.CoverageQuery.ID != canonicalQuery.ID ||
		value.Coverage.Query != canonicalQuery.ID {
		return errors.Join(err, errors.New("evaluation: improvement fitness coverage query differs"))
	}
	value.CoverageQuery = canonicalQuery
	if err := canonicalizeImprovementEndpoint(&value.Baseline); err != nil {
		return err
	}
	if err := canonicalizeImprovementEndpoint(&value.Candidate); err != nil {
		return err
	}
	if value.Baseline.Attempt == value.Candidate.Attempt || value.Baseline.Strategy == value.Candidate.Strategy ||
		value.Baseline.Trajectory == value.Candidate.Trajectory || value.Baseline.Evidence == value.Candidate.Evidence ||
		!value.Capability.NoRegression || !value.Quality.NoRegression || !value.Resource.NoRegression ||
		!value.Capability.Strict && !value.Quality.Strict && !value.Resource.Strict {
		return errors.New("evaluation: improvement fitness records no exact strict improvement")
	}
	sourceCount := len(value.SourceChunks)
	value.SourceChunks = slices.Clone(value.SourceChunks)
	slices.SortFunc(value.SourceChunks, artifact.CompareID)
	value.SourceChunks = slices.Compact(value.SourceChunks)
	if len(value.SourceChunks) == 0 || len(value.SourceChunks) != sourceCount ||
		slices.ContainsFunc(value.SourceChunks, func(id artifact.ID) bool { return id.Kind() != artifact.KindFile }) {
		return errors.New("evaluation: invalid improvement fitness source chunks")
	}
	return nil
}

func canonicalizeImprovementEndpoint(endpoint *ImprovementFitnessEndpoint) error {
	if endpoint == nil || endpoint.Strategy.Kind() != artifact.KindProfile ||
		endpoint.Attempt.Kind() != artifact.KindEvidence || endpoint.Trajectory.Kind() != artifact.KindEvidence ||
		endpoint.Evidence.Kind() != artifact.KindEvidence || endpoint.Plan.Kind() != artifact.KindProfile ||
		endpoint.Run.Kind() != artifact.KindRun || endpoint.Evaluation.Kind() != artifact.KindEvaluation ||
		len(endpoint.Quality) == 0 {
		return errors.New("evaluation: invalid improvement fitness endpoint")
	}
	endpoint.Quality = slices.Clone(endpoint.Quality)
	for index, metric := range endpoint.Quality {
		if metric.Name == "" || !finite(metric.Value) ||
			metric.Direction != runrecord.DirectionMinimize && metric.Direction != runrecord.DirectionMaximize ||
			index > 0 && endpoint.Quality[index-1].Name >= metric.Name {
			return errors.New("evaluation: invalid improvement fitness quality metric")
		}
	}
	return nil
}

func cloneImprovementFitness(value ImprovementFitness) ImprovementFitness {
	value.Baseline.Quality = slices.Clone(value.Baseline.Quality)
	value.Candidate.Quality = slices.Clone(value.Candidate.Quality)
	value.CoverageQuery = CoverageQuery{
		Units: cloneCoverageUnits(value.CoverageQuery.Units), Pairs: slices.Clone(value.CoverageQuery.Pairs),
		Bounds: value.CoverageQuery.Bounds, ID: value.CoverageQuery.ID,
	}
	value.Coverage.SourceChunks = slices.Clone(value.Coverage.SourceChunks)
	value.SourceChunks = slices.Clone(value.SourceChunks)
	return value
}
