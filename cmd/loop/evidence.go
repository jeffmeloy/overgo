package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/runrecord"
)

// strategySpec names the authorities one published strategy binds: the
// worker agent definition, the tool catalog, and the loop budgets.
type strategySpec struct {
	Worker  artifact.ID `json:"worker"`
	Catalog artifact.ID `json:"catalog"`
	Loop    loop.Config `json:"loop"`
}

// publishStrategy is the one production door that mints a strategy: it
// resolves the exact worker definition, derives the content-addressed
// strategy, and commits it with its lineage. The loop then runs it by
// identity through the existing strategy_id configuration.
func publishStrategy(root, specPath string, output io.Writer) error {
	var spec strategySpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	worker, err := recipe.RequireAgentDefinition(ctx, store, spec.Worker)
	if err != nil {
		return err
	}
	strategy, err := loop.NewStrategy(worker, spec.Catalog, spec.Loop)
	if err != nil {
		return err
	}
	content, err := strategy.Content()
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:       "loop/strategy/" + strategy.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
		Lineage:   strategy.Lineage(),
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "published strategy %s\n", strategy.ID)
	return err
}

// experimentSpec binds one strategy comparison: the task contract, the
// baseline commit, the competing candidates, and the fitness evidence.
type experimentSpec struct {
	Task       artifact.ID                        `json:"task"`
	Baseline   string                             `json:"baseline"`
	Candidates []loop.StrategyExperimentCandidate `json:"candidates"`
	Fitness    []artifact.ID                      `json:"fitness"`
}

// compareStrategies replays one strategy experiment from immutable store
// state and prints the typed comparison; it decides nothing and widens no
// budget on its own.
func compareStrategies(root, specPath string, output io.Writer) error {
	var spec experimentSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	task, err := recipe.RequireAgentTaskContract(ctx, store, spec.Task)
	if err != nil {
		return err
	}
	experiment, comparison, err := loop.CompareStrategyExperiment(
		ctx, store, task, spec.Baseline, spec.Candidates, spec.Fitness,
	)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(struct {
		Experiment loop.StrategyExperiment `json:"experiment"`
		Comparison loop.StrategyComparison `json:"comparison"`
	}{experiment, comparison}, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}

// publishFitness commits one pairwise improvement-fitness proof against the
// exact journal head its coverage projection observed.
func publishFitness(root, specPath string, output io.Writer) error {
	var request evaluation.ImprovementFitnessRequest
	if err := jsonfile.DecodeStrict(specPath, &request); err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	fitness, err := evaluation.PublishImprovementFitness(context.Background(), store, request)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "published improvement fitness %s\n", fitness.ID)
	return err
}

// taskContractSpec is one operator-authored agent task contract.
type taskContractSpec struct {
	Contract recipe.AgentTaskContract `json:"contract"`
}

// publishTaskContract mints and commits one agent task contract; the
// contract is the delegation ceiling every invocation must name.
func publishTaskContract(root, specPath string, output io.Writer) error {
	var spec taskContractSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	contract, err := recipe.NewAgentTaskContract(spec.Contract)
	if err != nil {
		return err
	}
	return commitDocument(root, "loop/agent-task/"+contract.ID.String(), contract, output,
		fmt.Sprintf("published agent task contract %s\n", contract.ID))
}

// delegationSpec is one operator-authored delegated agent invocation.
type delegationSpec struct {
	Invocation recipe.DelegatedAgentInvocation `json:"invocation"`
}

// publishDelegation mints and commits one delegated agent invocation: the
// exact worker, grant, scheduler, catalog, and model a delegated session
// runs under.
func publishDelegation(root, specPath string, output io.Writer) error {
	var spec delegationSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	invocation, err := recipe.NewDelegatedAgentInvocation(spec.Invocation)
	if err != nil {
		return err
	}
	return commitDocument(root, "loop/delegation/"+invocation.ID.String(), invocation, output,
		fmt.Sprintf("published delegated invocation %s\n", invocation.ID))
}

// automationTransitionSpec is one automation policy lifecycle transition.
type automationTransitionSpec struct {
	Lifecycle runrecord.AutomationPolicyLifecycle `json:"lifecycle"`
}

// publishAutomationTransition commits one standing-grant lifecycle
// transition through its chained owner; an invalid chain refuses.
func publishAutomationTransition(root, specPath string, output io.Writer) error {
	var spec automationTransitionSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	published, err := runrecord.PublishAutomationPolicyTransition(context.Background(), store, spec.Lifecycle)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "published automation policy transition %s\n", published.ID)
	return err
}

// trajectoryPlanSpec binds agent trajectories and advisory judges onto one
// stored evaluation plan.
type trajectoryPlanSpec struct {
	Plan           artifact.ID   `json:"plan"`
	Trajectories   []artifact.ID `json:"trajectories"`
	AdvisoryJudges []artifact.ID `json:"advisory_judges,omitempty"`
}

// bindTrajectoryPlan loads the base evaluation plan, binds the trajectory
// authorities -- deterministic scores stay fixed, model judges stay
// advisory -- and commits the re-identified plan.
func bindTrajectoryPlan(root, specPath string, output io.Writer) error {
	var spec trajectoryPlanSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	content, found, err := artifact.ReadContent(ctx, store, spec.Plan)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("loop: base evaluation plan is not in the store")
	}
	base, err := evaluation.ParsePlan(content.Data)
	if err != nil {
		return err
	}
	bound, err := evaluation.BindAgentTrajectoryPlan(base, spec.Trajectories, spec.AdvisoryJudges)
	if err != nil {
		return err
	}
	boundContent, err := bound.Content()
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:       "loop/trajectory-plan/" + bound.Identity().String(),
		Artifacts: []artifact.Descriptor{boundContent.Descriptor},
		Contents:  []artifact.Content{boundContent},
		Lineage:   bound.Lineage(),
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "bound trajectory plan %s\n", bound.Identity())
	return err
}

// efficiencyTraceSpec is one measured interaction-work record.
type efficiencyTraceSpec struct {
	Trace runrecord.EfficiencyTrace `json:"trace"`
}

// publishEfficiencyTrace mints and commits one efficiency trace: the exact
// work one representative task cost on one surface, bound to the completed
// result and its evidence so a reduction can never take credit for doing
// less of the task.
func publishEfficiencyTrace(root, specPath string, output io.Writer) error {
	var spec efficiencyTraceSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	trace, err := runrecord.NewEfficiencyTrace(spec.Trace)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	content, err := trace.Content()
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, artifact.Batch{
		Key:       "loop/efficiency-trace/" + trace.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "published efficiency trace %s\n", trace.ID)
	return err
}

// directionSpec is one extracted residual-direction claim.
type directionSpec struct {
	Direction representation.ResidualDirection `json:"direction"`
}

// publishDirection mints and commits one residual direction with the full
// authority lineage its claim depends on; the steering candidate spine
// admits it only through the common admission entry.
func publishDirection(root, specPath string, output io.Writer) error {
	var spec directionSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	direction, err := representation.NewResidualDirection(spec.Direction)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	content, err := direction.Content()
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, artifact.Batch{
		Key:       "loop/residual-direction/" + direction.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
		Lineage:   direction.Lineage(),
	}); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "published residual direction %s\n", direction.ID)
	return err
}

// attemptReceiptSpec addresses one operation's terminal attempt receipt.
type attemptReceiptSpec struct {
	Operation artifact.ID `json:"operation"`
	Attempt   uint32      `json:"attempt"`
}

// readAttemptReceipt resolves and prints one terminal attempt receipt: the
// operator's read path over the exact recovery evidence chain.
func readAttemptReceipt(root, specPath string, output io.Writer) error {
	var spec attemptReceiptSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	receipt, found, err := runrecord.ResolveTerminalAttemptReceipt(context.Background(), store, spec.Operation, spec.Attempt)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("loop: operation %s attempt %d has no terminal receipt", spec.Operation, spec.Attempt)
	}
	encoded, err := json.MarshalIndent(receipt, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}

// commitDocument commits one content-addressed document with its lineage.
func commitDocument(root, key string, document interface {
	Content() (artifact.Content, error)
	Lineage() []artifact.Lineage
}, output io.Writer, message string) error {
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	content, err := document.Content()
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, artifact.Batch{
		Key:       key,
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
		Lineage:   document.Lineage(),
	}); err != nil {
		return err
	}
	_, err = fmt.Fprint(output, message)
	return err
}

// candidateEvaluationSpec binds one controlled comparison: the admitted
// candidate, the driver decision that authorized evaluation, the frozen
// evaluator and inputs, the pre-declared requirements and tradeoffs, and
// the measured arms.
type candidateEvaluationSpec struct {
	Candidate    artifact.ID                              `json:"candidate"`
	Admission    artifact.ID                              `json:"admission"`
	Decision     artifact.ID                              `json:"decision"`
	Evaluator    artifact.ID                              `json:"evaluator"`
	Inputs       artifact.ID                              `json:"inputs"`
	Requirements []evaluation.CandidateFitnessRequirement `json:"requirements"`
	Tradeoffs    []evaluation.CandidateAcceptedTradeoff   `json:"tradeoffs,omitempty"`
	Arms         []evaluation.CandidateEvaluationArm      `json:"arms"`
}

// evaluateCandidate recompiles the admitted candidate deterministically,
// re-derives the frozen evaluation contract, and judges the measured arms
// through the one cross-domain evaluator. A refusal prints as evidence;
// nothing activates.
func evaluateCandidate(root, specPath string, output io.Writer) error {
	var spec candidateEvaluationSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	decision, err := runrecord.RequireDriverDecision(ctx, store, spec.Decision)
	if err != nil {
		return err
	}
	compiled, err := modelrecipe.CompileAdmittedCandidate(
		ctx, store, spec.Candidate, spec.Admission,
		composition.SameBaseCandidatePlugin{}, modelrecipe.SteeringCandidatePlugin{},
	)
	if err != nil {
		return err
	}
	contract, err := evaluation.NewCrossDomainEvaluationContract(
		decision, compiled, spec.Evaluator, spec.Inputs, spec.Requirements, spec.Tradeoffs,
	)
	if err != nil {
		return err
	}
	result, err := evaluation.EvaluateCrossDomainCandidate(contract, decision, compiled, spec.Arms)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(result, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}

// recipeDerivationSpec binds one materialization to the exact serving
// policy the derived per-arm recipes must satisfy.
type recipeDerivationSpec struct {
	Materialization artifact.ID            `json:"materialization"`
	Placement       recipe.Placement       `json:"placement"`
	Session         recipe.SessionPolicy   `json:"session"`
	Residency       recipe.ResidencyPolicy `json:"residency"`
	Tasks           []recipe.Task          `json:"tasks"`
}

// deriveRecipes projects one materialized candidate into its per-arm
// execution and evaluation recipes and prints the immutable set; it
// activates and serves nothing.
func deriveRecipes(root, specPath string, output io.Writer) error {
	var spec recipeDerivationSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	set, err := modelrecipe.DeriveCandidateRecipes(context.Background(), store, spec.Materialization, modelrecipe.CandidateRecipePolicy{
		Placement: spec.Placement, Session: spec.Session, Residency: spec.Residency, Tasks: spec.Tasks,
	})
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(set, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}

// moeCoverageSpec is one indexed router-observation evidence read; a
// summary identity additionally resolves its exact chunk.
type moeCoverageSpec struct {
	Query   runrecord.MoERouterObservationQuery `json:"query"`
	Summary artifact.ID                         `json:"summary,omitzero"`
}

// queryMoECoverage runs one auditable indexed read over MoE router
// observation evidence, optionally resolving one summary's raw chunk.
func queryMoECoverage(root, specPath string, output io.Writer) error {
	var spec moeCoverageSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	result, err := runrecord.QueryMoERouterObservationCoverage(ctx, store, spec.Query)
	if err != nil {
		return err
	}
	payload := struct {
		Result runrecord.MoERouterObservationQueryResult `json:"result"`
		Chunk  *runrecord.MoERouterObservationChunk      `json:"chunk,omitempty"`
	}{Result: result}
	if spec.Summary.Valid() {
		_, chunk, err := runrecord.RequireMoERouterObservationCoverage(ctx, store, spec.Summary)
		if err != nil {
			return err
		}
		payload.Chunk = &chunk
	}
	encoded, err := json.MarshalIndent(payload, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}

// resourceLanesSpec declares the compared workload lanes.
type resourceLanesSpec struct {
	Lanes []runrecord.ResourceFitnessLane `json:"lanes"`
}

// resourceRunsSpec names two committed runs whose resource observation
// streams form one lane: the baseline and the candidate of a
// no-regression judgment over the metrics the operator requires.
type resourceRunsSpec struct {
	Name            string                     `json:"name"`
	BaselineRun     artifact.ID                `json:"baseline_run"`
	CandidateRun    artifact.ID                `json:"candidate_run"`
	RequiredMetrics []runrecord.ResourceMetric `json:"required_metrics"`
}

// compareResourceRuns judges two runs through the resource no-regression
// owner: each run's committed observation stream is loaded at its current
// head and the lane replays through CompareResourceFitness, which refuses
// any candidate increase on a required metric. The typed comparison, or
// the exact refusal, is the verdict.
func compareResourceRuns(root, specPath string, output io.Writer) error {
	var spec resourceRunsSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	if spec.Name == "" || !spec.BaselineRun.Valid() || !spec.CandidateRun.Valid() || len(spec.RequiredMetrics) == 0 {
		return errors.New("loop: resource run comparison requires a lane name, two runs, and required metrics")
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	streams := make([]runrecord.ObservationStream, 0, 2)
	for _, run := range []artifact.ID{spec.BaselineRun, spec.CandidateRun} {
		// The complete stream: every chunk the run committed, whatever
		// its size, is the evidence under judgment.
		stream, found, err := runrecord.LoadObservationStream(ctx, store, run, runrecord.ObservationStreamBounds{
			MaxChunks: math.MaxInt, MaxRawBytes: math.MaxUint64,
		})
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("loop: run %s committed no resource observations", run)
		}
		streams = append(streams, stream)
	}
	comparison, err := runrecord.CompareResourceFitness(ctx, store, []runrecord.ResourceFitnessLane{{
		Name: spec.Name, RequiredMetrics: spec.RequiredMetrics, Baseline: streams[0], Candidate: streams[1],
	}})
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(comparison, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}

// compareResourceLanes replays a resource no-regression proof over every
// declared lane and prints the typed comparison.
func compareResourceLanes(root, specPath string, output io.Writer) error {
	var spec resourceLanesSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	if len(spec.Lanes) == 0 {
		return errors.New("loop: resource comparison requires at least one lane")
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	comparison, err := runrecord.CompareResourceFitness(context.Background(), store, spec.Lanes)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(comparison, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s\n", encoded)
	return err
}
