package loop

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/worklease"
)

// StrategyExperimentCandidate names only stored comparison authorities. The
// comparison loads their typed documents; callers cannot attach replacement
// records or derived outcome flags after those documents were identified.
type StrategyExperimentCandidate struct {
	Strategy           artifact.ID `json:"strategy"`
	Lease              artifact.ID `json:"lease"`
	Attempt            artifact.ID `json:"attempt"`
	EvaluationEvidence artifact.ID `json:"evaluation_evidence"`
}

// StrategyExperiment identifies one comparison of competing strategy
// executions over the same task and baseline commit.
type StrategyExperiment struct {
	ID         artifact.ID                  `json:"-"`
	Task       artifact.ID                  `json:"task"`
	Baseline   string                       `json:"baseline"`
	Candidates []StrategyComparisonEvidence `json:"candidates"`
}

// StrategyComparisonEvidence keeps one stored execution's authorities
// grouped. Trajectory and evaluation plan are derived from the stored attempt
// and evidence rather than repeated in the candidate request.
type StrategyComparisonEvidence struct {
	Strategy           artifact.ID `json:"strategy"`
	Lease              artifact.ID `json:"lease"`
	Attempt            artifact.ID `json:"attempt"`
	Trajectory         artifact.ID `json:"trajectory"`
	EvaluationPlan     artifact.ID `json:"evaluation_plan"`
	EvaluationEvidence artifact.ID `json:"evaluation_evidence"`
}

// StrategyComparison presents candidates in deterministic identity order.
// Fitness records are directed dominance proofs; presentation order never
// grants one strategy an advantage.
type StrategyComparison struct {
	ID              artifact.ID                  `json:"-"`
	Experiment      artifact.ID                  `json:"experiment"`
	Ranked          []artifact.ID                `json:"ranked"`
	Winner          *artifact.ID                 `json:"winner,omitempty"`
	Counterexamples []artifact.ID                `json:"counterexamples,omitempty"`
	Evidence        []StrategyComparisonEvidence `json:"evidence"`
	Fitness         []artifact.ID                `json:"fitness,omitempty"`
}

type resolvedStrategyCandidate struct {
	strategy   Strategy
	lease      worklease.Lease
	attempt    runrecord.AttemptRecord
	trajectory runrecord.InteractionTrace
	evaluation evaluation.EvaluationEvidence
}

type directedStrategyPair struct {
	baseline  artifact.ID
	candidate artifact.ID
}

func (candidate resolvedStrategyCandidate) evidence() StrategyComparisonEvidence {
	return StrategyComparisonEvidence{
		Strategy: candidate.strategy.ID, Lease: candidate.lease.ID, Attempt: candidate.attempt.ID,
		Trajectory: candidate.trajectory.ID, EvaluationPlan: candidate.evaluation.Plan,
		EvaluationEvidence: candidate.evaluation.ID,
	}
}

func (candidate resolvedStrategyCandidate) matches(endpoint evaluation.ImprovementFitnessEndpoint) bool {
	return endpoint.Strategy == candidate.strategy.ID && endpoint.Attempt == candidate.attempt.ID &&
		endpoint.Trajectory == candidate.trajectory.ID && endpoint.Evidence == candidate.evaluation.ID &&
		endpoint.Plan == candidate.evaluation.Plan && endpoint.Run == candidate.evaluation.Run &&
		endpoint.Evaluation == candidate.evaluation.Evaluation
}

// CompareStrategyExperiment loads exact execution authorities and admits only
// persisted pairwise fitness as evidence of improvement. A winner exists only
// when one strategy has an improved directed edge from every peer. Missing or
// incomparable edges produce a valid no-winner comparison; malformed or
// foreign edges are refused.
func CompareStrategyExperiment(
	ctx context.Context,
	reader artifact.Reader,
	task recipe.AgentTaskContract,
	baseline string,
	candidates []StrategyExperimentCandidate,
	fitnessIDs []artifact.ID,
) (StrategyExperiment, StrategyComparison, error) {
	if ctx == nil || reader == nil {
		return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: strategy experiment repository is absent")
	}
	if err := task.ValidateIdentity(); err != nil {
		return StrategyExperiment{}, StrategyComparison{}, err
	}
	storedTask, err := recipe.RequireAgentTaskContract(ctx, reader, task.ID)
	if err != nil || storedTask.ID != task.ID {
		return StrategyExperiment{}, StrategyComparison{}, errors.Join(err, errors.New("loop: strategy experiment task authority differs"))
	}
	if baseline == "" || len(candidates) < int(artifact.SecondDocumentVersion) ||
		len(candidates) > runrecord.MaximumAttemptPopulation || len(fitnessIDs) > runrecord.MaximumAttemptPopulation {
		return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: strategy experiment requires a baseline and competing candidates")
	}

	resolved := make(map[artifact.ID]resolvedStrategyCandidate, len(candidates))
	worktrees := make(map[string]struct{}, len(candidates))
	leaseIDs := make(map[artifact.ID]struct{}, len(candidates))
	attemptIDs := make(map[artifact.ID]struct{}, len(candidates))
	trajectoryIDs := make(map[artifact.ID]struct{}, len(candidates))
	evidenceIDs := make(map[artifact.ID]struct{}, len(candidates))
	for _, request := range candidates {
		candidate, resolveErr := resolveStrategyCandidate(ctx, reader, task, baseline, request)
		if resolveErr != nil {
			return StrategyExperiment{}, StrategyComparison{}, resolveErr
		}
		worktree, _ := worklease.ResolveWorkspaceClaims(candidate.lease.Worktree, nil, nil)
		worktreeIdentity := worklease.WorktreeAlias(worktree)
		_, duplicateStrategy := resolved[candidate.strategy.ID]
		_, duplicateWorktree := worktrees[worktreeIdentity]
		_, duplicateLease := leaseIDs[candidate.lease.ID]
		_, duplicateAttempt := attemptIDs[candidate.attempt.ID]
		_, duplicateTrajectory := trajectoryIDs[candidate.trajectory.ID]
		_, duplicateEvidence := evidenceIDs[candidate.evaluation.ID]
		if duplicateStrategy || duplicateWorktree || duplicateLease || duplicateAttempt || duplicateTrajectory || duplicateEvidence {
			return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: strategy candidate authority or isolation is reused")
		}
		resolved[candidate.strategy.ID] = candidate
		worktrees[worktreeIdentity] = struct{}{}
		leaseIDs[candidate.lease.ID] = struct{}{}
		attemptIDs[candidate.attempt.ID] = struct{}{}
		trajectoryIDs[candidate.trajectory.ID] = struct{}{}
		evidenceIDs[candidate.evaluation.ID] = struct{}{}
	}

	strategies := make([]artifact.ID, 0, len(resolved))
	for strategy := range resolved {
		strategies = append(strategies, strategy)
	}
	slices.SortFunc(strategies, artifact.CompareID)
	experiment := StrategyExperiment{
		Task: task.ID, Baseline: baseline, Candidates: make([]StrategyComparisonEvidence, len(strategies)),
	}
	for index, strategy := range strategies {
		experiment.Candidates[index] = resolved[strategy].evidence()
	}
	experiment.ID, err = artifact.JSONID(artifact.KindEvidence, experiment)
	if err != nil {
		return StrategyExperiment{}, StrategyComparison{}, err
	}

	comparison := StrategyComparison{
		Experiment: experiment.ID, Ranked: slices.Clone(strategies),
		Evidence: slices.Clone(experiment.Candidates),
	}
	for _, strategy := range strategies {
		candidate := resolved[strategy]
		if candidate.attempt.Outcome != runrecord.OutcomeSucceeded || candidate.trajectory.Terminal != runrecord.OutcomeSucceeded {
			comparison.Counterexamples = append(comparison.Counterexamples, candidate.attempt.ID)
		}
	}

	dominates := make(map[artifact.ID]map[artifact.ID]struct{}, len(strategies))
	seenFitness := make(map[artifact.ID]struct{}, len(fitnessIDs))
	seenPairs := make(map[directedStrategyPair]struct{}, len(fitnessIDs))
	for _, fitnessID := range fitnessIDs {
		if fitnessID.Kind() != artifact.KindEvidence {
			return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: invalid improvement fitness identity")
		}
		if _, duplicate := seenFitness[fitnessID]; duplicate {
			return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: duplicate improvement fitness identity")
		}
		fitness, requireErr := evaluation.RequireImprovementFitness(ctx, reader, fitnessID)
		if requireErr != nil || fitness.ID != fitnessID {
			return StrategyExperiment{}, StrategyComparison{}, errors.Join(requireErr, errors.New("loop: improvement fitness authority differs"))
		}
		baselineCandidate, baselineFound := resolved[fitness.Baseline.Strategy]
		improvedCandidate, candidateFound := resolved[fitness.Candidate.Strategy]
		if !baselineFound || !candidateFound || fitness.Baseline.Strategy == fitness.Candidate.Strategy ||
			!baselineCandidate.matches(fitness.Baseline) || !improvedCandidate.matches(fitness.Candidate) {
			return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: improvement fitness endpoints differ from experiment candidates")
		}
		pair := directedStrategyPair{baseline: fitness.Baseline.Strategy, candidate: fitness.Candidate.Strategy}
		if _, duplicate := seenPairs[pair]; duplicate {
			return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: duplicate directed improvement fitness")
		}
		seenFitness[fitnessID] = struct{}{}
		seenPairs[pair] = struct{}{}
		comparison.Fitness = append(comparison.Fitness, fitnessID)
		if fitness.Improved {
			if dominates[fitness.Candidate.Strategy] == nil {
				dominates[fitness.Candidate.Strategy] = make(map[artifact.ID]struct{})
			}
			dominates[fitness.Candidate.Strategy][fitness.Baseline.Strategy] = struct{}{}
		}
	}
	slices.SortFunc(comparison.Fitness, artifact.CompareID)

	var winners []artifact.ID
	for _, strategy := range strategies {
		complete := true
		for _, peer := range strategies {
			if peer == strategy {
				continue
			}
			if _, found := dominates[strategy][peer]; !found {
				complete = false
				break
			}
		}
		if complete {
			winners = append(winners, strategy)
		}
	}
	if len(winners) == 1 {
		winner := winners[0]
		comparison.Winner = &winner
	}
	comparison.ID, err = artifact.JSONID(artifact.KindEvidence, comparison)
	return experiment, comparison, err
}

func resolveStrategyCandidate(
	ctx context.Context,
	reader artifact.Reader,
	task recipe.AgentTaskContract,
	baseline string,
	request StrategyExperimentCandidate,
) (resolvedStrategyCandidate, error) {
	if request.Strategy.Kind() != artifact.KindProfile || request.Lease.Kind() != artifact.KindEvidence ||
		request.Attempt.Kind() != artifact.KindEvidence || request.EvaluationEvidence.Kind() != artifact.KindEvidence {
		return resolvedStrategyCandidate{}, errors.New("loop: invalid strategy candidate identity")
	}
	strategy, err := RequireStrategy(ctx, reader, request.Strategy)
	if err != nil {
		return resolvedStrategyCandidate{}, err
	}
	worker, err := recipe.RequireAgentDefinition(ctx, reader, strategy.Worker)
	if err != nil {
		return resolvedStrategyCandidate{}, err
	}
	modelRecipe, err := recipe.RequireDefinition(ctx, reader, strategy.ModelRecipe)
	if err != nil {
		return resolvedStrategyCandidate{}, err
	}
	lease, found, err := worklease.Read(ctx, reader, request.Lease)
	if err != nil || !found || lease.ID != request.Lease {
		return resolvedStrategyCandidate{}, errors.Join(err, errors.New("loop: strategy candidate lease is absent or incompatible"))
	}
	attempt, err := runrecord.RequireAttemptRecord(ctx, reader, request.Attempt)
	if err != nil {
		return resolvedStrategyCandidate{}, err
	}
	if attempt.CodeCommit != baseline {
		return resolvedStrategyCandidate{}, errors.New("loop: strategy candidate attempt commit differs from experiment baseline")
	}
	if attempt.WorkLease != lease.ID {
		return resolvedStrategyCandidate{}, errors.New("loop: strategy candidate attempt work lease differs")
	}
	gate, err := runrecord.RequireGateResult(ctx, reader, attempt.Result)
	if err != nil {
		return resolvedStrategyCandidate{}, err
	}
	trajectory, err := runrecord.RequireInteractionTrace(ctx, reader, attempt.Trajectory)
	if err != nil {
		return resolvedStrategyCandidate{}, err
	}
	evidence, err := evaluation.RequireEvaluationEvidence(ctx, reader, request.EvaluationEvidence)
	if err != nil {
		return resolvedStrategyCandidate{}, err
	}
	planContent, found, err := artifact.ReadContent(ctx, reader, evidence.Plan)
	if err != nil || !found {
		return resolvedStrategyCandidate{}, errors.Join(err, errors.New("loop: strategy evaluation plan content is absent"))
	}
	planContract := artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: evaluation.EvaluationPlanMediaType, Schema: evaluation.EvaluationPlanSchema,
	}
	if err := planContract.ValidateContent(planContent, evidence.Plan); err != nil {
		return resolvedStrategyCandidate{}, err
	}
	evaluationPlan, err := evaluation.ParsePlan(planContent.Data)
	if err != nil || evaluationPlan.Identity() != evidence.Plan {
		return resolvedStrategyCandidate{}, errors.Join(err, errors.New("loop: strategy evaluation plan identity differs"))
	}
	agentAuthorities, agentPlan := evaluationPlan.AgentTrajectoryAuthorities()
	if strategy.ID != request.Strategy || task.Agent != strategy.Worker ||
		worker.Prompt != strategy.Prompt || worker.ModelRecipe != strategy.ModelRecipe ||
		!slices.Equal(worker.Policies, strategy.Policies) || modelRecipe.Model != trajectory.Model ||
		lease.TargetHead != baseline ||
		attempt.ID != request.Attempt || attempt.StrategyID != strategy.ID || attempt.TaskContract != task.ID ||
		attempt.Trajectory != trajectory.ID || trajectory.Strategy != strategy.ID || trajectory.TaskContract != task.ID ||
		trajectory.Recipe != strategy.ModelRecipe || trajectory.Terminal != attempt.Outcome ||
		gate.Recipe != attempt.Recipe || gate.Environment != attempt.Environment || gate.CodeCommit != attempt.CodeCommit ||
		gate.Outcome != attempt.Outcome || gate.Failure != attempt.Failure ||
		evidence.ID != request.EvaluationEvidence || evidence.CodeCommit != attempt.CodeCommit ||
		evidence.Environment != attempt.Environment || !agentPlan || len(agentAuthorities.Trajectories) != 1 ||
		agentAuthorities.Trajectories[0] != trajectory.ID {
		return resolvedStrategyCandidate{}, errors.New("loop: strategy candidate authority or isolation differs")
	}
	return resolvedStrategyCandidate{
		strategy: strategy, lease: lease, attempt: attempt, trajectory: trajectory, evaluation: evidence,
	}, nil
}
