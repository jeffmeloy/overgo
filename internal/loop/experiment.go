package loop

import (
	"errors"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

type StrategyHardOutcome struct {
	TaskSucceeded     bool   `json:"task_succeeded"`
	EvidenceComplete  bool   `json:"evidence_complete"`
	GatePassed        bool   `json:"gate_passed"`
	UnsupportedClaims uint64 `json:"unsupported_claims"`
}

type StrategyExperimentCandidate struct {
	Strategy           Strategy                `json:"strategy"`
	Lease              plan.WorkLease          `json:"lease"`
	Attempt            runrecord.AttemptRecord `json:"attempt"`
	Trajectory         artifact.ID             `json:"trajectory"`
	EvaluationPlan     artifact.ID             `json:"evaluation_plan"`
	EvaluationEvidence artifact.ID             `json:"evaluation_evidence"`
	Hard               StrategyHardOutcome     `json:"hard"`
}

type StrategyExperiment struct {
	ID         artifact.ID   `json:"-"`
	Task       artifact.ID   `json:"task"`
	Baseline   string        `json:"baseline"`
	Strategies []artifact.ID `json:"strategies"`
}

type StrategyComparison struct {
	ID              artifact.ID   `json:"-"`
	Experiment      artifact.ID   `json:"experiment"`
	Ranked          []artifact.ID `json:"ranked"`
	Winner          *artifact.ID  `json:"winner,omitempty"`
	Counterexamples []artifact.ID `json:"counterexamples,omitempty"`
	Evidence        []artifact.ID `json:"evidence"`
}

// CompareStrategyExperiment validates isolated executions and ranks hard facts
// before cost. It consumes existing lease, attempt, trajectory, and evaluation
// records and schedules nothing.
func CompareStrategyExperiment(task recipe.AgentTaskContract, baseline string, candidates []StrategyExperimentCandidate) (StrategyExperiment, StrategyComparison, error) {
	if err := task.ValidateIdentity(); err != nil {
		return StrategyExperiment{}, StrategyComparison{}, err
	}
	if baseline == "" || len(candidates) < int(artifact.SecondDocumentVersion) {
		return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: strategy experiment requires a baseline and competing candidates")
	}
	worktrees := map[string]bool{}
	strategies := make([]artifact.ID, len(candidates))
	for index, candidate := range candidates {
		if err := candidate.Strategy.ValidateIdentity(); err != nil {
			return StrategyExperiment{}, StrategyComparison{}, err
		}
		if err := candidate.Lease.ValidateIdentity(); err != nil {
			return StrategyExperiment{}, StrategyComparison{}, err
		}
		if candidate.Lease.TargetHead != baseline || worktrees[strings.ToLower(candidate.Lease.Worktree)] || candidate.Attempt.StrategyID != candidate.Strategy.ID ||
			candidate.Trajectory.Kind() != artifact.KindEvidence || candidate.EvaluationPlan.Kind() != artifact.KindProfile || candidate.EvaluationEvidence.Kind() != artifact.KindEvidence {
			return StrategyExperiment{}, StrategyComparison{}, errors.New("loop: strategy candidate authority or isolation differs")
		}
		worktrees[strings.ToLower(candidate.Lease.Worktree)] = true
		strategies[index] = candidate.Strategy.ID
	}
	slices.SortFunc(strategies, artifact.CompareID)
	experimentID, err := artifact.JSONID(artifact.KindEvidence, struct {
		Task       artifact.ID   `json:"task"`
		Baseline   string        `json:"baseline"`
		Strategies []artifact.ID `json:"strategies"`
	}{task.ID, baseline, strategies})
	if err != nil {
		return StrategyExperiment{}, StrategyComparison{}, err
	}
	experiment := StrategyExperiment{ID: experimentID, Task: task.ID, Baseline: baseline, Strategies: strategies}
	ordered := slices.Clone(candidates)
	sort.SliceStable(ordered, func(i, j int) bool { return strategyCandidateBetter(ordered[i], ordered[j]) })
	comparison := StrategyComparison{Experiment: experiment.ID, Ranked: make([]artifact.ID, len(ordered))}
	for index, candidate := range ordered {
		comparison.Ranked[index] = candidate.Strategy.ID
		comparison.Evidence = append(comparison.Evidence, candidate.Attempt.ID, candidate.Trajectory, candidate.EvaluationPlan, candidate.EvaluationEvidence)
		if !candidate.Hard.TaskSucceeded || !candidate.Hard.EvidenceComplete || !candidate.Hard.GatePassed {
			comparison.Counterexamples = append(comparison.Counterexamples, candidate.Attempt.ID)
		}
	}
	best := ordered[0]
	if best.Hard.TaskSucceeded && best.Hard.EvidenceComplete && best.Hard.GatePassed && best.Hard.UnsupportedClaims == 0 && strategyCandidateBetter(best, ordered[1]) {
		winner := best.Strategy.ID
		comparison.Winner = &winner
	}
	comparison.ID, err = artifact.JSONID(artifact.KindEvidence, comparison)
	return experiment, comparison, err
}

func strategyCandidateBetter(left, right StrategyExperimentCandidate) bool {
	leftHard := []bool{left.Hard.TaskSucceeded, left.Hard.EvidenceComplete, left.Hard.GatePassed}
	rightHard := []bool{right.Hard.TaskSucceeded, right.Hard.EvidenceComplete, right.Hard.GatePassed}
	for index := range leftHard {
		if leftHard[index] != rightHard[index] {
			return leftHard[index]
		}
	}
	if left.Hard.UnsupportedClaims != right.Hard.UnsupportedClaims {
		return left.Hard.UnsupportedClaims < right.Hard.UnsupportedClaims
	}
	if left.Attempt.CostUnits != right.Attempt.CostUnits {
		return left.Attempt.CostUnits < right.Attempt.CostUnits
	}
	return artifact.CompareID(left.Strategy.ID, right.Strategy.ID) < 0
}
