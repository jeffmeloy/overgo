package loop

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// ReplayOutcome reports one replay: the invocation it ran under, the steps
// read from the log because their inputs were unchanged, and the steps
// executed because their inputs differed.
type ReplayOutcome struct {
	Invocation uint32
	Reused     []string
	Executed   []string
}

// ReplayExecutor runs one recorded step again and returns the artifact its
// new result is recorded as.
type ReplayExecutor func(ctx context.Context, step runrecord.GateStep) (artifact.ID, error)

// replayUnit: the durable unit of one plan row's replays.
func replayUnit(row Step) (artifact.ID, error) {
	return artifact.IdentifyBytes(artifact.KindEvidence, []byte("overgo/plan-replay/"+row.Key()))
}

// ReplayAttempt re-runs a recorded attempt of row from its receipt under a
// new invocation of the row's replay unit: each recorded step is keyed by
// its name and the inputs identity it observed, a step whose inputs are
// unchanged is read from the log, and a step whose inputs differ is
// executed and its result logged under this invocation.
func ReplayAttempt(
	ctx context.Context, store artifact.Repository, row Step, receipt runrecord.AttemptRecord, result runrecord.GateResult,
	inputs string, execute ReplayExecutor,
) (ReplayOutcome, error) {
	if ctx == nil || store == nil || execute == nil || strings.TrimSpace(inputs) == "" {
		return ReplayOutcome{}, errors.New("loop: replay requires a store, an executor and the inputs identity")
	}
	if receipt.PlanItem != row.Item || receipt.PlanStep != row.ID {
		return ReplayOutcome{}, fmt.Errorf("loop: receipt belongs to %s/%s, not %s", receipt.PlanItem, receipt.PlanStep, row.Key())
	}
	unit, err := replayUnit(row)
	if err != nil {
		return ReplayOutcome{}, err
	}
	attempt, err := runrecord.OpenDurableAttempt(ctx, store, unit)
	if err != nil {
		return ReplayOutcome{}, err
	}
	outcome := ReplayOutcome{Invocation: attempt.Invocation}
	for _, step := range result.Steps {
		key := step.Name + "@" + inputs
		if _, found, err := attempt.Lookup(ctx, store, runrecord.DurableMemo, key); err != nil {
			return outcome, err
		} else if found {
			outcome.Reused = append(outcome.Reused, step.Name)
			continue
		}
		if inputs == receipt.CodeCommit && result.ID.Valid() {
			// The receipt itself is the memo for unchanged inputs: the
			// recorded result stands and is logged under this invocation.
			if _, err := attempt.Record(ctx, store, runrecord.DurableMemo, key, result.ID); err != nil {
				return outcome, err
			}
			outcome.Reused = append(outcome.Reused, step.Name)
			continue
		}
		produced, err := execute(ctx, step)
		if err != nil {
			return outcome, fmt.Errorf("loop: replay step %s: %w", step.Name, err)
		}
		if _, err := attempt.Record(ctx, store, runrecord.DurableMemo, key, produced); err != nil {
			return outcome, err
		}
		outcome.Executed = append(outcome.Executed, step.Name)
	}
	return outcome, nil
}
