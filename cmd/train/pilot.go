package main

import (
	"context"
	"fmt"
	"io"

	"overgo/internal/artifact"
	"overgo/internal/modelintake"
	"overgo/internal/overgodb"
	"overgo/internal/trainingworkflow"
)

// runPilot runs the instruction pilot under a published acceptance
// (trainingworkflow.RunInstructionPilot) and writes what it came to: what the
// base passed of the selection groups, each arm's checkpoints with what the
// session that reached each did and what it passed of the selection groups,
// the checkpoint frozen for each arm, each comparison's verdict beside the
// confirmation and the charge it is recorded under, and whether the reward arm
// is adopted, with the reason when it is not. Not adopting is the pilot's
// answer and no failure of the command. The answers it draws are bound to the
// commit it was built at, so it refuses a tree with uncommitted source.
func runPilot(ctx context.Context, store *overgodb.Store, acceptanceText, model, out string, output io.Writer) error {
	acceptance, err := artifact.ParseID(acceptanceText)
	if err != nil || model == "" || out == "" {
		return fmt.Errorf("train: -pilot names an instruction acceptance, as training-objective -pilot wrote it, and needs -model, the base it was fixed over, and -out: %w", err)
	}
	code, err := modelintake.CleanRevision(ctx)
	if err != nil {
		return err
	}
	result, err := trainingworkflow.RunInstructionPilot(ctx, store, trainingworkflow.PilotRun{Acceptance: acceptance, Model: model, Out: out, Code: code, Device: denseDevice})
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "instruction pilot run, built at commit %s\nacceptance: %s\nbase: passed %d of %d selection groups\n", code, acceptance, result.BaseSelected, result.Selection)
	for _, arm := range result.Arms {
		for _, checkpoint := range arm.Checkpoints {
			trace := checkpoint.Trace
			fmt.Fprintf(output, "%s after %d groups: passed %d of %d selection groups; its session sampled %d groups, stepped on %d and passed %d, sampled %d tokens and trained on %d, %d answers at the output limit, %.0f s sampling, %.0f s scoring, %.0f s stepping, shift %.4f\n",
				arm.Name, checkpoint.Groups, checkpoint.Selected, result.Selection, trace.Sampled, trace.Updated, trace.Skipped, trace.SampledTokens, trace.TrainedTokens, trace.Truncated,
				trace.Generation.Seconds(), trace.Scoring.Seconds(), trace.Update.Seconds(), trace.Shift)
		}
		frozen := arm.Checkpoints[arm.Frozen]
		fmt.Fprintf(output, "%s: frozen after %d groups, %s\n", arm.Name, frozen.Groups, frozen.Directory)
	}
	for _, comparison := range result.Comparisons {
		fmt.Fprintf(output, "comparison %q: %s; confirmation %s, charge %s, %d left\n", comparison.Name, comparison.Confirmation.Verdict, comparison.Confirmation.ID, comparison.Confirmation.Charge, comparison.Left)
	}
	_, err = fmt.Fprintf(output, "adopted: %t %s\n", result.Adopted, result.Reason)
	return err
}
