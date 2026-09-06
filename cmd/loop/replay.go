package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
)

const (
	replayOutputMediaType = "text/plain"
	replayOutputSchema    = "overgo/replay-step-output/v1"
)

// replayAttempt re-runs the row's latest recorded attempt from its receipt:
// the current HEAD is the inputs identity, a step with unchanged inputs is
// read from the receipt, and a step with changed inputs runs the row's
// verifier and records its output.
func replayAttempt(root, reference string, output io.Writer) error {
	item, stepID, bound := strings.Cut(reference, "/")
	if !bound || item == "" || stepID == "" {
		return errors.New("usage: loop -replay-attempt <item>/<step>")
	}
	row := loop.Step{Item: item, ID: stepID}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	history, err := runrecord.LoadAttemptHistory(ctx, store, runrecord.AttemptFilter{PlanItem: item})
	if err != nil {
		return err
	}
	var latest *runrecord.AttemptRecord
	for index := range history.Attempts {
		if history.Attempts[index].PlanStep == stepID {
			latest = &history.Attempts[index]
		}
	}
	if latest == nil {
		return fmt.Errorf("loop: row %s has no recorded attempt to replay", reference)
	}
	result, err := runrecord.RequireGateResult(ctx, store, latest.Result)
	if err != nil {
		return err
	}
	head, err := runTool("git", "rev-parse", "HEAD")
	if err != nil {
		return fmt.Errorf("loop: resolve HEAD: %w", err)
	}
	document, err := plan.Load(filepath.FromSlash(plan.Path))
	if err != nil {
		return err
	}
	verify := ""
	for _, candidate := range document.Items {
		if candidate.ID != item {
			continue
		}
		for _, candidateStep := range candidate.Steps {
			if candidateStep.ID == stepID {
				verify = candidateStep.Verify
			}
		}
	}
	execute := func(_ context.Context, step runrecord.GateStep) (artifact.ID, error) {
		if verify == "" {
			return artifact.ID{}, fmt.Errorf("loop: row %s is not retained; its %s step cannot be re-executed", reference, step.Name)
		}
		out, runErr := runTool("sh", "-c", verify)
		text := fmt.Sprintf("step=%s verify=%s exit-error=%v\n%s", step.Name, verify, runErr, tailOf(out, clioptions.DiagnosticTailBytes))
		content, err := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: replayOutputMediaType, Schema: replayOutputSchema}.ContentBytes([]byte(text))
		if err != nil {
			return artifact.ID{}, err
		}
		batch, err := artifact.NewDocumentBatch("replay/output/"+content.Descriptor.ID.String(), []artifact.Content{content}, nil, nil)
		if err != nil {
			return artifact.ID{}, err
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			return artifact.ID{}, err
		}
		return content.Descriptor.ID, nil
	}
	outcome, err := loop.ReplayAttempt(ctx, store, row, *latest, result, strings.TrimSpace(head), execute)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "replayed %s attempt %.8s under invocation %d: reused=%s executed=%s\n",
		reference, latest.CodeCommit, outcome.Invocation, strings.Join(outcome.Reused, ","), strings.Join(outcome.Executed, ","))
	return err
}
