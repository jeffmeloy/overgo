package main

import (
	"context"
	"fmt"
	"io"

	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
)

// writeVerdicts lists the held-out verdicts the store held when its journal
// stood at a sequence, oldest first: what each training run was judged on and
// what its base and trained models scored. It is what a proposer of a model
// change may be shown of what was tried before it, so the listing ends with
// the sequence it was read through and the bytes of verdict documents it
// names, and a listing over the byte budget is refused whole. A verdict whose
// reports cannot be reconstructed is listed as that, with the reason.
func writeVerdicts(output io.Writer, repository string, through, budget uint64) error {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	verdicts, bytes, err := evaluation.PastHeldoutVerdicts(ctx, store, through, budget)
	if err != nil {
		return err
	}
	for _, verdict := range verdicts {
		facts, err := evaluation.ReadHeldoutVerdict(ctx, store, verdict)
		if err != nil {
			fmt.Fprintf(output, "verdict %s cannot be reconstructed: %v\n", verdict, err)
			continue
		}
		fmt.Fprintf(output, "verdict %s objective %s %s base %g trained %g checkpoint %s session %s\n",
			verdict, facts.Objective, facts.Metric, facts.BaselineValue, facts.CandidateValue, facts.Trained, facts.Observation)
	}
	if through == 0 {
		_, through = store.Head()
	}
	fmt.Fprintf(output, "%d held-out verdict(s) through sequence %d; %d bytes of verdict documents\n", len(verdicts), through, bytes)
	return nil
}
