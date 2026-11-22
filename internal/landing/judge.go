package landing

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// JudgeCheck judges the checks of the steps ref names over the working tree at repo as the commit attempt does, by its
// own stages (verify-as-gate): the tree is taken as a landing of ref takes it, each go test a check names is listed
// (form), and each check runs with the production the change wrote restored to HEAD (undo). passed holds the walls of
// the checks the caller has just passed on the change, which are not run again and bound their runs without it; a
// check the attempt takes that the caller did not run (the trunk's, where the change's plan sets another) runs on the
// change first, as the attempt's verify runs it. It is the refusal the attempt would give, nil where the attempt would
// take the checks, and says on log what the undo run found. plan -verify ran a check once in the working folder, so a
// check that holds no test it runs or that passes without the change passed it and was refused only by an attempt:
// 18 of the last 150 refusals with a cause on 2026-11-21, each minutes to an hour of gate in.
func JudgeCheck(ctx context.Context, repo, ref string, passed map[string]time.Duration, log io.Writer) error {
	scratch, err := os.MkdirTemp("", "overgo-judge-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(scratch)
	l := &landing{Options: Options{Repo: repo, Ref: ref, Log: log}, scratch: scratch, details: map[string]string{}, passedWalls: passed}
	if err := l.parents(ctx); err != nil {
		return err
	}
	if err := l.shipSet(ctx); err != nil {
		return err
	}
	// A change to the plan alone runs its check and nothing else, as the attempt runs it (Run).
	if !l.planOnly() {
		if err := l.form(ctx); err != nil {
			return err
		}
	}
	if err := l.verify(ctx); err != nil {
		return err
	}
	fmt.Fprintf(log, "%s\n- %s\n", undoSection, strings.Join(l.undone, "\n- "))
	return nil
}
