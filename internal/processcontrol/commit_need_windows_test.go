//go:build windows

package processcontrol

import (
	"errors"
	"io"
	"strings"
	"testing"
)

// TestATakeOfCommitSaysTheMachinesFigures holds TakeCommit to the machine it runs on: a take that succeeds is all it
// does; a take of more than the machine could give, overgo's own processes' commit given back, fails at once as a
// shortfall, never waiting; a take that fails with the commit left stands with its own error and the figures.
func TestATakeOfCommitSaysTheMachinesFigures(t *testing.T) {
	t.Parallel()
	taken := errors.New("the take failed")
	takes := 0
	if err := TakeCommit(1, "a test", func() error { takes++; return nil }, io.Discard); err != nil || takes != 1 {
		t.Fatalf("a take that succeeds: %v after %d takes", err, takes)
	}
	err := TakeCommit(1<<62, "a test", func() error { return taken }, io.Discard)
	if !errors.Is(err, ErrCommitShort) || !errors.Is(err, taken) || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("a take past the machine: %v, want a shortfall naming the limit beside the take's own error", err)
	}
	err = TakeCommit(1, "a test", func() error { return taken }, io.Discard)
	if errors.Is(err, ErrCommitShort) || !errors.Is(err, taken) || !strings.Contains(err.Error(), "left of the machine's") {
		t.Fatalf("a failed take with the commit left: %v, want its own error with the figures", err)
	}
}
