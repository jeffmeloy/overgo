package processcontrol

import (
	"errors"
	"strings"
	"testing"
)

// TestACommitShortfallWaitsOrFailsOnTheMachinesFigures holds the decision a process makes on the commit it needs to
// the machine's own figures: enough left goes on; a shortfall overgo's own other processes hold waits for them; a
// shortfall beyond what they hold, or a need past the limit, fails at once, its error naming the commit needed, the
// commit left and the limit. The figures are Flash-Next's open on 2026-11-24 (flashnext-memory/reserve).
func TestACommitShortfallWaitsOrFailsOnTheMachinesFigures(t *testing.T) {
	t.Parallel()
	const gib = uint64(1) << 30
	cases := []struct {
		name    string
		need    CommitNeed
		wait    bool
		fails   bool
		figures []string
	}{
		{name: "enough left", need: CommitNeed{Needed: 116 * gib, Left: 120 * gib, Limit: 157 * gib, Own: 0}},
		{name: "the shortfall is overgo's own", need: CommitNeed{Needed: 116 * gib, Left: 100 * gib, Limit: 157 * gib, Own: 20 * gib}, wait: true},
		{name: "the shortfall is the rest of the machine's", need: CommitNeed{Needed: 116 * gib, Left: 100 * gib, Limit: 157 * gib, Own: 10 * gib},
			fails: true, figures: []string{"needs 116.0 GiB", "100.0 GiB left", "157.0 GiB limit", "holding 10.0 GiB"}},
		{name: "the need is past the limit", need: CommitNeed{Needed: 116 * gib, Left: 30 * gib, Limit: 99 * gib, Own: 90 * gib},
			fails: true, figures: []string{"needs 116.0 GiB", "99.0 GiB limit", "30.0 GiB left"}},
	}
	for _, each := range cases {
		wait, err := each.need.Decide()
		if wait != each.wait || (err != nil) != each.fails || each.fails && !errors.Is(err, ErrCommitShort) {
			t.Fatalf("%s: wait %t, error %v; want wait %t, a shortfall %t", each.name, wait, err, each.wait, each.fails)
		}
		for _, figure := range each.figures {
			if !strings.Contains(err.Error(), figure) {
				t.Fatalf("%s: the error %q does not name %q", each.name, err, figure)
			}
		}
	}
}
