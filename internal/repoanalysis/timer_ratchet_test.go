package repoanalysis

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestNoFixedTimers holds the owner's rule that no wait is bound to a
// duration the source fixes: the census classifies literal, unit, constant
// and constant-initialised durations as fixed, a constant a sibling file
// declares included, and parameters, fields and flags as the caller's
// declaration; a duration under a testing/synctest bubble advances a
// simulated clock and is no wall wait; any fixed timer is refused, and the
// live tree holds none.
func TestNoFixedTimers(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("fixed.go", `package probe
import ("context"; "time")
const settle = 2 * time.Minute
var grace = 15 * time.Second
func Fixed(ctx context.Context) {
	_, cancel := context.WithTimeout(ctx, 5*time.Second)
	cancel()
	_, cancel = context.WithTimeoutCause(ctx, settle, nil)
	cancel()
	time.Sleep(grace)
	<-time.After(time.Duration(3) * time.Millisecond)
	_ = time.Tick(time.Millisecond)
	time.Sleep(sharedBudget)
	_ = time.NewTimer(sharedRetry)
}
`)
	write("shared.go", `package probe
import "time"
const sharedBudget = 10 * time.Minute
var sharedRetry = sharedBudget / 100
`)
	write("bubble_test.go", `package probe
import ("testing"; "testing/synctest"; "time")
func TestBubble(t *testing.T) {
	t.Parallel()
	synctest.Test(t, func(t *testing.T) {
		time.Sleep(time.Second)
		<-time.After(time.Minute)
	})
}
`)
	write("declared.go", `package probe
import ("context"; "flag"; "time")
type options struct{ Budget time.Duration }
var budgetFlag = flag.Duration("budget", 0, "")
func Declared(ctx context.Context, budget time.Duration, o options) {
	_, cancel := context.WithTimeout(ctx, budget)
	cancel()
	_, cancel = context.WithTimeoutCause(ctx, o.Budget, nil)
	cancel()
	_, cancel = context.WithTimeout(ctx, *budgetFlag)
	cancel()
	derived := budget / 2
	time.Sleep(derived)
}
`)
	snapshot, err := DiscoverGo(root, ".")
	if err != nil {
		t.Fatal(err)
	}
	census, err := FixedTimerCensus(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(census, []FixedTimer{{File: "fixed.go", Function: "Fixed", Count: 7}}) {
		t.Fatalf("fixture census = %+v", census)
	}
	if err := RefuseFixedTimers(census); err == nil || !strings.Contains(err.Error(), "fixed.go Fixed waits 7") {
		t.Fatalf("a fixed timer was admitted: %v", err)
	}
	if err := RefuseFixedTimers(nil); err != nil {
		t.Fatalf("a source with no fixed timer was refused: %v", err)
	}

	repository, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	live, err := DiscoverGo(repository, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	liveCensus, err := FixedTimerCensus(live)
	if err != nil {
		t.Fatal(err)
	}
	if err := RefuseFixedTimers(liveCensus); err != nil {
		t.Fatal(err)
	}
}
