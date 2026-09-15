package repoanalysis

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/jsonfile"
)

// TestNoFixedTimers holds the owner's rule that no wait is bound to a
// duration the source fixes: the census classifies literal, unit, constant
// and constant-initialised durations as fixed and parameters, fields and
// flags as the caller's declaration; over the live tree the census never
// exceeds the reviewed baseline and the baseline never lists a timer the
// source has already lost, so the count only falls.
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
	if !slices.Equal(census, []FixedTimer{{File: "fixed.go", Function: "Fixed", Count: 5}}) {
		t.Fatalf("fixture census = %+v", census)
	}
	baseline := FixedTimerBaseline{Version: 1, Timers: []FixedTimer{{File: "fixed.go", Function: "Fixed", Count: 5}}}
	if err := AdmitFixedTimers(baseline, census); err != nil {
		t.Fatalf("exact baseline refused: %v", err)
	}
	if err := AdmitFixedTimers(FixedTimerBaseline{Version: 1}, census); err == nil {
		t.Fatal("a new fixed timer was admitted")
	}
	if err := AdmitFixedTimers(baseline, nil); err == nil {
		t.Fatal("a stale baseline entry was admitted")
	}
	if err := AdmitFixedTimers(baseline, []FixedTimer{{File: "fixed.go", Function: "Fixed", Count: 4}}); err == nil {
		t.Fatal("a removal without lowering the baseline was admitted")
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
	var reviewed FixedTimerBaseline
	if err := jsonfile.DecodeStrict(filepath.Join(repository, filepath.FromSlash(FixedTimerBaselineFile)), &reviewed); err != nil {
		t.Fatal(err)
	}
	if err := AdmitFixedTimers(reviewed, liveCensus); err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, timer := range liveCensus {
		total += timer.Count
	}
	t.Logf("fixed timers remaining: %d sites in %d functions", total, len(liveCensus))
}
