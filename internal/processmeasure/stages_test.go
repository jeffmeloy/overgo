package processmeasure

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestStagesDivideARequest holds what makes a request's parts a division of
// it: a part's wall leaves out what a part begun inside it took, a part gone
// through twice is one line, and the time left over is the request's wall
// less the parts'.
func TestStagesDivideARequest(t *testing.T) {
	t.Parallel()
	ctx, stages := WithStages(t.Context())
	// work passes time the clock can read: it reads the clock itself.
	work := func() {
		for range 20000 {
			if _, err := Counter(); err != nil {
				t.Fatal(err)
			}
		}
	}
	clock := NewStopwatch()
	outer := Stage(ctx, "outer")
	for range 2 {
		inner := Stage(ctx, "inner")
		work()
		inner()
	}
	work()
	outer()
	elapsed, err := clock.Elapsed()
	if err != nil {
		t.Fatal(err)
	}
	var report strings.Builder
	unnamed, err := stages.Report(&report, time.Duration(elapsed))
	if err != nil {
		t.Fatal(err)
	}
	if len(stages.parts) != 2 || stages.parts[0].name != "inner" || stages.parts[0].times != 2 || stages.parts[1].name != "outer" || stages.parts[1].times != 1 {
		t.Fatalf("the parts are %+v", stages.parts)
	}
	inner, own := stages.parts[0].wall, stages.parts[1].wall
	if inner <= 0 || own <= 0 {
		t.Errorf("the inner part took %v twice over and the outer %v of its own", inner, own)
	}
	// An outer wall that counted the inner part's time would pass the request's.
	if unnamed < 0 || unnamed != time.Duration(elapsed)-inner-own {
		t.Errorf("%v of %v is left unnamed by parts of %v and %v", unnamed, time.Duration(elapsed), inner, own)
	}
	if !strings.Contains(report.String(), "inner") || !strings.Contains(report.String(), "x2") {
		t.Errorf("the report is\n%s", report.String())
	}
}

// TestStagesRefuseOverlappingParts: parts that overlap divide nothing, and
// the ledger says so in place of reporting walls that count a moment twice.
func TestStagesRefuseOverlappingParts(t *testing.T) {
	t.Parallel()
	ctx, stages := WithStages(t.Context())
	first, second := Stage(ctx, "first"), Stage(ctx, "second")
	first()
	second()
	if _, err := stages.Report(&strings.Builder{}, time.Second); err == nil {
		t.Fatal("overlapping parts were reported as a division")
	}
	// A request that carries no ledger times nothing.
	Stage(context.Background(), "unmeasured")()
}
