package evaluation

import "testing"

func TestProgressTrackerReportsEveryCaseToTheBoundSink(t *testing.T) {
	var seen []Progress
	ctx := WithProgress(t.Context(), func(value Progress) { seen = append(seen, value) })
	tracker := trackProgress(ctx, "store/bbh", 3)
	tracker.hit(true)
	tracker.hit(false)
	tracker.advance()
	if len(seen) != 3 {
		t.Fatalf("sink saw %d reports, want 3", len(seen))
	}
	if seen[0].Suite != "store/bbh" || seen[0].Done != 1 || seen[0].Total != 3 || !seen[0].Scored || seen[0].Score != 1 {
		t.Fatalf("first report = %+v", seen[0])
	}
	if seen[1].Done != 2 || seen[1].Score != 0.5 {
		t.Fatalf("second report = %+v", seen[1])
	}
	if seen[2].Done != 3 || seen[2].Scored || seen[2].Score != 0 {
		t.Fatalf("advance report = %+v", seen[2])
	}
	if seen[2].Elapsed < seen[0].Elapsed {
		t.Fatalf("elapsed regressed: %v then %v", seen[0].Elapsed, seen[2].Elapsed)
	}
}

func TestProgressTrackerIsInertWithoutASink(t *testing.T) {
	if tracker := trackProgress(t.Context(), "store/mmlu", 1); tracker != nil {
		t.Fatal("unbound context produced a tracker")
	}
	var tracker *progressTracker
	tracker.observe(1)
	tracker.advance()
	if WithProgress(t.Context(), nil).Value(progressKey{}) != nil {
		t.Fatal("nil sink was bound")
	}
}
