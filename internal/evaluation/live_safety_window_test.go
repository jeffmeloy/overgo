package evaluation

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestLiveSafetyWindowDerivation pins the distribution-free window
// contract: boundaries derive from each metric's own history by order
// statistics with no Gaussian, stationary, or IID assumption; small
// histories keep the exact envelope and only complete separation can
// decide, histories large enough to trim a tail use the symmetric
// quantile band strictly inside the extremes, the required observation
// count derives from the history size alone and shrinks as evidence
// grows, and the published derivation cites its history evidence.
func TestLiveSafetyWindowDerivation(t *testing.T) {
	ctx := context.Background()
	historyEvidence := planID(t, artifact.KindEvidence, "safety-window-history")

	small := []float64{0.52, 0.58, 0.55, 0.50, 0.60, 0.54, 0.53, 0.57, 0.51, 0.56}
	envelope, err := DeriveLiveSafetyWindow("exact-match", runrecord.DirectionMaximize, small, historyEvidence)
	if err != nil {
		t.Fatal(err)
	}
	if envelope.Mode != LiveSafetyModeEnvelope || envelope.Lower != 0.50 || envelope.Upper != 0.60 ||
		envelope.Samples != 10 || envelope.RequiredObservations < 1 {
		t.Fatalf("small-history window = %+v", envelope)
	}

	large := make([]float64, 0, 99)
	for index := 0; index < 97; index++ {
		large = append(large, 0.5+float64(index%10)/100)
	}
	large = append(large, 0.05, 0.95)
	band, err := DeriveLiveSafetyWindow("exact-match", runrecord.DirectionMaximize, large, historyEvidence)
	if err != nil {
		t.Fatal(err)
	}
	if band.Mode != LiveSafetyModeQuantile || band.Samples != 99 {
		t.Fatalf("large-history window = %+v", band)
	}
	if band.Lower <= 0.05 || band.Upper >= 0.95 || band.Lower >= band.Upper {
		t.Fatalf("quantile band [%v, %v] is not strictly inside the extremes", band.Lower, band.Upper)
	}
	if band.RequiredObservations >= envelope.RequiredObservations || band.RequiredObservations < 1 {
		t.Fatalf(
			"required observations must shrink as history grows: %d (n=99) vs %d (n=10)",
			band.RequiredObservations, envelope.RequiredObservations,
		)
	}

	refusals := []struct {
		name      string
		metric    string
		direction runrecord.Direction
		history   []float64
		evidence  artifact.ID
		want      string
	}{
		{"unnamed metric", "", runrecord.DirectionMaximize, small, historyEvidence, "metric name"},
		{"neutral direction", "exact-match", runrecord.DirectionNeutral, small, historyEvidence, "comparison direction"},
		{"unwitnessed history", "exact-match", runrecord.DirectionMaximize, small, artifact.ID{}, "history evidence"},
		{"single observation", "exact-match", runrecord.DirectionMaximize, []float64{0.5}, historyEvidence, "cannot bound"},
		{"non-finite observation", "exact-match", runrecord.DirectionMaximize,
			[]float64{0.5, math.NaN(), 0.6}, historyEvidence, "non-finite"},
	}
	for _, refusal := range refusals {
		if _, err := DeriveLiveSafetyWindow(
			refusal.metric, refusal.direction, refusal.history, refusal.evidence,
		); err == nil || !strings.Contains(err.Error(), refusal.want) {
			t.Fatalf("%s derived: %v", refusal.name, err)
		}
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "live-safety/window-fixture/history",
		Artifacts: []artifact.Descriptor{{ID: historyEvidence}},
	}); err != nil {
		t.Fatal(err)
	}
	published, err := PublishLiveSafetyWindow(ctx, store, band)
	if err != nil {
		t.Fatal(err)
	}
	content, found, err := artifact.ReadContent(ctx, store, published)
	if err != nil || !found {
		t.Fatalf("derivation was not published: (%v, %v)", found, err)
	}
	var recorded LiveSafetyWindow
	if err := json.Unmarshal(content.Data, &recorded); err != nil {
		t.Fatal(err)
	}
	if recorded != band {
		t.Fatalf("published derivation = %+v, want %+v", recorded, band)
	}
	if _, err := PublishLiveSafetyWindow(ctx, store, LiveSafetyWindow{}); err == nil {
		t.Fatal("underived window published")
	}
}
