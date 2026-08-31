package loop

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func breakerObservations(value float64, count int) []LiveObservation {
	observations := make([]LiveObservation, count)
	for index := range observations {
		observations[index] = LiveObservation{Value: value, Applicable: true}
	}
	return observations
}

// TestCandidateCircuitBreakerTransitions pins the breaker contract: a
// judged window needs enough total observations, skipped and inapplicable
// work never enters the ratios, a regression breach requires every
// comparable observation across the boundary, comparable starvation while
// work keeps arriving is a coverage breach, a single breached window can
// only advise, escalation climbs the recorded confirmation ladder one
// step per consecutive breach to operator-stop, a clear window resets the
// count, and the published transition cites its window derivation.
func TestCandidateCircuitBreakerTransitions(t *testing.T) {
	ctx := t.Context()
	historyEvidence := testutil.ArtifactID(t, artifact.KindEvidence, "breaker-history")
	window, err := evaluation.DeriveLiveSafetyWindow(
		"exact-match", runrecord.DirectionMaximize,
		[]float64{0.52, 0.58, 0.55, 0.50, 0.60, 0.54, 0.53, 0.57, 0.51, 0.56},
		historyEvidence,
	)
	if err != nil {
		t.Fatal(err)
	}
	windowEvidence := testutil.ArtifactID(t, artifact.KindEvidence, "breaker-window-derivation")
	fill := int(window.RequiredObservations)

	clear, err := JudgeCircuitBreaker(window, windowEvidence, breakerObservations(0.55, fill), 2)
	if err != nil || clear.Level != BreakerClear || clear.ConsecutiveBreaches != 0 ||
		clear.Comparable != uint64(fill) || clear.Breached != 0 {
		t.Fatalf("healthy window = (%+v, %v)", clear, err)
	}

	regressed := breakerObservations(0.40, fill)
	ladder := []string{BreakerAdvisory, BreakerFinding, BreakerQuarantine, BreakerOperatorStop, BreakerOperatorStop}
	prior := uint64(0)
	for step, level := range ladder {
		transition, err := JudgeCircuitBreaker(window, windowEvidence, regressed, prior)
		if err != nil || transition.Level != level || transition.Cause != BreakerCauseRegression ||
			transition.ConsecutiveBreaches != prior+1 {
			t.Fatalf("escalation step %d = (%+v, %v), want level %s", step, transition, err, level)
		}
		prior = transition.ConsecutiveBreaches
	}

	noisy := append(breakerObservations(0.40, fill-1), LiveObservation{Value: 0.55, Applicable: true})
	if transition, err := JudgeCircuitBreaker(window, windowEvidence, noisy, 3); err != nil ||
		transition.Level != BreakerClear || transition.ConsecutiveBreaches != 0 {
		t.Fatalf("unsustained window escalated: (%+v, %v)", transition, err)
	}

	excludedInside := append(breakerObservations(0.40, fill),
		LiveObservation{Value: 0.55}, LiveObservation{Value: 0.55}, LiveObservation{Value: 0.55})
	shielded, err := JudgeCircuitBreaker(window, windowEvidence, excludedInside, 0)
	if err != nil || shielded.Level != BreakerAdvisory || shielded.Cause != BreakerCauseRegression ||
		shielded.Excluded != 3 || shielded.Comparable != uint64(fill) {
		t.Fatalf("excluded work entered the ratio: (%+v, %v)", shielded, err)
	}

	starved := append(breakerObservations(0.55, fill-1),
		LiveObservation{Value: 0.55}, LiveObservation{Value: 0.55})
	for index := range starved {
		starved[index].Applicable = index < fill-1
	}
	coverage, err := JudgeCircuitBreaker(window, windowEvidence, starved, 0)
	if err != nil || coverage.Level != BreakerAdvisory || coverage.Cause != BreakerCauseCoverage {
		t.Fatalf("coverage starvation cleared: (%+v, %v)", coverage, err)
	}

	if _, err := JudgeCircuitBreaker(
		window, windowEvidence, breakerObservations(0.55, fill-1), 0,
	); err == nil || !strings.Contains(err.Error(), "cannot fill the derived comparison window") {
		t.Fatalf("underfilled window judged: %v", err)
	}
	if _, err := JudgeCircuitBreaker(
		evaluation.LiveSafetyWindow{}, windowEvidence, breakerObservations(0.55, fill), 0,
	); err == nil || !strings.Contains(err.Error(), "derived safety window") {
		t.Fatalf("underived window judged: %v", err)
	}
	if _, err := JudgeCircuitBreaker(
		window, artifact.ID{}, breakerObservations(0.55, fill), 0,
	); err == nil || !strings.Contains(err.Error(), "published window derivation") {
		t.Fatalf("unwitnessed judgment ran: %v", err)
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	testutil.PublishArtifact(t, store, windowEvidence)
	quarantine, err := JudgeCircuitBreaker(window, windowEvidence, regressed, 2)
	if err != nil || quarantine.Level != BreakerQuarantine {
		t.Fatalf("quarantine step = (%+v, %v)", quarantine, err)
	}
	published, err := PublishCircuitBreakerTransition(ctx, store, quarantine)
	if err != nil {
		t.Fatal(err)
	}
	content, found, err := artifact.ReadContent(ctx, store, published)
	if err != nil || !found {
		t.Fatalf("transition was not published: (%v, %v)", found, err)
	}
	var recorded CircuitBreakerTransition
	if err := json.Unmarshal(content.Data, &recorded); err != nil {
		t.Fatal(err)
	}
	if recorded != quarantine {
		t.Fatalf("published transition = %+v, want %+v", recorded, quarantine)
	}
	if _, err := PublishCircuitBreakerTransition(ctx, store, CircuitBreakerTransition{}); err == nil {
		t.Fatal("unjudged transition published")
	}
}
