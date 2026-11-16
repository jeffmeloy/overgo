package evaluation

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
)

// TestTheSemanticScreenAndStillnessAreDiagnostics holds the screen and stillness to what their evidence supports
// (preset-evidence/diagnostics, SQA-342): how far a candidate's embedding lies from the reference's beside another
// seed's establishes no loss of what a prompt asked, so a screen whose every case lies farther refuses no candidate,
// holds no verdict and says it is a diagnostic beside its counts. Stillness is read against each clip's own source, as
// a diagnostic that bounds nothing: of two clips, one as still as its stationary source and one frozen where its source
// moves, the frozen clip alone reads as stiller, where the stillest source of the suite let the freeze pass.
func TestTheSemanticScreenAndStillnessAreDiagnostics(t *testing.T) {
	t.Parallel()
	const screened = 20
	cases := make([]ScreenCase, screened)
	for index := range cases {
		cases[index] = ScreenCase{Name: fmt.Sprintf("case-%02d", index), Candidate: 0.4, Reseeded: 0.1}
	}
	screen := testcheck.Must(ScreenCandidate(cases))(t)
	held := map[string]any{}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(json.Marshal(screen))(t), &held))
	for _, verdict := range []string{"passed", "level", "tail"} {
		if _, kept := held[verdict]; kept {
			t.Errorf("a screen records %q, a verdict its evidence does not support: %v", verdict, held)
		}
	}
	if says := screen.Says(); screen.Farther != screened || !strings.Contains(says, "refusing no candidate") ||
		!strings.Contains(says, fmt.Sprintf("farther from the reference than another seed in %d of %d cases", screened, screened)) {
		t.Errorf("a screen whose every case lies farther counts %d and says %q; want %d, said as a diagnostic", screen.Farther, says, screened)
	}

	stationary := FrameSimilarity{Neighbours: []float64{0.999, 0.999}, Mean: 0.999, Lowest: 0.999}
	moving := FrameSimilarity{Neighbours: []float64{0.8, 0.8}, Mean: 0.8, Lowest: 0.8}
	frozen := FrameSimilarity{Neighbours: []float64{0.99, 0.99}, Mean: 0.99, Lowest: 0.99}
	motion := ClipMotion{
		Encoder: "encoder",
		Clips:   map[string]FrameSimilarity{"harbour": stationary, "street": frozen},
		Sources: map[string]FrameSimilarity{"harbour": stationary, "street": moving},
	}
	var read *runrecord.Metric
	for _, metric := range motion.Metrics() {
		if metric.Name == StillerThanSourceMetric {
			read = &metric
		}
	}
	if read == nil || read.Value != 0.5 || read.Direction != runrecord.DirectionNeutral {
		t.Errorf("a stationary clip beside its stationary source and a frozen one beside its moving source read %+v; want half the clips stiller than their source, neutral", read)
	}
}
