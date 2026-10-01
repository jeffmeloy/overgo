package landing

import (
	"testing"
)

// TestRecoveryDrills replays, as the loop meets them, the refusals that cost
// landings time without saying why. Each drill feeds the refusal texts of a
// sequence of landings through the classification and the checkout's record
// that the loop's land command runs on every refusal, and checks the
// recovery's signal on the last one: its class, whether it names an
// external cause with the probe that re-checks it, and how often its class
// recurred. No Defender, gate or go test runs: the texts are the ones those
// tools print.
func TestRecoveryDrills(t *testing.T) {
	t.Parallel()
	// The build of the gate prints the scanner's refusal as Windows words it.
	quarantinedGate := "go build overgo/cmd/gate: copying C:\\Temp\\go-build\\b001\\exe\\a.out.exe: open C:\\Temp\\overgo-gate-1\\gate.exe: " + quarantined.Error()
	for _, drill := range []struct {
		name string
		// refused are the landings refused in order, as row and refusal.
		refused     [][2]string
		class       string
		external    bool
		recurrences int
	}{
		{
			// Defender quarantined the freshly built gate, then did so again
			// for a later row: the loop records an external prerequisite, not
			// a code failure, and the second refusal shows it recurred.
			name:    "the scanner quarantines the gate a landing built, twice",
			refused: [][2]string{{"merge-0123/do", quarantinedGate}, {"recovery-failures-seen/do", quarantinedGate}},
			class:   quarantineClass, external: true, recurrences: 1,
		},
		{
			// A refused landing leaves its dirty tree; landing the row again
			// to the same stage refusal is a recurrence of that row's class,
			// and another row's test refusal is not counted against it.
			name: "a row's test stage refuses it again",
			refused: [][2]string{
				{"row/do", "gate: land: test: failed tests/packages: overgo/cmd/train: TestSeq2SeqRoutePublishesCheckpoint"},
				{"other/do", "gate: land: test: failed tests/packages: overgo/internal/plan: TestAdvance"},
				{"row/do", "gate: land: test: failed tests/packages: overgo/cmd/train: TestSeq2SeqRoutePublishesCheckpoint"},
			},
			class: "test", recurrences: 1,
		},
		{
			name:    "a lane refuses a landing",
			refused: [][2]string{{"row/do", "gate: land: lanes: lane webui: go run ./cmd/webui-lane: exit 1: TestWebUIBrowserSession failed"}},
			class:   "lanes",
		},
		{
			// A refusal no stage made is the gate's own: here the tree moved
			// while the checks ran.
			name:    "the tree changes during the landing",
			refused: [][2]string{{"row/do", "gate: land: the tree changed during the landing: a.go; land again once it is settled"}},
			class:   gateClass,
		},
	} {
		root := t.TempDir()
		var last Refusal
		for _, refused := range drill.refused {
			var err error
			if last, err = RecordRefusal(root, Classify(refused[0], refused[1])); err != nil {
				t.Fatalf("%s: %v", drill.name, err)
			}
		}
		if last.Class != drill.class || (last.External != "") != drill.external || (last.Probe != "") != drill.external || last.Recurrences != drill.recurrences {
			t.Fatalf("%s: recorded %+v, want class %q, external %t, %d recurrences", drill.name, last, drill.class, drill.external, drill.recurrences)
		}
	}
}
