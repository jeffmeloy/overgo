package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/landing"
	"overgo/internal/recipe"
)

// TestOnlyTheCellsThatReadTimesTakeTheQuietMachine holds which cells a cycle runs with no
// landing beside them: the serving and capability measurements and the proofs that train,
// whose first output, throughput and load a build or a test beside them slows, and not the
// benchmarks, the guards or the lanes, whose results a busy machine does not change, so a
// landing that reaches no device lane is not delayed by a quality cell.
func TestOnlyTheCellsThatReadTimesTakeTheQuietMachine(t *testing.T) {
	t.Parallel()
	for validation, timed := range map[string]bool{
		servingValidation: true, trainingProof: true, projectionProof: true,
		"image-proof": true, "video-proof": true, "speech-proof": true, "vqa-proof": true, "specialized-proof": true,
		guardValidation: false, "mmlu-pro": false, "bbh": false, "ifeval": false, "mmlu": false,
	} {
		if got := timedValidation(validation); got != timed {
			t.Errorf("%s: timed = %t, want %t", validation, got, timed)
		}
	}
}

// TestAContendedCellIsMeasuredOnceMoreAfterTheOthers holds what a run does with a timed cell whose
// measurement saw other work on the machine. The measurement is given, and starts no command: it
// answers each call in turn with what the machine showed and how the cell ended.
//
// Four cells run in order. The first is measured beside other work, the second on a quiet machine,
// the third beside other work and fails, the fourth beside other work. The first and the fourth
// are measured once more, after every other cell and in their order. The second time the first is
// quiet, and the fourth is beside other work again, which the run says. A cell that failed is not
// measured again. The run ends with three cells acquired and the one failure named.
func TestAContendedCellIsMeasuredOnceMoreAfterTheOthers(t *testing.T) {
	// Serial: it clears the name of the cycle's credit file in the process environment, so no cell of
	// this fixture is credited to a measurement run that started the test.
	t.Setenv(clioptions.CycleCreditEnvironment, "")
	cell := func(name string) Selected {
		return Selected{ModelValidation: ModelValidation{ModelName: name, Kind: recipe.TaskInference, Validation: servingValidation, Location: "C:/models/" + name + ".gguf"}}
	}
	plan := Plan{Run: []Selected{cell("busy-once"), cell("quiet"), cell("busy-and-failed"), cell("busy-twice")}}
	failure, busy := errors.New("the cell's own failure"), []string{"compile.exe"}
	answers := []struct {
		competing []string
		err       error
	}{{busy, nil}, {nil, nil}, {busy, failure}, {busy, nil}, {nil, nil}, {busy, nil}}
	calls := 0
	measure := func(context.Context, io.Writer, func() error) (landing.Reading, error) {
		if calls >= len(answers) {
			t.Fatalf("the run measured a cell a %dth time, want %d measurements", calls+1, len(answers))
		}
		answer := answers[calls]
		calls++
		return landing.Reading{Competing: answer.competing}, answer.err
	}
	var out bytes.Buffer
	err := executeMeasured(t.Context(), &out, ".", "C:/stores/selected", nil, plan, scopeCells, measure)
	if err == nil || !strings.Contains(err.Error(), "re-acquired 3 cell(s); 1 failed") || !strings.Contains(err.Error(), "busy-and-failed") || !strings.Contains(err.Error(), failure.Error()) {
		t.Fatalf("the run ended %v, want three cells acquired and the failed one named with its failure\n%s", err, out.String())
	}
	if calls != len(answers) {
		t.Fatalf("the run measured %d times, want %d: each cell once and the two contended successes once more\n%s", calls, len(answers), out.String())
	}
	said := out.String()
	// The lines the run prints, in the order it must print them.
	at := 0
	for _, line := range []string{
		"validate -run 1/4: " + servingValidation + " busy-once", "validate -run 2/4: " + servingValidation + " quiet",
		"validate -run 3/4: " + servingValidation + " busy-and-failed", "validate -run 4/4: " + servingValidation + " busy-twice",
		"validate -run again: busy-once " + servingValidation, "validate -run again: busy-twice " + servingValidation,
		"busy-twice " + servingValidation + " was measured beside other work both times",
	} {
		found := strings.Index(said[at:], line)
		if found < 0 {
			t.Fatalf("the run's output lacks %q after its first %d bytes:\n%s", line, at, said)
		}
		at += found + len(line)
	}
	for _, absent := range []string{"again: quiet", "again: busy-and-failed", "busy-once " + servingValidation + " was measured beside other work both times", "set aside"} {
		if strings.Contains(said, absent) {
			t.Errorf("the run's output holds %q, which it must not: only a contended success is measured again, and no store is selected here\n%s", absent, said)
		}
	}
}
