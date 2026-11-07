package scorecard

import (
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
)

// TestPeakMemoryIsReadAtEachPromptLength holds a chat model's peak device memory to each prompt length its first output
// is read at. A verified request that held 1 GiB, a fresh prompt of about 4K tokens that held 2 GiB and one of about 16K
// that held 5 GiB give the row's own peak 1 GiB and a peak at each longer prompt, each beside that prompt's first output
// on its own contract and in the peak memory column alone. A peak at a longer prompt is judged on its own readings: after
// five of 5 GiB, one of 8 GiB is filed as a decline of that cell with 5 GiB as its target, and the history says it moved.
func TestPeakMemoryIsReadAtEachPromptLength(t *testing.T) {
	t.Parallel()
	short, long := evaluation.PromptLengths[0], evaluation.PromptLengths[1]
	const gib = 1 << 30
	// Oldest first: the verified request on a fresh load, then the two longer prompts on the loaded model.
	var attempts []runrecord.ServingObservation
	for _, attempt := range []struct{ tokens, peak uint64 }{{5, gib}, {uint64(short) + 1, 2 * gib}, {uint64(long) + 1, 5 * gib}} {
		attempts = append(attempts, runrecord.ServingObservation{
			SessionReused: attempt.tokens > 5, MeasuredNS: uint64(time.Second), Usage: runrecord.ServingUsage{InputTokens: attempt.tokens, OutputTokens: 1},
			Hardware: []runrecord.ServingHardwareSample{
				{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: uint64(time.Second)},
				{Stage: runrecord.ServingHardwareFinish, ElapsedNS: uint64(time.Second), DevicePeakBytes: attempt.peak},
			},
		})
	}
	model, observed := servedAttempts(t, attempts)
	if cell := peakMemoryCell(evaluation.BenchmarkSummary{}, false, observed); cell.Value != gib {
		t.Fatalf("the row's own peak = %v bytes, want 1 GiB, its verified request's", cell.Value)
	}
	row := measuredFixture(model, map[metric.Name]Cell{})
	row.Lengths = lengthCells(recipe.TaskInference, observed)
	peaks := row.LengthCells(metric.PeakMemory)
	if len(peaks) != 2 || peaks[0].Prompt != short || peaks[0].Value != 2*gib || peaks[1].Prompt != long || peaks[1].Value != 5*gib || peaks[0].Contract == peaks[1].Contract {
		t.Fatalf("peaks at the longer prompts = %+v, want 2 GiB at %d tokens and 5 GiB at %d, on two contracts", peaks, short, long)
	}
	if outputs := row.LengthCells(metric.FirstOutput); len(outputs) != 2 || slices.ContainsFunc(outputs, func(cell Cell) bool { return cell.Unit != "ms" }) {
		t.Fatalf("first outputs at the longer prompts = %+v, want two, in ms", outputs)
	}
	if !slices.ContainsFunc(row.Measured(metric.PeakMemory), func(cell Cell) bool { return cell.Prompt == long }) {
		t.Fatal("the 16K peak is not among the row's peak memory readings, so no baseline holds it")
	}

	record := func(value float64) Record {
		one := measuredFixture(model, map[metric.Name]Cell{})
		// Each record's reading is its own, as a measurement run's is: a source read again is one reading.
		peak := peaks[1]
		peak.Value, peak.Source = value, artifact.ID{}
		one.Lengths = []Cell{peak}
		return Record{Rows: []Row{one}}
	}
	history := slices.Repeat([]Record{record(5 * gib)}, fewestReadings)
	larger := record(8 * gib)
	key := regressionKey(larger.Rows[0], metric.PeakMemory, larger.Rows[0].Lengths[0])
	filed := slices.DeleteFunc(larger.Gaps(Reference(append(slices.Clone(history), larger))), func(gap Gap) bool { return gap.kind != string(regressed) })
	if len(filed) != 1 || filed[0].Key != key || filed[0].Target != 5*gib {
		t.Fatalf("declines of a 16K peak from 5 to 8 GiB = %+v, want the one, with 5 GiB as its target", filed)
	}
	moves := Moves(history[0], larger)
	if len(moves) != 1 || moves[0].Prompt != long || moves[0].Column != metric.PeakMemory || moves[0].Before != 5*gib || moves[0].After != 8*gib {
		t.Fatalf("moves = %+v, want the 16K peak's alone, from 5 to 8 GiB", moves)
	}
}
