package longform

import (
	"slices"
	"strings"
	"testing"
)

// The ladder plan doubles from the start while the rung fits the
// corpus with its scored tail, the declared context, and the ceiling.
func TestLadderRungsFollowContextCorpusAndCeiling(t *testing.T) {
	floors := DeclaredFloors()
	cases := []struct {
		name    string
		context uint32
		corpus  int
		ceiling int
		want    []int
	}{
		{name: "context bound", context: 4096 + 256, corpus: 1 << 20, want: []int{1024, 2048, 4096}},
		{name: "generation must fit the context", context: 4096, corpus: 1 << 20, want: []int{1024, 2048}},
		{name: "corpus bound", context: 1 << 20, corpus: 8192 + 100, want: []int{1024, 2048, 4096}},
		{name: "ceiling bound", context: 1 << 20, corpus: 1 << 20, ceiling: floors.CheckRungCeiling, want: []int{1024, 2048, 4096, 8192}},
		{name: "too short", context: 512, corpus: 1 << 20, want: nil},
	}
	for _, tc := range cases {
		got := LadderRungs(tc.context, tc.corpus, floors, tc.ceiling)
		if len(got) != len(tc.want) {
			t.Errorf("%s: rungs = %v, want %v", tc.name, got, tc.want)
			continue
		}
		for index := range got {
			if got[index] != tc.want[index] {
				t.Errorf("%s: rungs = %v, want %v", tc.name, got, tc.want)
				break
			}
		}
	}
}

// A fresh run that reproduces the record passes; a diverging token, a
// moved NLL, a slower rate, or a shorter ladder each name their shape.
func TestCompareNamesTheShapeThatMoved(t *testing.T) {
	floors := DeclaredFloors()
	ids := make([]int32, 64)
	for index := range ids {
		ids[index] = int32(1000 + index)
	}
	rung := func(length int, prompt, decode float64) Rung {
		return Rung{
			Measure: Measure{PromptTokens: length, OutputTokens: 256, PromptTokensPerSecond: prompt, DecodeTokensPerSecond: decode,
				Score: ContextScore{ScoreTokens: 128, LongContextNLL: 2.5, ShortContextNLL: 2.8, ContextGain: 0.3}},
			OutputIDs: ids,
		}
	}
	record := Result{
		Shape: ShortShape{PromptTokens: 160, OutputIDs: ids, NLL: 3.1, Measure: Measure{PromptTokensPerSecond: 5000, DecodeTokensPerSecond: 200}},
		Rungs: []Rung{rung(1024, 3000, 70), rung(2048, 3100, 68), rung(4096, 3200, 66)},
	}
	if verdict := Compare(record, record, floors, 0); !verdict.Passed {
		t.Fatalf("identical run = %+v", verdict)
	}
	diverged := record
	moved := slices.Clone(ids)
	moved[47] = 7
	diverged.Shape = ShortShape{PromptTokens: 160, OutputIDs: moved, NLL: 3.1, Measure: record.Shape.Measure}
	verdict := Compare(record, diverged, floors, 0)
	if verdict.Passed || len(verdict.Reasons) != 1 || !strings.Contains(verdict.Reasons[0], "short: greedy token 47") {
		t.Fatalf("diverged verdict = %+v", verdict)
	}
	late := record
	late.Shape = ShortShape{PromptTokens: 160, OutputIDs: append(slices.Clone(ids[:48]), 5, 5, 5), NLL: 3.1, Measure: record.Shape.Measure}
	if verdict := Compare(record, late, floors, 0); !verdict.Passed {
		t.Fatalf("a divergence past the identical prefix must pass: %+v", verdict)
	}
	slow := record
	slow.Rungs = []Rung{rung(1024, 3000, 70), rung(2048, 2000, 68), rung(4096, 3200, 66)}
	verdict = Compare(record, slow, floors, 0)
	if verdict.Passed || len(verdict.Reasons) != 1 || !strings.Contains(verdict.Reasons[0], "rung 2048: prompt 2000.0 tok/s") {
		t.Fatalf("slow verdict = %+v", verdict)
	}
	drifted := record
	drifted.Rungs = []Rung{rung(1024, 3000, 70), rung(2048, 3100, 68), rung(4096, 3200, 66)}
	drifted.Rungs[2].Measure.Score.LongContextNLL = 2.6
	verdict = Compare(record, drifted, floors, 0)
	if verdict.Passed || len(verdict.Reasons) != 1 || !strings.Contains(verdict.Reasons[0], "rung 4096: NLL moved 0.1000") {
		t.Fatalf("drifted verdict = %+v", verdict)
	}
	short := record
	short.Rungs = record.Rungs[:2]
	short.LadderStop = "rung 4096: out of device memory"
	verdict = Compare(record, short, floors, 0)
	if verdict.Passed || len(verdict.Reasons) != 1 || !strings.Contains(verdict.Reasons[0], "rung 4096: the record climbed it") {
		t.Fatalf("short ladder verdict = %+v", verdict)
	}
	if verdict := Compare(record, short, floors, 2048); !verdict.Passed {
		t.Fatalf("a rung past the ceiling is not required: %+v", verdict)
	}
}
