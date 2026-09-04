package longform

import (
	"math"
	"slices"
	"testing"
)

// The degeneration measures separate a continuation that keeps moving
// from one that has locked into a loop: a period-4 loop over 40 tokens
// holds four distinct 4-grams out of 37 and repeats a 36-token span,
// while a sequence without a repeated token keeps every 4-gram distinct
// and repeats nothing.
func TestDegenerationMeasuresSeparateLoopsFromFreshText(t *testing.T) {
	fresh := make([]int32, 40)
	for index := range fresh {
		fresh[index] = int32(index)
	}
	if ratio := DistinctNGramRatio(fresh, 4); ratio != 1 {
		t.Fatalf("fresh distinct 4-gram ratio = %v, want 1", ratio)
	}
	if span := LongestRepeatedSpan(fresh); span != 0 {
		t.Fatalf("fresh longest repeated span = %d, want 0", span)
	}
	loop := make([]int32, 40)
	for index := range loop {
		loop[index] = int32(index % 4)
	}
	if ratio := DistinctNGramRatio(loop, 4); math.Abs(ratio-4.0/37.0) > 1e-12 {
		t.Fatalf("loop distinct 4-gram ratio = %v, want %v", ratio, 4.0/37.0)
	}
	if span := LongestRepeatedSpan(loop); span != 36 {
		t.Fatalf("loop longest repeated span = %d, want 36", span)
	}
	// A repeated phrase inside otherwise fresh text is found at its
	// length, wherever the two occurrences sit.
	phrase := slices.Clone(fresh[:10])
	phrase = append(phrase, 100, 101, 102, 103, 104)
	phrase = append(phrase, fresh[20:30]...)
	phrase = append(phrase, 100, 101, 102, 103, 104, 200)
	if span := LongestRepeatedSpan(phrase); span != 5 {
		t.Fatalf("phrase longest repeated span = %d, want 5", span)
	}
	// Too few tokens for one n-gram: nothing repeats.
	if ratio := DistinctNGramRatio([]int32{1, 2, 3}, 4); ratio != 1 {
		t.Fatalf("short distinct ratio = %v, want 1", ratio)
	}
	if ratio := DistinctNGramRatio(nil, 0); ratio != 1 {
		t.Fatalf("empty distinct ratio = %v, want 1", ratio)
	}
}

// The verdict reads every floor: rates against the model's own
// short-prompt record, and the output measures only once the output is
// long enough to carry them.
func TestJudgeAppliesEveryDeclaredFloor(t *testing.T) {
	floors := DeclaredFloors()
	short := ShortRates{PromptTokensPerSecond: 2000, DecodeTokensPerSecond: 40}
	healthy := Measure{
		PromptTokens: floors.PromptTokens, OutputTokens: floors.OutputTokens,
		PromptTokensPerSecond: 2400, DecodeTokensPerSecond: 36,
		DistinctFourGramRatio: 0.94, LongestRepeatedSpan: 6,
		Score: ContextScore{ScoreTokens: floors.ScoreTokens, LongContextNLL: 1.4, ShortContextNLL: 1.9, ContextGain: 0.5},
	}
	if verdict := Judge(healthy, short, floors); !verdict.Passed || len(verdict.Reasons) != 0 {
		t.Fatalf("healthy verdict = %+v", verdict)
	}
	cases := []struct {
		name    string
		measure Measure
		short   ShortRates
		reasons int
	}{
		{name: "no benchmark", measure: healthy, short: ShortRates{}, reasons: 1},
		{name: "slow prompt", measure: func() Measure { m := healthy; m.PromptTokensPerSecond = 900; return m }(), short: short, reasons: 1},
		{name: "slow decode", measure: func() Measure { m := healthy; m.DecodeTokensPerSecond = 19; return m }(), short: short, reasons: 1},
		{name: "stopped early", measure: func() Measure {
			m := healthy
			m.OutputTokens, m.StoppedEarly, m.DecodeTokensPerSecond = 5, true, 0
			return m
		}(), short: short, reasons: 1},
		{name: "unscored", measure: func() Measure { m := healthy; m.Score = ContextScore{}; return m }(), short: short, reasons: 1},
		{name: "long context hurts", measure: func() Measure {
			m := healthy
			m.Score = ContextScore{ScoreTokens: 128, LongContextNLL: 6.2, ShortContextNLL: 1.9, ContextGain: -4.3}
			return m
		}(), short: short, reasons: 1},
	}
	for _, tc := range cases {
		verdict := Judge(tc.measure, tc.short, floors)
		if verdict.Passed || len(verdict.Reasons) != tc.reasons {
			t.Errorf("%s: verdict = %+v, want %d reason(s)", tc.name, verdict, tc.reasons)
		}
	}
	// A greedy loop is recorded, not bounded: the context gain is the
	// output floor, and a looping output with a positive gain passes.
	looping := healthy
	looping.DistinctFourGramRatio, looping.LongestRepeatedSpan = 0.1, 231
	if verdict := Judge(looping, short, floors); !verdict.Passed {
		t.Fatalf("looping verdict = %+v, want pass", verdict)
	}
	if got := (Verdict{Passed: true}).String(); got != "PASS" {
		t.Fatalf("pass string = %q", got)
	}
}
