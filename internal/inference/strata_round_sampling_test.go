package inference

import (
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/strataspec"
)

// TestStrataSamplingRefusesWhatItCannotHonor: a request the device's sampler would answer with another distribution than it
// declared goes to the host: min_keep beside min_p, and a named stage at 0, which keeps one token there and idles here.
func TestStrataSamplingRefusesWhatItCannotHonor(t *testing.T) {
	t.Parallel()
	topK, topP, typicalP, minP, temperature := sampling.SamplerTopK, sampling.SamplerTopP, sampling.SamplerTypicalP, sampling.SamplerMinP, sampling.SamplerTemperature
	for _, test := range []struct {
		name   string
		config sampling.Config
		round  bool
	}{
		{"min_p alone", sampling.Config{Temperature: 1, TopK: 20, MinP: 0.1}, true},
		{"min_keep of 1 beside min_p", sampling.Config{Temperature: 1, TopK: 20, MinP: 0.1, MinKeep: 1}, true},
		{"min_keep of 5 without min_p", sampling.Config{Temperature: 1, TopK: 20, MinKeep: 5}, true},
		{"min_keep of 5 beside min_p", sampling.Config{Temperature: 1, TopK: 20, MinP: 0.1, MinKeep: 5}, false},
		{"top_p of 0 on an unnamed chain", sampling.Config{Temperature: 1, TopK: 20}, true},
		{"top_p of 0 on a chain that leaves it out", sampling.Config{Temperature: 1, TopK: 20, Samplers: []sampling.SamplerStage{topK, temperature}}, true},
		{"top_p of 0 on a chain that names it", sampling.Config{Temperature: 1, TopK: 20, Samplers: []sampling.SamplerStage{topK, topP, temperature}}, false},
		{"typical_p of 0 on a chain that names it", sampling.Config{Temperature: 1, TopK: 20, TopP: 0.9, Samplers: []sampling.SamplerStage{topK, typicalP, topP, temperature}}, false},
		{"a named chain with every stage in reach", sampling.Config{Temperature: 1, TopK: 20, TopP: 0.9, MinP: 0.1, Samplers: []sampling.SamplerStage{topK, topP, minP, temperature}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			sampler, err := sampling.New(test.config)
			if err != nil {
				t.Fatal(err)
			}
			if _, round := strataSamplingOf(GenerateOptions{Sampler: sampler}); round != test.round {
				t.Fatalf("the round takes this request: %t, want %t", round, test.round)
			}
		})
	}
}

// TestANonFiniteRowEndsTheRound: the guard's flags name the first row holding a NaN or an infinity in any slice, and a row
// of clean flags ends nothing. The device's row of NaN is TestStrataHeadPicksMatchStrata's.
func TestANonFiniteRowEndsTheRound(t *testing.T) {
	t.Parallel()
	const rows = 3
	flags := make([]int32, rows*strataspec.NonfiniteBlocks)
	if _, found := strataspec.NonfiniteRow(flags, rows); found {
		t.Fatal("clean flags ended a round")
	}
	flags[2*strataspec.NonfiniteBlocks-1], flags[2*strataspec.NonfiniteBlocks] = 1, 1
	if row, found := strataspec.NonfiniteRow(flags, rows); !found || row != 1 {
		t.Fatalf("flags in two rows read as row %d, %t", row, found)
	}
	if _, found := strataspec.NonfiniteRow(flags, 1); found {
		t.Fatal("a row past the window's ended the round")
	}
}
