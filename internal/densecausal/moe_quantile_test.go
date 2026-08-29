package densecausal

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestQuantileEstimatorReportsRankErrorSmallSampleAndStratumFailures(t *testing.T) {
	id := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindDatasetShard, name) }
	complete := moeQuantileStratum{
		Stratum: id("complete"),
		ExpertSamples: [][]float64{
			{0, 1, 2, 3, 4, 5, 6, 7},
			{0, 0, 0, 0, 1, 1, 1, 1},
		},
	}
	exact, err := estimateMoEQuantiles(moeQuantileSpec{
		Method: moeQuantileExact, Numerator: 3, Denominator: 4,
	}, []moeQuantileStratum{complete})
	if err != nil || !exact.Complete || exact.Strata[0].Experts[0].ValueLower != 5 ||
		exact.Strata[0].Experts[0].MaximumRankError != 0 || exact.Strata[0].Experts[0].Bins != 0 {
		t.Fatalf("exact quantile evidence = (%+v, %v)", exact, err)
	}

	histogram, err := estimateMoEQuantiles(moeQuantileSpec{
		Method: moeQuantileHistogram, Numerator: 3, Denominator: 4, Bins: 2,
	}, []moeQuantileStratum{complete})
	if err != nil || !histogram.Complete {
		t.Fatalf("histogram quantile evidence = (%+v, %v)", histogram, err)
	}
	approximation := histogram.Strata[0].Experts[0]
	if approximation.RankLower != 4 || approximation.RankUpper != 7 || approximation.MaximumRankError != 2 ||
		approximation.ValueLower != 3.5 || approximation.ValueUpper != 7 {
		t.Fatalf("histogram rank envelope = %+v", approximation)
	}

	partial, err := estimateMoEQuantiles(moeQuantileSpec{
		Method: moeQuantileExact, Numerator: 3, Denominator: 4,
	}, []moeQuantileStratum{
		complete,
		{Stratum: id("small"), ExpertSamples: [][]float64{{1, 2}}},
		{Stratum: id("missing"), ExpertSamples: nil},
		{Stratum: id("nonfinite"), ExpertSamples: [][]float64{{0, 1, 2, math.NaN()}}},
	})
	if err != nil || partial.Complete {
		t.Fatalf("partial evidence = (%+v, %v)", partial, err)
	}
	failures := make(map[artifact.ID]moeQuantileFailure)
	for _, stratum := range partial.Strata {
		if stratum.Failure != "" {
			failures[stratum.Stratum] = stratum.Failure
			continue
		}
		for _, expert := range stratum.Experts {
			if expert.Failure != "" {
				failures[stratum.Stratum] = expert.Failure
			}
		}
	}
	if failures[id("small")] != moeQuantileSmallSample || failures[id("missing")] != moeQuantileInvalidStratum ||
		failures[id("nonfinite")] != moeQuantileNonFinite || failures[id("complete")] != "" {
		t.Fatalf("stratum failures = %+v", failures)
	}

	if _, err := estimateMoEQuantiles(moeQuantileSpec{
		Method: moeQuantileHistogram, Numerator: 1, Denominator: 2,
	}, []moeQuantileStratum{complete}); err == nil {
		t.Fatal("histogram estimator admitted without an explicit bin budget")
	}
	if _, err := estimateMoEQuantiles(moeQuantileSpec{
		Method: moeQuantileExact, Numerator: 1, Denominator: 2,
	}, []moeQuantileStratum{complete, complete}); err == nil {
		t.Fatal("duplicate strata admitted")
	}
}
