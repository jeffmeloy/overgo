package densecausal

import (
	"cmp"
	"errors"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

type moeQuantileMethod string

const (
	moeQuantileExact     moeQuantileMethod = "exact"
	moeQuantileHistogram moeQuantileMethod = "histogram"
)

type moeQuantileSpec struct {
	Method      moeQuantileMethod
	Numerator   uint64
	Denominator uint64
	Bins        uint64
}

type moeQuantileStratum struct {
	Stratum       artifact.ID
	ExpertSamples [][]float64
}

type moeQuantileFailure string

const (
	moeQuantileInvalidStratum moeQuantileFailure = "invalid-stratum"
	moeQuantileNoSamples      moeQuantileFailure = "no-samples"
	moeQuantileNonFinite      moeQuantileFailure = "non-finite-sample"
	moeQuantileRangeOverflow  moeQuantileFailure = "sample-range-overflow"
	moeQuantileSmallSample    moeQuantileFailure = "small-sample"
)

type moeExpertQuantileEvidence struct {
	Samples          uint64
	TargetRank       uint64
	RankLower        uint64
	RankUpper        uint64
	MaximumRankError uint64
	ValueLower       float64
	ValueUpper       float64
	Bins             uint64
	Failure          moeQuantileFailure
}

type moeStratumQuantileEvidence struct {
	Stratum  artifact.ID
	Complete bool
	Experts  []moeExpertQuantileEvidence
	Failure  moeQuantileFailure
}

type moeQuantileEvidence struct {
	Spec     moeQuantileSpec
	Complete bool
	Strata   []moeStratumQuantileEvidence
}

// estimateMoEQuantiles reports empirical estimator evidence. It does not infer
// population confidence, pool strata, or authorize a routing controller.
var estimateMoEQuantiles = func(spec moeQuantileSpec, strata []moeQuantileStratum) (moeQuantileEvidence, error) {
	if (spec.Method != moeQuantileExact && spec.Method != moeQuantileHistogram) || spec.Numerator == 0 ||
		spec.Denominator == 0 || spec.Numerator >= spec.Denominator ||
		spec.Method == moeQuantileExact && spec.Bins != 0 || spec.Method == moeQuantileHistogram && spec.Bins == 0 || len(strata) == 0 {
		return moeQuantileEvidence{}, errors.New("densecausal: invalid empirical quantile specification")
	}
	ordered := slices.Clone(strata)
	slices.SortFunc(ordered, func(left, right moeQuantileStratum) int {
		return cmp.Compare(left.Stratum.String(), right.Stratum.String())
	})
	evidence := moeQuantileEvidence{Spec: spec, Complete: true, Strata: make([]moeStratumQuantileEvidence, len(ordered))}
	for index, stratum := range ordered {
		if index > 0 && ordered[index-1].Stratum == stratum.Stratum {
			return moeQuantileEvidence{}, errors.New("densecausal: duplicate empirical quantile stratum")
		}
		result := moeStratumQuantileEvidence{Stratum: stratum.Stratum, Complete: true}
		if stratum.Stratum.Kind() != artifact.KindDatasetShard || len(stratum.ExpertSamples) == 0 {
			result.Complete, result.Failure, evidence.Complete = false, moeQuantileInvalidStratum, false
			evidence.Strata[index] = result
			continue
		}
		result.Experts = make([]moeExpertQuantileEvidence, len(stratum.ExpertSamples))
		for expert, samples := range stratum.ExpertSamples {
			result.Experts[expert] = empiricalExpertQuantile(spec, samples)
			if result.Experts[expert].Failure != "" {
				result.Complete, evidence.Complete = false, false
			}
		}
		evidence.Strata[index] = result
	}
	return evidence, nil
}

func empiricalExpertQuantile(spec moeQuantileSpec, samples []float64) moeExpertQuantileEvidence {
	result := moeExpertQuantileEvidence{Samples: uint64(len(samples))}
	if len(samples) == 0 {
		result.Failure = moeQuantileNoSamples
		return result
	}
	values := slices.Clone(samples)
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			result.Failure = moeQuantileNonFinite
			return result
		}
	}
	count := uint64(len(values))
	lowerTail, lowerOK := checked.Mul64(count, spec.Numerator)
	upperTail, upperOK := checked.Mul64(count, spec.Denominator-spec.Numerator)
	if !lowerOK || !upperOK {
		result.Failure = moeQuantileSmallSample
		return result
	}
	rank := lowerTail / spec.Denominator
	if lowerTail%spec.Denominator != 0 {
		rank++
	}
	result.TargetRank = rank - 1
	if lowerTail < spec.Denominator || upperTail < spec.Denominator {
		result.Failure = moeQuantileSmallSample
		return result
	}
	if spec.Method == moeQuantileExact {
		slices.Sort(values)
		result.RankLower, result.RankUpper = result.TargetRank, result.TargetRank
		result.ValueLower, result.ValueUpper = values[result.TargetRank], values[result.TargetRank]
		return result
	}
	return histogramExpertQuantile(spec.Bins, result, values)
}

func histogramExpertQuantile(requestedBins uint64, result moeExpertQuantileEvidence, values []float64) moeExpertQuantileEvidence {
	minimum, maximum := values[0], values[0]
	for _, value := range values[1:] {
		minimum, maximum = min(minimum, value), max(maximum, value)
	}
	bins := min(requestedBins, uint64(len(values)))
	result.Bins = bins
	counts := make([]uint64, int(bins))
	span := maximum - minimum
	if math.IsInf(span, 0) {
		result.Failure = moeQuantileRangeOverflow
		return result
	}
	for _, value := range values {
		bin := uint64(0)
		if span != 0 {
			bin = uint64((value - minimum) / span * float64(bins))
			if bin == bins {
				bin--
			}
		}
		counts[bin]++
	}
	var before uint64
	for bin, count := range counts {
		if result.TargetRank >= before+count {
			before += count
			continue
		}
		result.RankLower, result.RankUpper = before, before+count-1
		result.MaximumRankError = max(result.TargetRank-result.RankLower, result.RankUpper-result.TargetRank)
		if span == 0 {
			result.ValueLower, result.ValueUpper = minimum, maximum
			return result
		}
		result.ValueLower = minimum + span*float64(bin)/float64(bins)
		result.ValueUpper = minimum + span*float64(bin+1)/float64(bins)
		return result
	}
	panic("densecausal: empirical histogram rank was not covered")
}
