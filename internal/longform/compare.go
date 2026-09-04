package longform

import (
	"fmt"
	"math"
	"slices"
)

// Compare judges a fresh run against a record of the same model from
// another inference surface: every shape the two share must reproduce
// the record's leading greedy tokens, hold its NLL within the
// tolerance, and keep the declared fraction of its rates, and the
// fresh ladder must climb every rung the record climbed up to the
// ceiling. A difference names the shape, so the report says which
// context length moved.
func Compare(record, fresh Result, floors Floors, ceiling int) Verdict {
	var reasons []string
	reasons = append(reasons, compareTokens("short", record.Shape.OutputIDs, fresh.Shape.OutputIDs, floors)...)
	if delta := math.Abs(fresh.Shape.NLL - record.Shape.NLL); delta > floors.NLLTolerance {
		reasons = append(reasons, fmt.Sprintf("short: NLL moved %.4f nat/token (%.4f to %.4f), past the %.2f tolerance",
			delta, record.Shape.NLL, fresh.Shape.NLL, floors.NLLTolerance))
	}
	reasons = append(reasons, compareRates("short", record.Shape.Measure, fresh.Shape.Measure, floors)...)
	recorded := map[int]Rung{}
	for _, rung := range record.Rungs {
		recorded[rung.Measure.PromptTokens] = rung
	}
	climbed := map[int]bool{}
	for _, rung := range fresh.Rungs {
		climbed[rung.Measure.PromptTokens] = true
		reference, known := recorded[rung.Measure.PromptTokens]
		if !known {
			continue
		}
		name := fmt.Sprintf("rung %d", rung.Measure.PromptTokens)
		reasons = append(reasons, compareTokens(name, reference.OutputIDs, rung.OutputIDs, floors)...)
		if delta := math.Abs(rung.Measure.Score.LongContextNLL - reference.Measure.Score.LongContextNLL); delta > floors.NLLTolerance {
			reasons = append(reasons, fmt.Sprintf("%s: NLL moved %.4f nat/token (%.4f to %.4f), past the %.2f tolerance",
				name, delta, reference.Measure.Score.LongContextNLL, rung.Measure.Score.LongContextNLL, floors.NLLTolerance))
		}
		reasons = append(reasons, compareRates(name, reference.Measure, rung.Measure, floors)...)
	}
	for _, rung := range record.Rungs {
		length := rung.Measure.PromptTokens
		if (ceiling == 0 || length <= ceiling) && !climbed[length] {
			reasons = append(reasons, fmt.Sprintf("rung %d: the record climbed it, the fresh run stopped (%s)", length, fresh.LadderStop))
		}
	}
	return Verdict{Passed: len(reasons) == 0, Reasons: reasons}
}

func compareTokens(name string, record, fresh []int32, floors Floors) []string {
	want := min(floors.IdenticalTokens, len(record))
	if len(fresh) < want {
		return []string{fmt.Sprintf("%s: %d greedy tokens where the record holds %d", name, len(fresh), want)}
	}
	if index := slices.Index(equalPrefixMask(record[:want], fresh[:want]), false); index >= 0 {
		return []string{fmt.Sprintf("%s: greedy token %d diverges from the record (%d to %d)", name, index, record[index], fresh[index])}
	}
	return nil
}

func equalPrefixMask(record, fresh []int32) []bool {
	mask := make([]bool, len(record))
	for index := range record {
		mask[index] = record[index] == fresh[index]
	}
	return mask
}

func compareRates(name string, record, fresh Measure, floors Floors) []string {
	var reasons []string
	if floor := floors.RateRegressionFraction * record.PromptTokensPerSecond; fresh.PromptTokensPerSecond < floor {
		reasons = append(reasons, fmt.Sprintf("%s: prompt %.1f tok/s is below %.1f (%.0f%% of the record's %.1f)",
			name, fresh.PromptTokensPerSecond, floor, 100*floors.RateRegressionFraction, record.PromptTokensPerSecond))
	}
	if floor := floors.RateRegressionFraction * record.DecodeTokensPerSecond; fresh.DecodeTokensPerSecond < floor {
		reasons = append(reasons, fmt.Sprintf("%s: decode %.1f tok/s is below %.1f (%.0f%% of the record's %.1f)",
			name, fresh.DecodeTokensPerSecond, floor, 100*floors.RateRegressionFraction, record.DecodeTokensPerSecond))
	}
	return reasons
}
