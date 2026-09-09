package longform

import (
	"fmt"
	"math"
	"slices"

	"overgo/internal/cuda/driver"
)

// Compare judges a fresh run against a record of the same model from
// another inference surface: every shape the two share must reproduce
// the record's leading greedy tokens, hold its NLL within the
// tolerance, and keep the declared fraction of its rates, and the
// fresh ladder must climb every rung the record climbed up to the
// ceiling. A difference names the shape, so the report says which
// context length moved.
func Compare(record, fresh Result, floors Floors, ceiling int) Verdict {
	// Legacy records bind the device class, before UUID capture existed.
	// Preserve that comparison scope; a recorded UUID may never disappear.
	device := fresh.Device
	if record.Device.UUID == "" {
		device.UUID = ""
	}
	if !record.Inputs.valid() || !fresh.Inputs.valid() || record.Inputs != fresh.Inputs || record.Floors != fresh.Floors || record.Floors != floors ||
		record.Program != fresh.Program || record.Device != device || record.ContextLength != fresh.ContextLength {
		return Verdict{Reasons: []string{"comparison inputs differ or are unbound: model, corpus, tokens, protocol, floors, recipe, device or context"}}
	}
	var reasons []string
	reasons = append(reasons, compareTokens("short", record.Shape.OutputIDs, fresh.Shape.OutputIDs, floors)...)
	if delta := math.Abs(fresh.Shape.NLL - record.Shape.NLL); math.IsNaN(delta) || math.IsInf(delta, 0) || delta > floors.NLLTolerance {
		reasons = append(reasons, fmt.Sprintf("short: NLL moved %.4f nat/token (%.4f to %.4f), past the %.2f tolerance",
			delta, record.Shape.NLL, fresh.Shape.NLL, floors.NLLTolerance))
	}
	reasons = append(reasons, compareResources("short", record.Shape.Measure, fresh.Shape.Measure, floors)...)
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
		if delta := math.Abs(rung.Measure.Score.LongContextNLL - reference.Measure.Score.LongContextNLL); math.IsNaN(delta) || math.IsInf(delta, 0) || delta > floors.NLLTolerance {
			reasons = append(reasons, fmt.Sprintf("%s: NLL moved %.4f nat/token (%.4f to %.4f), past the %.2f tolerance",
				name, delta, reference.Measure.Score.LongContextNLL, rung.Measure.Score.LongContextNLL, floors.NLLTolerance))
		}
		reasons = append(reasons, compareResources(name, reference.Measure, rung.Measure, floors)...)
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

func compareResources(name string, record, fresh Measure, floors Floors) []string {
	var reasons []string
	for _, rate := range []float64{record.PromptTokensPerSecond, record.DecodeTokensPerSecond, fresh.PromptTokensPerSecond, fresh.DecodeTokensPerSecond} {
		if math.IsNaN(rate) || math.IsInf(rate, 0) || rate <= 0 {
			return []string{fmt.Sprintf("%s: rate is absent or non-finite", name)}
		}
	}
	if floor := floors.RateRegressionFraction * record.PromptTokensPerSecond; fresh.PromptTokensPerSecond < floor {
		reasons = append(reasons, fmt.Sprintf("%s: prompt %.1f tok/s is below %.1f (%.0f%% of the record's %.1f)",
			name, fresh.PromptTokensPerSecond, floor, 100*floors.RateRegressionFraction, record.PromptTokensPerSecond))
	}
	if floor := floors.RateRegressionFraction * record.DecodeTokensPerSecond; fresh.DecodeTokensPerSecond < floor {
		reasons = append(reasons, fmt.Sprintf("%s: decode %.1f tok/s is below %.1f (%.0f%% of the record's %.1f)",
			name, fresh.DecodeTokensPerSecond, floor, 100*floors.RateRegressionFraction, record.DecodeTokensPerSecond))
	}
	return append(reasons, compareMemory(name, record.Memory, fresh.Memory)...)
}

func compareMemory(name string, record, fresh driver.MemoryStats) []string {
	// Legacy comparisons retain their original scope; guard admission requires
	// allocation measurements before a record can establish memory protection.
	if record.PeakBytes == 0 && fresh.PeakBytes == 0 {
		return nil
	}
	for _, memory := range []driver.MemoryStats{record, fresh} {
		if memory.CurrentBytes == 0 || memory.PeakBytes < memory.CurrentBytes {
			return []string{fmt.Sprintf("%s: allocation accounting is absent or inconsistent", name)}
		}
	}
	var reasons []string
	if fresh.CurrentBytes > record.CurrentBytes {
		reasons = append(reasons, fmt.Sprintf("%s: retained device allocations grew from %d to %d bytes", name, record.CurrentBytes, fresh.CurrentBytes))
	}
	if fresh.PeakBytes > record.PeakBytes {
		reasons = append(reasons, fmt.Sprintf("%s: peak device allocations grew from %d to %d bytes", name, record.PeakBytes, fresh.PeakBytes))
	}
	return reasons
}
