package executor

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's sampler chain for the round's sampled windows (kernels/cuda/strata_sampler.cuh, ported from Strata at
// 296b59d2, MIT: src/kernels/cuda/sampler.cu): sample_tokens' sampled path, the split top_k, its host side translated.

// The split path's geometry (sampler.cu): a block's 4,096 logits on four warps, the widest list 64, at most 64 blocks a
// row, and the scratch sized for 16 rows at least.
const (
	strataSamplerBlockSpan  = 4096
	strataSamplerWarps      = 4
	strataSamplerWarp       = 32
	strataSamplerMaxList    = 64
	strataSamplerMaxBlocks  = 64
	strataSamplerScratchRow = 16
	strataSamplerEntryBytes = 8 // an int2
)

// StrataSampling is sampler.hpp's SamplerParams: the chain penalties, top_k, top_p, min_p, temperature, then one
// Philox draw keyed by Seed and the row's position.
type StrataSampling struct {
	TopK                                       int32
	TopP, MinP, Temperature                    float32
	MinKeep, PenaltyLastN                      int32
	PenaltyRepeat, PenaltyFreq, PenaltyPresent float32
	Seed                                       uint64
}

// strataSamplerParams is SamplerParams in its C layout (80 bytes), passed by value.
type strataSamplerParams struct {
	TopK                                       int32
	TopP, MinP, Temperature                    float32
	MinKeep, PenaltyLastN                      int32
	PenaltyRepeat, PenaltyFreq, PenaltyPresent float32
	_                                          [strataCPadWord]byte
	Seed, Counter                              uint64
	Greedy                                     uint8
	_                                          [strataCPadByte]byte
	LogitBias                                  driver.DevicePtr
	Gumbel                                     uint8
	_                                          [strataCPadByte]byte
}

// strataSampledK is sampled_k: top_k 1..64 as given, 0 and wider keep 64, never more than the vocabulary.
func strataSampledK(topK, vocab int32) int32 {
	k := int32(strataSamplerMaxList)
	if topK > 0 && topK < strataSamplerMaxList {
		k = topK
	}
	return min(k, vocab)
}

// strataSamplerScratchBytes is the split path's block lists for rows of vocab logits.
func strataSamplerScratchBytes(rows, vocab int32) uint64 {
	blocks := (vocab + strataSamplerBlockSpan - 1) / strataSamplerBlockSpan
	return uint64(max(rows, strataSamplerScratchRow)) * uint64(blocks) * strataSamplerMaxList * strataSamplerEntryBytes
}

// launchStrataSampleTokens is sample_tokens' sampled path: rows rows of vocab logits, each drawn with Philox(seed,
// counter + row), into out; history (rows of historyLen, penalties) may be 0.
func launchStrataSampleTokens(state *device.State, functions functionSet, logits driver.DevicePtr, rows, vocab int32,
	history driver.DevicePtr, historyLen int32, sampling StrataSampling, counter uint64, out, scratch driver.DevicePtr) error {
	if rows <= 0 || vocab <= 0 {
		return nil
	}
	if sampling.PenaltyLastN > 0 && (history == 0 || historyLen <= 0) {
		return errors.New("sample_tokens: penalty_last_n needs a history")
	}
	blocks := (vocab + strataSamplerBlockSpan - 1) / strataSamplerBlockSpan
	if blocks > strataSamplerMaxBlocks || scratch == 0 {
		return errors.New("sample_tokens: the split path holds at most 64 blocks of logits and needs its scratch")
	}
	params := strataSamplerParams{
		TopK: sampling.TopK, TopP: sampling.TopP, MinP: sampling.MinP, Temperature: sampling.Temperature,
		MinKeep: sampling.MinKeep, PenaltyLastN: sampling.PenaltyLastN, PenaltyRepeat: sampling.PenaltyRepeat,
		PenaltyFreq: sampling.PenaltyFreq, PenaltyPresent: sampling.PenaltyPresent, Seed: sampling.Seed, Counter: counter,
	}
	k := strataSampledK(sampling.TopK, vocab)
	if err := launchGridABI(state, functions[kernelStrataSamplerSplitPart], driver.Dim3{X: uint32(blocks), Y: uint32(rows), Z: 1},
		driver.Dim3{X: strataSamplerWarps * strataSamplerWarp, Y: 1, Z: 1},
		&logits, &vocab, &history, &historyLen, &params, &k, &blocks, &scratch); err != nil {
		return err
	}
	return launchGridABI(state, functions[kernelStrataSamplerSplitMerge], driver.Dim3{X: uint32(rows), Y: 1, Z: 1},
		driver.Dim3{X: strataSamplerWarp, Y: 1, Z: 1}, &scratch, &blocks, &vocab, &params, &k, &out)
}
