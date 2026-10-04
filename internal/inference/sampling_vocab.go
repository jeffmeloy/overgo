package inference

import (
	"cmp"
	"context"
	"errors"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

func (r *Runner) DeviceMemoryStats(ctx context.Context) (driver.MemoryStats, error) {
	if r == nil || r.worker == nil {
		return driver.MemoryStats{}, errRunnerNil
	}
	return r.worker.MemoryStats(ctx)
}

// ExecutionMetrics reports the executor's device-memory attribution: arena,
// cuBLAS and Q8 staging, and buffer-pool residency. Callers bound retained
// device bytes against the model's loaded footprint plus these owners.
func (r *Runner) ExecutionMetrics(ctx context.Context) (executor.ExecutionMetrics, error) {
	if r == nil || r.cuda == nil {
		return executor.ExecutionMetrics{}, errRunnerNil
	}
	return r.cuda.Metrics(ctx)
}

// DeviceMemory is the served device's memory: the driver's total and free
// bytes and the runner's live bytes by owner, largest share first.
type DeviceMemory struct {
	TotalBytes uint64              `json:"total_bytes"`
	FreeBytes  uint64              `json:"free_bytes"`
	Owners     []DeviceMemoryOwner `json:"owners"`
}

// DeviceMemoryOwner is one owner of the runner's live device bytes.
type DeviceMemoryOwner struct {
	Name  string `json:"name"`
	Bytes uint64 `json:"bytes"`
}

// DeviceMemory reports the device's memory with the runner's live bytes
// attributed to their owners; weights are the live bytes no executor owner
// holds.
func (r *Runner) DeviceMemory(ctx context.Context) (DeviceMemory, error) {
	if r == nil || r.worker == nil || r.cuda == nil {
		return DeviceMemory{}, errRunnerNil
	}
	metrics, err := r.cuda.Metrics(ctx)
	if err != nil {
		return DeviceMemory{}, err
	}
	var memory DeviceMemory
	var live uint64
	if err := r.worker.Do(ctx, func(state *device.State) error {
		live = state.Driver.MemoryStats().CurrentBytes
		free, total, infoErr := state.Driver.MemInfo()
		memory.FreeBytes, memory.TotalBytes = free, total
		return infoErr
	}); err != nil {
		return DeviceMemory{}, err
	}
	owners := []DeviceMemoryOwner{
		{"KV cache and buffers", metrics.PoolTotalBytes},
		{"activations", metrics.ArenaCommittedBytes},
		{"matmul staging", metrics.BLASStagingBytes},
		{"attention scores", metrics.BLASScoreBytes},
		{"quantized input staging", metrics.Q8StagingBytes},
	}
	attributed := uint64(0)
	for _, owner := range owners {
		attributed += owner.Bytes
	}
	memory.Owners = append(memory.Owners, DeviceMemoryOwner{"weights", live - min(live, attributed)})
	for _, owner := range owners {
		if owner.Bytes > 0 {
			memory.Owners = append(memory.Owners, owner)
		}
	}
	slices.SortStableFunc(memory.Owners, func(a, b DeviceMemoryOwner) int { return cmp.Compare(b.Bytes, a.Bytes) })
	return memory, nil
}

// ResetDeviceMemoryPeak starts an allocation high-water window at the current
// live bytes. Callers must serialize the window with the measured execution.
func (r *Runner) ResetDeviceMemoryPeak(ctx context.Context) error {
	if r == nil || r.worker == nil {
		return errRunnerNil
	}
	return r.worker.Do(ctx, func(state *device.State) error {
		state.Driver.ResetPeakBytes()
		return nil
	})
}

func (r *Runner) DeviceExecutionStats(ctx context.Context) (driver.ExecutionStats, error) {
	if r == nil || r.worker == nil {
		return driver.ExecutionStats{}, errors.New("inference: CUDA worker is unavailable")
	}
	return r.worker.ExecutionStats(ctx)
}

// TokenizeSamplingText: expands textual sampling parameter without adding
// BOS/EOS tokens
func (r *Runner) TokenizeSamplingText(text string) ([]tokenizer.TokenID, error) {
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	return r.vocab.Encode(text, tokenizer.EncodeOptions{})
}

// SamplingEOGTokens: returns vocabulary's recognized end-of-generation IDs
func (r *Runner) SamplingEOGTokens() []tokenizer.TokenID {
	if r == nil || r.vocab == nil {
		return nil
	}
	return r.vocab.EOGTokens()
}

func (r *Runner) SamplingVocabularySize() int {
	if r == nil || r.vocab == nil {
		return 0
	}
	return len(r.vocab.Tokens)
}

// SamplingInfillVocabulary: returns token pieces and terminal metadata used
// by llama.cpp's infill sampler; Special/control pieces are intentionally
// decoded as empty because llama_sampler_infill requests token pieces with
// render_special disabled
func (r *Runner) SamplingInfillVocabulary() (*sampling.InfillVocabulary, error) {
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	pieces := make([]string, len(r.vocab.Tokens))
	eog := make([]bool, len(r.vocab.Tokens))
	for index := range pieces {
		id := tokenizer.TokenID(index)
		piece, err := r.vocab.DecodePiece(id, false)
		if err != nil {
			return nil, err
		}
		pieces[index] = piece
		eog[index] = r.vocab.IsEOG(id)
	}
	return &sampling.InfillVocabulary{
		Pieces: pieces,
		EOG:    eog,
		EOT:    samplingTokenID(r.vocab.EOT),
		EOS:    samplingTokenID(r.vocab.EOS),
	}, nil
}

func samplingTokenID(id tokenizer.TokenID) int {
	if id == tokenizer.NullToken {
		return -1
	}
	return int(id)
}

func (r *Runner) TokenizeText(
	text string,
	addSpecial, parseSpecial bool,
) ([]tokenizer.TokenID, error) {
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	return r.vocab.Encode(text, tokenizer.EncodeOptions{
		AddSpecial:   addSpecial,
		ParseSpecial: parseSpecial,
	})
}

// TokenizeTextRuns inserts verified media-token runs without re-encoding
// repeated placeholder spellings.
func (r *Runner) TokenizeTextRuns(
	text, placeholder string,
	counts []int,
	addSpecial bool,
) ([]tokenizer.TokenID, []int, error) {
	if r == nil || r.vocab == nil {
		return nil, nil, errRunnerNil
	}
	return tokenizer.EncodeRuns(
		r.vocab.Encode,
		text,
		placeholder,
		counts,
		tokenizer.EncodeOptions{AddSpecial: addSpecial, ParseSpecial: true},
	)
}

// TokenizeTextMarkers expands compact media-token run markers.
func (r *Runner) TokenizeTextMarkers(
	text, placeholder string,
	counts []int,
	addSpecial bool,
) ([]tokenizer.TokenID, []int, error) {
	if r == nil || r.vocab == nil {
		return nil, nil, errRunnerNil
	}
	return tokenizer.EncodeMarkers(
		r.vocab.Encode,
		text,
		placeholder,
		counts,
		tokenizer.EncodeOptions{AddSpecial: addSpecial, ParseSpecial: true},
	)
}

type TokenRenderMode uint8

const (
	RenderText TokenRenderMode = iota
	RenderPrompt
)

func (r *Runner) Detokenize(tokens []tokenizer.TokenID, mode TokenRenderMode) (string, error) {
	if r == nil || r.vocab == nil {
		return "", errRunnerNil
	}
	switch mode {
	case RenderText:
		return r.vocab.Decode(tokens, false)
	case RenderPrompt:
		return r.vocab.Decode(tokens, true)
	default:
		return "", errors.New("inference: invalid token render mode")
	}
}

// TokenPiece: renders one vocabulary token exactly as llama.cpp's tokenizer
// endpoints do, including control-token spellings and raw byte tokens
func (r *Runner) TokenPiece(token tokenizer.TokenID) (string, error) {
	if r == nil || r.vocab == nil {
		return "", errRunnerNil
	}
	return r.vocab.DecodePiece(token, true)
}
