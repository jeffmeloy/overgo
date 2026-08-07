package inference

import (
	"context"
	"errors"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

func (r *Runner) DeviceMemoryStats(ctx context.Context) (driver.MemoryStats, error) {
	if r == nil || r.worker == nil {
		return driver.MemoryStats{}, errors.New("inference: runner is nil")
	}
	return r.worker.MemoryStats(ctx)
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
		return nil, errors.New("inference: runner is nil")
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
		return nil, errors.New("inference: runner is nil")
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
		return nil, errors.New("inference: runner is nil")
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
		return nil, nil, errors.New("inference: runner is nil")
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
		return nil, nil, errors.New("inference: runner is nil")
	}
	return tokenizer.EncodeMarkers(
		r.vocab.Encode,
		text,
		placeholder,
		counts,
		tokenizer.EncodeOptions{AddSpecial: addSpecial, ParseSpecial: true},
	)
}

func (r *Runner) DetokenizeTokens(tokens []tokenizer.TokenID) (string, error) {
	if r == nil || r.vocab == nil {
		return "", errors.New("inference: runner is nil")
	}
	return r.vocab.Decode(tokens, false)
}

// TokenPiece: renders one vocabulary token exactly as llama.cpp's tokenizer
// endpoints do, including control-token spellings and raw byte tokens
func (r *Runner) TokenPiece(token tokenizer.TokenID) (string, error) {
	if r == nil || r.vocab == nil {
		return "", errors.New("inference: runner is nil")
	}
	return r.vocab.DecodePiece(token, true)
}

// DetokenizePromptTokens: renders exact prompt sequence for native server
// metadata while preserving control-token spellings like llama.cpp
func (r *Runner) DetokenizePromptTokens(tokens []tokenizer.TokenID) (string, error) {
	if r == nil || r.vocab == nil {
		return "", errors.New("inference: runner is nil")
	}
	return r.vocab.Decode(tokens, true)
}
