package latentvideo

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"

	"overgo/internal/checked"
	"overgo/internal/processmeasure"
	"overgo/internal/pytorchzip"
)

// keptPrompts bounds the prompts whose rows one encoder keeps. A prompt's
// rows are sixteen kilobytes a token, so the rows kept of this many prompts
// of thirty tokens are thirty megabytes.
const keptPrompts = 64

// promptEncoder is what encoding prompts with one text encoder needs, kept
// once it has been read: the tokenizer, the checkpoint's plan, and the rows of
// the prompts it encoded last. The encoder's rows of a prompt depend on the
// prompt and the encoder alone, so an instruction given again with another
// seed, or the negative prompt every clip of a model shares, is encoded once
// by a process and not once a request.
type promptEncoder struct {
	mu        sync.Mutex
	tokenizer *fixedUnigramTokenizer
	plan      encoderPlan
	rows      map[string][]float32
	// recent is the kept prompts, the one encoded longest ago first.
	recent []string
	// passes counts the passes made over the checkpoint.
	passes int
}

// promptEncoderKey names a text encoder by what a prompt's rows depend on: its
// files as they stand on disk, and the policy the checkpoint does not carry.
type promptEncoderKey struct {
	tokenizer, checkpoint       fileState
	sequenceLength, maxDistance int
	normEps                     float64
}

// fileState is a file as a change to it shows: its place, size and time.
type fileState struct {
	path     string
	bytes    int64
	modified time.Time
}

func stateOf(path string) (fileState, error) {
	info, err := os.Stat(path)
	if err != nil {
		return fileState{}, err
	}
	return fileState{path: path, bytes: info.Size(), modified: info.ModTime()}, nil
}

var promptEncoders = struct {
	sync.Mutex
	byKey map[promptEncoderKey]*promptEncoder
}{byKey: map[promptEncoderKey]*promptEncoder{}}

// promptEncoderFor gives the encoder of a text pipeline, reading its
// tokenizer and compiling its checkpoint's plan the first time it is asked
// for.
func promptEncoderFor(spec TextConditioningSpec) (*promptEncoder, error) {
	if !checked.PositiveInts(spec.SequenceLength) {
		return nil, fmt.Errorf("text conditioning: positive sequence length required")
	}
	tokenizer, err := stateOf(filepath.Join(spec.TokenizerDir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	checkpoint, err := stateOf(spec.EncoderCheckpoint)
	if err != nil {
		return nil, err
	}
	key := promptEncoderKey{
		tokenizer: tokenizer, checkpoint: checkpoint,
		sequenceLength: spec.SequenceLength, maxDistance: spec.RelativeMaxDistance, normEps: spec.NormEps,
	}
	promptEncoders.Lock()
	defer promptEncoders.Unlock()
	if encoder, ok := promptEncoders.byKey[key]; ok {
		return encoder, nil
	}
	table, err := loadFixedUnigramTokenizer(spec.TokenizerDir, spec.SequenceLength)
	if err != nil {
		return nil, err
	}
	catalog, err := pytorchzip.ReadCatalog(spec.EncoderCheckpoint)
	if err != nil {
		return nil, err
	}
	plan, err := CompileEncoderPlan(catalog.Tensors, encoderConfig{RelativeMaxDistance: spec.RelativeMaxDistance, NormEps: spec.NormEps})
	if err != nil {
		return nil, err
	}
	if !plan.OK {
		return nil, fmt.Errorf("text conditioning encoder contract: missing=%v unexpected=%v dtype=%v shape=%v", plan.Missing, plan.Unexpected, plan.DTypeMismatches, plan.ShapeMismatches)
	}
	encoder := &promptEncoder{tokenizer: table, plan: plan, rows: map[string][]float32{}}
	promptEncoders.byKey[key] = encoder
	return encoder, nil
}

// encodePrompts gives the text encoder's rows of each prompt, the rows of its
// tokens before any projection, with their width. A prompt this process has
// encoded with this encoder is given the rows kept of it; the others are
// encoded together in one pass over the checkpoint, whose measures it
// returns, zero when no pass was made. The rows are the encoder's to keep: a
// caller reads them.
func encodePrompts(ctx context.Context, spec TextConditioningSpec, prompts ...string) ([][]float32, int, encoderStats, error) {
	var stats encoderStats
	if err := ctx.Err(); err != nil {
		return nil, 0, stats, err
	}
	encoder, err := promptEncoderFor(spec)
	if err != nil {
		return nil, 0, stats, err
	}
	encoder.mu.Lock()
	defer encoder.mu.Unlock()
	var missing []string
	var sequences []encoderSequence
	for _, prompt := range prompts {
		if _, kept := encoder.rows[prompt]; kept || slices.Contains(missing, prompt) {
			continue
		}
		tokens, mask, err := encoder.tokenizer.EncodeWithMask(prompt)
		if err != nil {
			return nil, 0, stats, err
		}
		missing, sequences = append(missing, prompt), append(sequences, encoderSequence{tokens: tokens, mask: mask})
	}
	if len(missing) > 0 {
		run := processmeasure.Stage(ctx, "text encoder run")
		encoded, passed, err := encodeSequencesStreamed(ctx, spec.EncoderCheckpoint, encoder.plan, sequences)
		run()
		if err != nil {
			return nil, 0, passed, err
		}
		stats = passed
		encoder.passes++
		for index, prompt := range missing {
			encoder.rows[prompt] = encoded[index]
		}
	}
	rows := make([][]float32, len(prompts))
	for index, prompt := range prompts {
		rows[index] = encoder.rows[prompt]
		// A prompt asked for is the most recent, kept before or not.
		encoder.recent = append(slices.DeleteFunc(encoder.recent, func(held string) bool { return held == prompt }), prompt)
	}
	for len(encoder.recent) > keptPrompts {
		delete(encoder.rows, encoder.recent[0])
		encoder.recent = encoder.recent[1:]
	}
	return rows, encoder.plan.Config.Dim, stats, nil
}
