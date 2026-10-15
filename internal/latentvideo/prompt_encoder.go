package latentvideo

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/pathidentity"
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
	mu         sync.Mutex
	key        promptEncoderKey
	checkpoint string
	tokenizer  *fixedUnigramTokenizer
	plan       encoderPlan
	rows       map[string][]float32
	// recent is the kept prompts, the one encoded longest ago first.
	recent []string
	// passes counts the passes made over the checkpoint.
	passes int
}

// promptEncoderKey names a text encoder by what a prompt's rows depend on:
// the bytes they are made from, and the policy the checkpoint does not carry.
type promptEncoderKey struct {
	content                     string
	sequenceLength, maxDistance int
	normEps                     float64
}

// content names the admitted checkpoint and the current tokenizer bytes.
// The canonical checkpoint path keeps subsequent reads on the admitted file.
func (spec TextConditioningSpec) content(admitAbsent bool) (identity, checkpoint string, err error) {
	named := sha256.New()
	archive, err := admitCheckpoint(spec.EncoderCheckpoint)
	if admitAbsent && errors.Is(err, fs.ErrNotExist) {
		fmt.Fprintln(named, "absent")
	} else if err != nil {
		return "", "", err
	} else {
		fmt.Fprintln(named, archive.Identity)
		checkpoint = archive.File.Name()
	}
	for _, name := range []string{tokenizerTable, tokenizerRoles} {
		id, err := artifact.IdentifyFile(artifact.KindFile, filepath.Join(spec.TokenizerDir, name))
		if admitAbsent && errors.Is(err, fs.ErrNotExist) {
			fmt.Fprintln(named, "absent")
			continue
		}
		if err != nil {
			return "", "", err
		}
		fmt.Fprintln(named, id)
	}
	return hex.EncodeToString(named.Sum(nil)), checkpoint, nil
}

// admitCheckpoint keeps immutable archives with the process's existing prompt cache.
// A platform without write exclusion rehashes before reuse instead of trusting metadata.
func admitCheckpoint(path string) (*pytorchzip.Admission, error) {
	canonical, err := pathidentity.Canonical(path)
	if err != nil {
		return nil, err
	}
	promptEncoders.Lock()
	defer promptEncoders.Unlock()
	if held := promptEncoders.archives[canonical]; held != nil {
		return held, nil
	}
	admitted, err := pytorchzip.Admit(canonical)
	if err != nil {
		return nil, err
	}
	if admitted.Immutable {
		promptEncoders.archives[canonical] = admitted
	} else if err := admitted.File.Close(); err != nil {
		return nil, err
	}
	return admitted, nil
}

var promptEncoders = struct {
	sync.Mutex
	byKey    map[promptEncoderKey]*promptEncoder
	archives map[string]*pytorchzip.Admission
}{byKey: map[promptEncoderKey]*promptEncoder{}, archives: map[string]*pytorchzip.Admission{}}

// promptEncoderFor gives the encoder of a text pipeline, reading its
// tokenizer and compiling its checkpoint's plan the first time it is asked
// for.
func promptEncoderFor(spec TextConditioningSpec) (*promptEncoder, error) {
	if !checked.PositiveInts(spec.SequenceLength) {
		return nil, fmt.Errorf("text conditioning: positive sequence length required")
	}
	content, checkpoint, err := spec.content(false)
	if err != nil {
		return nil, err
	}
	key := promptEncoderKey{
		content:        content,
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
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
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
	encoder := &promptEncoder{key: key, checkpoint: checkpoint, tokenizer: table, plan: plan, rows: map[string][]float32{}}
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
	// A request whose prompts are being encoded beside its model's loading waits here for that pass.
	awaited := processmeasure.Stage(ctx, promptRowsAwaited)
	encoder.mu.Lock()
	awaited()
	defer encoder.mu.Unlock()
	shared := spec.Kept != nil && spec.KeptUnder != ""
	var missing []string
	var sequences []encoderSequence
	for _, prompt := range prompts {
		if _, kept := encoder.rows[prompt]; !kept && shared && prompt == spec.Common {
			rows, err := encoder.keptCommon(ctx, spec)
			if err != nil {
				return nil, 0, stats, err
			}
			if rows != nil {
				encoder.rows[prompt] = rows
			}
		}
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
		encoded, passed, err := encodeSequencesStreamed(ctx, encoder.checkpoint, encoder.plan, sequences)
		run()
		if err != nil {
			return nil, 0, passed, err
		}
		stats = passed
		encoder.passes++
		for index, prompt := range missing {
			encoder.rows[prompt] = encoded[index]
			if shared && prompt == spec.Common {
				if err := encoder.keepCommon(ctx, spec, encoded[index]); err != nil {
					return nil, 0, passed, err
				}
			}
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

// commonRowsContract is the encoder's rows of a prompt as a store keeps them:
// each value's four bytes, the low byte first.
var commonRowsContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: "application/vnd.overgo.prompt-rows", Schema: "overgo/prompt-rows/v1",
}

// commonRowsName names the kept rows of a spec's common prompt by everything
// they depend on: the code that computes them, the bytes they are made from,
// the policy the encoder's checkpoint does not carry, and the prompt.
func (e *promptEncoder) commonRowsName(spec TextConditioningSpec) string {
	named := sha256.New()
	fmt.Fprintln(named, e.key.content, e.key.sequenceLength, e.key.maxDistance, e.key.normEps)
	fmt.Fprint(named, spec.Common)
	return "latentvideo/prompt-rows/" + spec.KeptUnder + "/" + hex.EncodeToString(named.Sum(nil))
}

// keptCommon reads the common prompt's rows from the spec's store, and gives
// none when the store holds none under their name.
func (e *promptEncoder) keptCommon(ctx context.Context, spec TextConditioningSpec) ([]float32, error) {
	defer processmeasure.Stage(ctx, "common prompt rows read")()
	id, bound, err := artifact.ResolveAlias(ctx, spec.Kept, e.commonRowsName(spec))
	if err != nil || !bound {
		return nil, err
	}
	content, found, err := artifact.ReadContent(ctx, spec.Kept, id)
	if err != nil || !found {
		return nil, err
	}
	if len(content.Data) == 0 || len(content.Data)%(binaryschema.Uint32Bytes*e.plan.Config.Dim) != 0 {
		return nil, fmt.Errorf("text conditioning: the store keeps %d bytes as the common prompt's rows, which are no rows of %d values", len(content.Data), e.plan.Config.Dim)
	}
	rows := make([]float32, len(content.Data)/binaryschema.Uint32Bytes)
	for index := range rows {
		rows[index] = binaryschema.LittleEndian.Float32(content.Data[index*binaryschema.Uint32Bytes:])
	}
	return rows, nil
}

// keepCommon writes the common prompt's rows to the spec's store under their
// name. It is not stopped by a cancellation: the rows are computed, and the
// next process reads them.
func (e *promptEncoder) keepCommon(ctx context.Context, spec TextConditioningSpec, rows []float32) error {
	content, err := commonRowsContract.OwnedContentBytes(binaryschema.LittleEndian.Float32s(rows))
	if err != nil {
		return err
	}
	name := e.commonRowsName(spec)
	_, err = artifact.Publish(context.WithoutCancel(ctx), spec.Kept, artifact.Batch{
		Key: name, Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{{Name: name, Target: content.Descriptor.ID}},
	})
	return err
}
