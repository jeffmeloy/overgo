package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	"io/fs"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/rae"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingprogram"
)

const (
	// captionTokens bounds the caption tokens a training sequence holds after its prompt; a caption that fits whole
	// is followed by the text model's end token, and one cut short is not.
	captionTokens = 96
	// captionFamily names the prompt a projector's training builds, as a composite's session names its own.
	captionFamily = "caption-projector"
	// captionTokensSuffix ends the name of a sample's cached frame tokens, as seam-align -capture names them.
	captionTokensSuffix = ".tokens.f32"
	// captionZeroStart and captionMeanStart name the bridges a run with none starts from.
	captionZeroStart = "zero"
	captionMeanStart = "mean"
)

// captionModel is the frozen text model a caption projector trains through: the mean loss over a sequence's trained
// tokens, each predicted from the token before it, with rows in place of the embeddings at positions, and the loss's
// gradient with respect to the rows.
type captionModel func(tokens []int, trained []bool, positions []int, rows []float32) (float64, []float32, error)

// captionDevice opens the frozen text model at path on the card for sequences of length tokens, giving its width and
// what closes it, where the platform's build holds the card.
var captionDevice func(ctx context.Context, path string, length int) (captionModel, int, func() error, error)

// captionSequence is one sample's training sequence: its clip and the identity of its frame's tokens, the prompt's
// tokens then its caption's, padded to the run's length, and which tokens the loss reads.
type captionSequence struct {
	vid     string
	frame   artifact.ID
	tokens  []int
	trained []bool
}

// trainCaptionProjector is the caption-projector route: it trains the bridge that carries an image encoder's tokens
// to a text model's placeholders, the bridge of an image composite, through the frozen text model. Each sample is
// the prompt a composite's session builds for a frame, asked the frame suite's question, and the frame's caption
// after it; the loss is the text model's over the caption's tokens, carried back through the frozen model to the
// bridge's rows and from them to the bridge. The samples are the training side seam-align -capture read, with
// their frames' tokens as it cached them, so no held-out clip is trained on, and the trained bridge publishes as a
// checkpoint.
func trainCaptionProjector(ctx context.Context, flags *flag.FlagSet, args []string) (err error) {
	capturePath := flags.String("capture", "", "the samples file seam-align -capture writes beside its capture: the training side's clips and captions, where their frame tokens are cached, and the encoder and text model it paired; the run's dataset")
	placeholder := flags.String("placeholder", "", "the text model's image placeholder token, which each of a frame's rows takes in the prompt")
	from := flags.String("from", "", "the bridge the run trains from, a safetensors file of bridge.weight and bridge.bias; when absent the run writes the -start bridge there and trains from it")
	start := flags.String("start", captionZeroStart, "the bridge a run whose -from file is absent writes there: zero, the zero-output bridge, the floor the acceptance reads; or mean, weights zero and offsets the mean of the captions' mean embeddings over the samples the run trains on, for a text model without biases, which keeps zero rows zero")
	batch := flags.Int("batch", 0, "samples a step is taken on")
	steps := clioptions.IntOverride(flags, "steps", "required observed steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	held := flags.Int("held", 0, "source videos, the first in the capture's seeded order, whose samples no step trains on and whose loss scores the run before and after in place of its first batch")
	seed := flags.Uint64("seed", 0, "the seed that orders the samples the run trains on, for an independent run of the same bridge; 0 keeps the capture's order")
	host := flags.Bool("host", false, "run the frozen text model on the host instead of the card")
	storePath := flags.String("store", "", "OvergoDB for the session observation, recipe authority and checkpoint record; empty resolves through the data root")
	out := flags.String("out", "", "directory the trained bridge's checkpoint publishes to")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if err := dataroot.ResolveStore(storePath); err != nil {
		return err
	}
	name := flags.Name()
	if *capturePath == "" || *placeholder == "" || *from == "" || *out == "" || *steps <= 0 || *batch <= 0 {
		return fmt.Errorf("%s: capture, placeholder, from, out and positive steps and batch required", name)
	}
	var capture evaluation.CaptionTrainingCapture
	if err := jsonfile.DecodeStrict(*capturePath, &capture); err != nil {
		return err
	}
	if capture.EncoderWidth <= 0 || capture.TextWidth <= 0 || len(capture.Samples) == 0 {
		return fmt.Errorf("%s: the capture holds %d samples of widths %d and %d", name, len(capture.Samples), capture.EncoderWidth, capture.TextWidth)
	}
	// The samples the run trains on are admitted before a start is written or anything is opened.
	if admitted, _ := evaluation.HeldBack(capture.Samples, capture.Samples, *held); len(admitted) < *batch {
		return fmt.Errorf("%s: %d samples are left to train on beside the held sources, fewer than a batch of %d", name, len(admitted), *batch)
	}
	if _, statErr := os.Stat(*from); errors.Is(statErr, fs.ErrNotExist) {
		first, err := captionStart(*start, *capturePath, capture, *held)
		if err != nil {
			return err
		}
		if err := rae.SaveBridgeWeights(*from, first); err != nil {
			return err
		}
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	session, err := openSessionAs(ctx, store, *from, *capturePath,
		generationContract(trainingprogram.ObjectiveTokenPrediction, recipecontract.ModalityImage, recipecontract.ModalityText, trainingprogram.MetricGeneratedAnswer))
	if err != nil {
		return err
	}
	defer session.settle(ctx, &err)
	bridge, err := rae.LoadEncoderBridge(*from)
	if err != nil {
		return err
	}
	rows, err := captionFrameRows(capture)
	if err != nil {
		return err
	}
	sequences, positions, err := captionSequences(ctx, capture, *placeholder, rows)
	if err != nil {
		return err
	}
	frozen, width, release, err := captionFrozen(ctx, capture.TextModel, len(sequences[0].tokens), *host)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, release()) }()
	if width != capture.TextWidth || len(bridge.Bias) != capture.TextWidth || len(bridge.Weight) != capture.TextWidth*capture.EncoderWidth {
		return fmt.Errorf("%s: a bridge of %d offsets and %d weights does not carry the capture's %d-wide tokens to a text model %d wide",
			name, len(bridge.Bias), len(bridge.Weight), capture.EncoderWidth, width)
	}
	plan, err := probeSession(trainingprogram.ObjectiveTokenPrediction, *steps, len(bridge.Weight)+len(bridge.Bias), *maxWall)
	if err != nil {
		return err
	}
	config := plan.Optimizer()
	trainer, err := rae.NewEncoderBridgeTrainer(bridge, config)
	if err != nil {
		return err
	}
	trained, scored := evaluation.HeldBack(capture.Samples, sequences, *held)
	seededOrder(trained, *seed)
	// With no source held, the run is held to the loss of its first batch, before and after.
	if len(scored) == 0 {
		scored = wrappedBatch(trained, 0, *batch)
	}
	// Each sample's frame tokens are read as it is taken.
	samplesOf := func(chosen []captionSequence) ([]rae.BridgeSample, error) {
		samples := make([]rae.BridgeSample, len(chosen))
		for at, sequence := range chosen {
			raw, err := os.ReadFile(filepath.Join(capture.Tokens, sequence.vid+captionTokensSuffix))
			frame, identifyErr := artifact.IdentifyBytes(artifact.KindFile, raw)
			tokens, decodeErr := media.DecodeFloat32LE(raw)
			if errors.Join(err, identifyErr, decodeErr) != nil || frame != sequence.frame || len(tokens) != rows*capture.EncoderWidth {
				return nil, fmt.Errorf("%s: %s's frame tokens are not the %d rows the capture recorded", name, sequence.vid, rows)
			}
			samples[at] = rae.BridgeSample{Tokens: tokens, Loss: func(mapped []float32, _ bool) (float64, []float32, error) {
				return frozen(sequence.tokens, sequence.trained, positions, mapped)
			}}
		}
		return samples, nil
	}
	return boundedRun{name: name, plan: plan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			samples, err := samplesOf(wrappedBatch(trained, step, *batch))
			if err != nil {
				return optimizer.ObservedStepResult{}, err
			}
			loss, err := trainer.StepThrough(ctx, samples)
			return optimizer.ObservedStepResult{
				Loss: loss, Step: step + tensor.SingletonExtent, LearningRate: config.LearningRate(step + tensor.SingletonExtent), GradientL2: trainer.GradientL2(),
			}, err
		},
		loss: func() (float64, error) {
			samples, err := samplesOf(scored)
			if err != nil {
				return 0, err
			}
			return trainer.LossThrough(ctx, samples)
		},
		// Each update consumed one batch of samples.
		publish: func(ctx context.Context) error {
			return session.publishPlanned(ctx, *out, trainer, plan.Updates()*(*batch))
		},
	}.train(ctx)
}

// captionFrozen opens the frozen text model at path for sequences of length tokens, on the card unless host, and gives
// its width and what closes it. The host walks a dense model's directory alone.
func captionFrozen(ctx context.Context, path string, length int, host bool) (captionModel, int, func() error, error) {
	if !host {
		if captionDevice == nil {
			return nil, 0, nil, errors.New("caption projector: this build holds no card; -host trains on the host")
		}
		return captionDevice(ctx, path, length)
	}
	model, err := densecausal.Load(path)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("caption projector: the host walks a dense text model's directory: %w", err)
	}
	return model.MaskedLossAndInputGradient, model.Dims.Hidden, func() error { return nil }, nil
}

// captionStart is the bridge a run with none starts from: the zero-output bridge, the floor the acceptance reads, or
// the bridge whose offsets are the mean of the captions' mean embeddings the capture holds for the samples the run
// trains on, its weights zero.
func captionStart(start, capturePath string, capture evaluation.CaptionTrainingCapture, held int) (first rae.EncoderBridge, err error) {
	first = rae.EncoderBridge{Weight: make([]float32, capture.TextWidth*capture.EncoderWidth), Bias: make([]float32, capture.TextWidth)}
	if start == captionZeroStart {
		return first, nil
	}
	if start != captionMeanStart {
		return rae.EncoderBridge{}, fmt.Errorf("caption projector: a run starts from %s or %s, not %q", captionZeroStart, captionMeanStart, start)
	}
	file, err := safetensors.OpenFile(strings.TrimSuffix(capturePath, evaluation.CaptureSamplesSuffix))
	if err != nil {
		return rae.EncoderBridge{}, err
	}
	defer func() { err = errors.Join(err, file.Close()) }()
	targets, err := safetensors.ReadF32(file.Tensors[evaluation.CaptureTarget])
	if err != nil || len(targets) != len(capture.Samples)*capture.TextWidth {
		return rae.EncoderBridge{}, errors.Join(err, fmt.Errorf("caption projector: the capture holds no %s row for each of its %d samples", evaluation.CaptureTarget, len(capture.Samples)))
	}
	rows := make([][]float32, len(capture.Samples))
	for index := range rows {
		rows[index] = targets[index*capture.TextWidth : (index+1)*capture.TextWidth]
	}
	trained, _ := evaluation.HeldBack(capture.Samples, rows, held)
	for _, row := range trained {
		for column, value := range row {
			first.Bias[column] += value / float32(len(trained))
		}
	}
	return first, nil
}

// seededOrder orders a run's samples in place by seed, keeping the capture's order for seed 0.
func seededOrder[T any](items []T, seed uint64) {
	if seed == 0 {
		return
	}
	rand.New(rand.NewPCG(seed, 0)).Shuffle(len(items), func(left, right int) { items[left], items[right] = items[right], items[left] })
}

// captionFrameRows is the count of tokens the encoder gave the capture's first frame, the rows every frame's prompt
// holds.
func captionFrameRows(capture evaluation.CaptionTrainingCapture) (int, error) {
	info, err := os.Stat(filepath.Join(capture.Tokens, capture.Samples[0].Vid+captionTokensSuffix))
	if err != nil {
		return 0, err
	}
	row := int64(capture.EncoderWidth) * int64(len(tokenBytes([]float32{0})))
	if info.Size() == 0 || info.Size()%row != 0 {
		return 0, fmt.Errorf("caption projector: %s's frame tokens hold %d bytes, not rows %d values wide", capture.Samples[0].Vid, info.Size(), capture.EncoderWidth)
	}
	return int(info.Size() / row), nil
}

// captionSequences are the samples' training sequences and the positions a frame's rows take in each: the prompt a
// composite's session builds for a frame of rows tokens asked the frame suite's question, then the caption's tokens,
// padded to one length with tokens the loss does not read.
func captionSequences(ctx context.Context, capture evaluation.CaptionTrainingCapture, placeholder string, rows int) (_ []captionSequence, _ []int, err error) {
	text, err := evaluation.OpenCaptionText(capture.TextModel)
	if err != nil {
		return nil, nil, err
	}
	defer func() { err = errors.Join(err, text.Close()) }()
	// The prompt's tokens follow from its rows alone, not from what they hold, so one picture builds it for every
	// sample, through the session a composite serves its frames through.
	session, err := projector.NewEncodedImageSession(captionFamily, placeholder, capture.TextWidth, func(context.Context, image.Image) (reference.Value, error) {
		shape, err := tensor.NewShape(uint64(capture.TextWidth), uint64(rows))
		if err != nil {
			return reference.Value{}, err
		}
		return reference.NewValue(shape, make([]float32, rows*capture.TextWidth))
	})
	if err != nil {
		return nil, nil, err
	}
	built, err := session.BuildImagePrompt(ctx, captionTokenizer{text.Encode}, image.NewGray(image.Rect(0, 0, tensor.SingletonExtent, tensor.SingletonExtent)), "", evaluation.FrameQuestion, false)
	if err = errors.Join(err, session.Close()); err != nil {
		return nil, nil, err
	}
	positions := make([]int, len(built.EmbeddingTokenIndices))
	for index, position := range built.EmbeddingTokenIndices {
		positions[index] = int(position)
	}
	length := len(built.TokenIDs) + captionTokens
	sequences := make([]captionSequence, len(capture.Samples))
	for index, sample := range capture.Samples {
		caption, err := text.Encode(sample.Caption)
		if err != nil || len(caption) == 0 {
			return nil, nil, errors.Join(err, fmt.Errorf("caption projector: %s's caption gives no tokens", sample.Vid))
		}
		if len(caption) < captionTokens {
			caption = append(caption, text.End)
		}
		caption = caption[:min(len(caption), captionTokens)]
		sequence := captionSequence{vid: sample.Vid, frame: sample.Frame, tokens: make([]int, length), trained: make([]bool, length)}
		for at, id := range built.TokenIDs {
			sequence.tokens[at] = int(id)
		}
		for at := len(built.TokenIDs); at < length; at++ {
			sequence.tokens[at] = text.End
		}
		for at, id := range caption {
			sequence.tokens[len(built.TokenIDs)+at], sequence.trained[len(built.TokenIDs)+at] = id, true
		}
		sequences[index] = sequence
	}
	return sequences, positions, nil
}

// captionTokenizer tokenizes a projector's prompt with the text model's own tokenizer, which adds no token of its own
// to a text: the prompt a model that opens with no beginning token is served.
type captionTokenizer struct{ encode func(string) ([]int, error) }

// TokenizeText is text's tokens, its added tokens read as the tokens they name.
func (t captionTokenizer) TokenizeText(text string, _, parseSpecial bool) ([]tokenizer.TokenID, error) {
	if !parseSpecial {
		return nil, errors.New("caption projector: a prompt's placeholders are read as the tokens they name")
	}
	ids, err := t.encode(text)
	if err != nil {
		return nil, err
	}
	tokens := make([]tokenizer.TokenID, len(ids))
	for index, id := range ids {
		tokens[index] = tokenizer.TokenID(id)
	}
	return tokens, nil
}
