package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	"io/fs"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/rae"
	"overgo/internal/recipecontract"
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
)

// captionModel is the frozen text model a caption projector trains through: the mean loss over a sequence's trained
// tokens, each predicted from the token before it, with rows in place of the embeddings at positions, and the loss's
// gradient with respect to the rows.
type captionModel func(tokens []int, trained []bool, positions []int, rows []float32) (float64, []float32, error)

// captionDevice opens the frozen text model on the card for sequences of length tokens, and what closes it, where
// the platform's build holds the card.
var captionDevice func(model *densecausal.Model, length int) (captionModel, func() error, error)

// captionSequence is one sample's training sequence: its clip, the prompt's tokens then its caption's, padded to the
// run's length, and which tokens the loss reads.
type captionSequence struct {
	vid     string
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
	from := flags.String("from", "", "the bridge the run trains from, a safetensors file of bridge.weight and bridge.bias; when absent the run writes the zero-output bridge there and trains from it")
	batch := flags.Int("batch", 0, "samples a step is taken on")
	steps := clioptions.IntOverride(flags, "steps", "required observed steps")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
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
	if capture.EncoderWidth <= 0 || capture.TextWidth <= 0 || len(capture.Samples) < *batch {
		return fmt.Errorf("%s: the capture holds %d samples of widths %d and %d, for a batch of %d", name, len(capture.Samples), capture.EncoderWidth, capture.TextWidth, *batch)
	}
	// The floor the acceptance reads is the composite whose bridge gives nothing: a run with no bridge starts there.
	if _, statErr := os.Stat(*from); errors.Is(statErr, fs.ErrNotExist) {
		zero := rae.EncoderBridge{Weight: make([]float32, capture.TextWidth*capture.EncoderWidth), Bias: make([]float32, capture.TextWidth)}
		if err := rae.SaveBridgeWeights(*from, zero); err != nil {
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
	model, err := densecausal.Load(capture.TextModel)
	if err != nil {
		return err
	}
	if model.Dims.Hidden != capture.TextWidth || len(bridge.Bias) != capture.TextWidth || len(bridge.Weight) != capture.TextWidth*capture.EncoderWidth {
		return fmt.Errorf("%s: a bridge of %d offsets and %d weights does not carry the capture's %d-wide tokens to a text model %d wide",
			name, len(bridge.Bias), len(bridge.Weight), capture.EncoderWidth, model.Dims.Hidden)
	}
	rows, err := captionFrameRows(capture)
	if err != nil {
		return err
	}
	sequences, positions, err := captionSequences(ctx, capture, *placeholder, rows)
	if err != nil {
		return err
	}
	frozen := captionModel(model.MaskedLossAndInputGradient)
	if !*host {
		if captionDevice == nil {
			return fmt.Errorf("%s: this build holds no card; -host trains on the host", name)
		}
		var release func() error
		if frozen, release, err = captionDevice(model, len(sequences[0].tokens)); err != nil {
			return err
		}
		defer func() { err = errors.Join(err, release()) }()
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
	// A step's batch is the next run of the samples, wrapping at their end, each frame's tokens read as it is taken.
	batchAt := func(step int) ([]rae.BridgeSample, error) {
		samples := make([]rae.BridgeSample, *batch)
		for at := range samples {
			sequence := sequences[(step*(*batch)+at)%len(sequences)]
			tokens, kept := readTokens(capture.Tokens, sequence.vid+captionTokensSuffix, rows*capture.EncoderWidth)
			if !kept {
				return nil, fmt.Errorf("%s: %s's frame tokens are absent or not %d rows", name, sequence.vid, rows)
			}
			samples[at] = rae.BridgeSample{Tokens: tokens, Loss: func(mapped []float32, _ bool) (float64, []float32, error) {
				return frozen(sequence.tokens, sequence.trained, positions, mapped)
			}}
		}
		return samples, nil
	}
	return boundedRun{name: name, plan: plan, session: session,
		step: func(step int) (optimizer.ObservedStepResult, error) {
			samples, err := batchAt(step)
			if err != nil {
				return optimizer.ObservedStepResult{}, err
			}
			loss, err := trainer.StepThrough(ctx, samples)
			return optimizer.ObservedStepResult{
				Loss: loss, Step: step + tensor.SingletonExtent, LearningRate: config.LearningRate(step + tensor.SingletonExtent), GradientL2: trainer.GradientL2(),
			}, err
		},
		// The run is held to the loss of the first batch, before and after.
		loss: func() (float64, error) {
			samples, err := batchAt(0)
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
func captionSequences(ctx context.Context, capture evaluation.CaptionTrainingCapture, placeholder string, rows int) ([]captionSequence, []int, error) {
	vocabulary, err := hfbpe.Load(capture.TextModel)
	if err != nil {
		return nil, nil, err
	}
	var config struct {
		End *int `json:"eos_token_id"`
	}
	if err := jsonfile.Decode(filepath.Join(capture.TextModel, "config.json"), &config); err != nil || config.End == nil {
		return nil, nil, errors.Join(err, errors.New("caption projector: the text model's configuration names no single end token"))
	}
	var opening struct {
		Beginning bool `json:"add_bos_token"`
	}
	if err := jsonfile.Decode(filepath.Join(capture.TextModel, "tokenizer_config.json"), &opening); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, nil, err
	}
	if opening.Beginning {
		return nil, nil, errors.New("caption projector: the text model opens a prompt with a beginning token, which its tokenizer file does not add")
	}
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
	built, err := session.BuildImagePrompt(ctx, captionTokenizer{vocabulary}, image.NewGray(image.Rect(0, 0, tensor.SingletonExtent, tensor.SingletonExtent)), "", evaluation.FrameQuestion, false)
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
		caption, err := vocabulary.Encode(sample.Caption)
		if err != nil || len(caption) == 0 {
			return nil, nil, errors.Join(err, fmt.Errorf("caption projector: %s's caption gives no tokens", sample.Vid))
		}
		if len(caption) < captionTokens {
			caption = append(caption, *config.End)
		}
		caption = caption[:min(len(caption), captionTokens)]
		sequence := captionSequence{vid: sample.Vid, tokens: make([]int, length), trained: make([]bool, length)}
		for at, id := range built.TokenIDs {
			sequence.tokens[at] = int(id)
		}
		for at := len(built.TokenIDs); at < length; at++ {
			sequence.tokens[at] = *config.End
		}
		for at, id := range caption {
			sequence.tokens[len(built.TokenIDs)+at], sequence.trained[len(built.TokenIDs)+at] = id, true
		}
		sequences[index] = sequence
	}
	return sequences, positions, nil
}

// captionTokenizer tokenizes a projector's prompt with the text model's own tokenizer file, which adds no token of
// its own to a text: the prompt a model that opens with no beginning token is served.
type captionTokenizer struct{ vocabulary *hfbpe.Tokenizer }

// TokenizeText is text's tokens, its added tokens read as the tokens they name.
func (t captionTokenizer) TokenizeText(text string, _, parseSpecial bool) ([]tokenizer.TokenID, error) {
	if !parseSpecial {
		return nil, errors.New("caption projector: a prompt's placeholders are read as the tokens they name")
	}
	ids, err := t.vocabulary.Encode(text)
	if err != nil {
		return nil, err
	}
	tokens := make([]tokenizer.TokenID, len(ids))
	for index, id := range ids {
		tokens[index] = tokenizer.TokenID(id)
	}
	return tokens, nil
}
