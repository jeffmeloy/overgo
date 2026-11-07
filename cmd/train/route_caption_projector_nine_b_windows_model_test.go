//go:build windows && modeltest

package main

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/evaluation"
	"overgo/internal/gguf"
	"overgo/internal/jsonfile"
	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingprogram"
)

// TestTheCaptionProjectorTrainsThroughTheNineB runs the caption-projector route on Qwen3.5-9B's own file, its hybrid
// stack walked frozen on the card. Each sample's sequence reads the file's own tokenizer: the image placeholder at
// every row's position, then the caption, ending in the file's end token. The route's steps lower the loss of the
// samples they train on, the checkpoint the store records holds a bridge moved from the one it started from, and the
// model file is untouched. The run starts from a small bridge, not the zero-output one: zero rows open the prompt,
// a stack without biases keeps them zero through every layer, and each layer's norms then multiply their gradient by
// the inverse square root of their epsilon.
func TestTheCaptionProjectorTrainsThroughTheNineB(t *testing.T) {
	// Serial: holds the 9B's F32 layers on the card.
	cudatest.Require(t)
	const encoderWidth, rows, steps, scale, placeholder = 8, 4, 3, 0.02, "<|image_pad|>"
	path := modeltest.Directory(t, "Qwen3.5-9B-Q8_0.gguf")
	before := testcheck.Must(os.Stat(path))(t)
	file := testcheck.Must(gguf.Open(path))(t)
	vocabulary := testcheck.Must(tokenizer.Load(file))(t)
	spec := testcheck.Must(model.ReadSpec(file))(t)
	testcheck.NoError(t, file.Close())
	width := int(spec.EmbeddingLength)
	encode := func(text string) []int {
		ids := testcheck.Must(vocabulary.Encode(text, tokenizer.EncodeOptions{ParseSpecial: true}))(t)
		tokens := make([]int, len(ids))
		for index, id := range ids {
			tokens[index] = int(id)
		}
		return tokens
	}
	work := t.TempDir()
	tokens := filepath.Join(work, "tokens")
	testcheck.NoError(t, os.MkdirAll(tokens, 0o700))
	captions := []string{"A woman walks a dog along a beach at sunset.", "Two cars wait at a red light on a wet street."}
	capture := evaluation.CaptionTrainingCapture{Encoder: "encoder", TextModel: path, Acceptance: "sha256:acceptance", Tokens: tokens, EncoderWidth: encoderWidth, TextWidth: width}
	draws := rand.New(rand.NewPCG(5, 9))
	for index, caption := range captions {
		frame := make([]float32, rows*encoderWidth)
		for at := range frame {
			frame[at] = float32(draws.NormFloat64())
		}
		vid := "clip-" + strconv.Itoa(index)
		testcheck.NoError(t, writeTokens(tokens, vid+captionTokensSuffix, frame))
		capture.Samples = append(capture.Samples, evaluation.CapturedCaption{
			CaptionTrainingClip: evaluation.CaptionTrainingClip{Vid: vid, Source: "source-" + strconv.Itoa(index), Caption: caption},
			Frame:               testcheck.Must(artifact.IdentifyFile(artifact.KindFile, filepath.Join(tokens, vid+captionTokensSuffix)))(t),
		})
	}
	sequences, positions, err := captionSequences(t.Context(), capture, placeholder, rows)
	testcheck.NoError(t, err)
	image := encode(placeholder)
	if len(image) != 1 || len(positions) != rows {
		t.Fatalf("the placeholder reads as %v and a frame's rows take %d positions; want one token and %d", image, len(positions), rows)
	}
	for index, sequence := range sequences {
		for _, position := range positions {
			if sequence.tokens[position] != image[0] {
				t.Fatalf("sample %d holds token %d at a row's position; want the placeholder %d", index, sequence.tokens[position], image[0])
			}
		}
		var trained []int
		for at, flag := range sequence.trained {
			if flag {
				trained = append(trained, sequence.tokens[at])
			}
		}
		if want := append(encode(captions[index]), int(vocabulary.EOS)); !slices.Equal(trained, want) {
			t.Fatalf("sample %d trains the tokens %v; want its caption and the file's end token %v", index, trained, want)
		}
	}

	store, from, out := filepath.Join(work, "store"), filepath.Join(work, "from.safetensors"), filepath.Join(work, "out")
	testcheck.NoError(t, testcheck.Must(overgodb.Create(store))(t).Close())
	capturePath := filepath.Join(work, "capture.samples.json")
	testcheck.NoError(t, jsonfile.Write(capturePath, capture, 0o600))
	start := rae.EncoderBridge{Weight: make([]float32, width*encoderWidth), Bias: make([]float32, width)}
	for at := range start.Weight {
		start.Weight[at] = float32(draws.NormFloat64() * scale)
	}
	testcheck.NoError(t, rae.SaveBridgeWeights(from, start))
	// Every step trains on both samples, the batch the run's loss is read on before and after.
	testcheck.NoError(t, runRoute(t.Context(), []string{"caption-projector", "-capture", capturePath, "-placeholder", placeholder,
		"-from", from, "-out", out, "-steps", strconv.Itoa(steps), "-batch", strconv.Itoa(len(captions)), "-store", store}))
	checkpoint := testcheck.Must(trainingprogram.LoadCheckpoint(out))(t)
	opened := testcheck.Must(overgodb.Open(store))(t)
	_, recorded, err := opened.Artifact(t.Context(), checkpoint.ID())
	testcheck.NoError(t, opened.Close())
	if err != nil || !recorded || checkpoint.Optimizer.Step != steps {
		t.Fatalf("the checkpoint %s is recorded %t (%v) at step %d; want it recorded at step %d", checkpoint.ID(), recorded, err, checkpoint.Optimizer.Step, steps)
	}
	bridge := testcheck.Must(rae.LoadEncoderBridge(filepath.Join(out, trainingprogram.CheckpointWeights)))(t)
	if len(bridge.Bias) != width || slices.Equal(bridge.Weight, start.Weight) {
		t.Fatalf("the trained bridge holds %d offsets and the weights it started from or not; want %d, moved", len(bridge.Bias), width)
	}
	if after := testcheck.Must(os.Stat(path))(t); after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
		t.Fatal("training the bridge wrote the text model's file")
	}
}
