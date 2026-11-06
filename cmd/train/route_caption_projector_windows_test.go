package main

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestEncoderBridgeTrainsACaptionProjector runs the caption-projector route on a tiny text model, on the card and on
// the host. A run given no bridge writes the zero-output bridge and starts from it. Each sample's sequence is the
// prompt a composite's session builds for its frame, a placeholder at each of the frame's rows and then the frame
// suite's question, followed by its caption, the caption alone trained, ending in the end token when it fits whole.
// The bridge's steps descend through the frozen model, the checkpoint the store records holds the bridge in the
// layout a composite reads, moved from zero, and the card's run trains the bridge the host's does.
func TestEncoderBridgeTrainsACaptionProjector(t *testing.T) {
	cudatest.Require(t)
	const encoderWidth, textWidth, rows, steps = 6, 8, 3, 3
	const placeholder, end = "<|image_pad|>", "<|endoftext|>"
	work := t.TempDir()
	captions := []string{"a cat sat", "the dog ran far", "a big bird", strings.Repeat("red cars ", captionTokens)}
	// The tokenizer holds a symbol for every character the question and the captions write, a space as a byte-level
	// vocabulary writes it, and the placeholder and end tokens beside them.
	vocabulary := map[string]int{}
	for _, character := range evaluation.FrameQuestion + strings.Join(captions, "") {
		symbol := string(character)
		if character == ' ' {
			symbol = "Ġ"
		}
		if _, held := vocabulary[symbol]; !held {
			vocabulary[symbol] = len(vocabulary)
		}
	}
	placeholderID, endID := len(vocabulary), len(vocabulary)+1
	textModel := filepath.Join(work, "text")
	testcheck.NoError(t, os.MkdirAll(textModel, 0o755))
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: endID + 1, Hidden: textWidth, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 11,
	})
	testcheck.NoError(t, safetensors.Save(filepath.Join(textModel, trainingprogram.CheckpointWeights), weights, shapes, nil))
	frozen := testcheck.Must(os.ReadFile(filepath.Join(textModel, trainingprogram.CheckpointWeights)))(t)
	added := []map[string]any{{"id": placeholderID, "content": placeholder, "special": true}, {"id": endID, "content": end, "special": true}}
	tokenizerFile := testcheck.Must(json.Marshal(map[string]any{"model": map[string]any{"type": "BPE", "vocab": vocabulary, "merges": []string{}}, "added_tokens": added}))(t)
	config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"max_position_embeddings":256,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true,"eos_token_id":` + strconv.Itoa(endID) + `}`
	testcheck.NoError(t, os.WriteFile(filepath.Join(textModel, "tokenizer.json"), tokenizerFile, 0o600))
	testcheck.NoError(t, os.WriteFile(filepath.Join(textModel, "config.json"), []byte(config), 0o600))
	tokens := filepath.Join(work, "tokens")
	testcheck.NoError(t, os.MkdirAll(tokens, 0o700))
	draws := rand.New(rand.NewPCG(7, 13))
	capture := evaluation.CaptionTrainingCapture{Encoder: "encoder", TextModel: textModel, Acceptance: "sha256:acceptance", Tokens: tokens, EncoderWidth: encoderWidth, TextWidth: textWidth}
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
	capturePath := filepath.Join(work, "capture.samples.json")
	testcheck.NoError(t, jsonfile.Write(capturePath, capture, 0o600))

	// Serial: the runs hold the card.
	sequences, positions, err := captionSequences(t.Context(), capture, placeholder, rows)
	testcheck.NoError(t, err)
	tokenizer := testcheck.Must(hfbpe.Load(textModel))(t)
	question := testcheck.Must(tokenizer.Encode(evaluation.FrameQuestion))(t)
	if !slices.Equal(positions, []int{0, 1, 2}) {
		t.Fatalf("a frame's rows take the positions %v; want the %d placeholders that open the prompt", positions, rows)
	}
	prompt := rows + len(question)
	for index, sequence := range sequences {
		if len(sequence.tokens) != prompt+captionTokens || !slices.Equal(sequence.tokens[rows:prompt], question) {
			t.Fatalf("sample %d's sequence opens %v; want %d placeholders, the question %v and room for the caption", index, sequence.tokens[:prompt], rows, question)
		}
		for _, position := range positions {
			if sequence.tokens[position] != placeholderID {
				t.Fatalf("sample %d holds token %d at a row's position; want the placeholder %d", index, sequence.tokens[position], placeholderID)
			}
		}
		want := testcheck.Must(tokenizer.Encode(captions[index]))(t)
		if len(want) < captionTokens {
			want = append(want, endID)
		}
		want = want[:min(len(want), captionTokens)]
		var trained []int
		for at, flag := range sequence.trained {
			if flag {
				trained = append(trained, sequence.tokens[at])
			}
		}
		if !slices.Equal(trained, want) || !slices.Equal(sequence.tokens[prompt:prompt+len(want)], want) {
			t.Fatalf("sample %d trains the tokens %v; want its caption %v after the prompt, its end token after it only when it fits whole", index, trained, want)
		}
	}

	store, from := filepath.Join(work, "store"), filepath.Join(work, "from.safetensors")
	testcheck.NoError(t, testcheck.Must(overgodb.Create(store))(t).Close())
	run := func(out string, more ...string) []string {
		return append([]string{"caption-projector", "-capture", capturePath, "-placeholder", placeholder, "-from", from, "-out", out,
			"-steps", strconv.Itoa(steps), "-batch", "2", "-store", store}, more...)
	}
	if err := runRoute(t.Context(), slices.DeleteFunc(run(filepath.Join(work, "refused")), func(argument string) bool { return argument == "-from" || argument == from })); err == nil || !strings.Contains(err.Error(), "from") {
		t.Fatalf("a run naming no bridge to train from was not refused for it: %v", err)
	}
	testcheck.NoError(t, runRoute(t.Context(), run(filepath.Join(work, "card"))))
	zero := testcheck.Must(rae.LoadEncoderBridge(from))(t)
	if len(zero.Weight) != textWidth*encoderWidth || len(zero.Bias) != textWidth || slices.ContainsFunc(append(zero.Weight, zero.Bias...), func(value float32) bool { return value != 0 }) {
		t.Fatalf("the run wrote a bridge of %d weights and %d offsets to start from; want the zero-output bridge from %d channels to %d", len(zero.Weight), len(zero.Bias), encoderWidth, textWidth)
	}
	testcheck.NoError(t, runRoute(t.Context(), run(filepath.Join(work, "host"), "-host")))
	if held := testcheck.Must(os.ReadFile(filepath.Join(textModel, trainingprogram.CheckpointWeights)))(t); !bytes.Equal(held, frozen) {
		t.Fatal("training the bridge changed the text model's weights")
	}
	model := testcheck.Must(densecausal.Load(textModel))(t)
	losses := map[string]float64{}
	for _, side := range []string{"card", "host"} {
		out := filepath.Join(work, side)
		checkpoint := testcheck.Must(trainingprogram.LoadCheckpoint(out))(t)
		opened := testcheck.Must(overgodb.Open(store))(t)
		_, recorded, err := opened.Artifact(t.Context(), checkpoint.ID())
		testcheck.NoError(t, opened.Close())
		if err != nil || !recorded || checkpoint.Optimizer.Step != steps {
			t.Fatalf("the %s run's checkpoint %s is recorded %t (%v) at step %d; want it recorded at step %d", side, checkpoint.ID(), recorded, err, checkpoint.Optimizer.Step, steps)
		}
		file := testcheck.Must(safetensors.OpenFile(filepath.Join(out, trainingprogram.CheckpointWeights)))(t)
		if shape := file.Tensors["bridge.weight"].Shape; len(shape) != 2 || shape[0] != textWidth || shape[1] != encoderWidth {
			t.Fatalf("the %s run's bridge weight is %v; want a row of %d source weights for each of %d target channels", side, shape, encoderWidth, textWidth)
		}
		testcheck.NoError(t, file.Close())
		bridge := testcheck.Must(rae.LoadEncoderBridge(filepath.Join(out, trainingprogram.CheckpointWeights)))(t)
		if !slices.ContainsFunc(append(bridge.Weight, bridge.Bias...), func(value float32) bool { return value != 0 }) {
			t.Fatalf("the %s run's bridge is the zero bridge it started from", side)
		}
		// The host scores each run's bridge over every sample.
		for _, sequence := range sequences {
			frame, kept := readTokens(tokens, sequence.vid+captionTokensSuffix, rows*encoderWidth)
			mapped := testcheck.Must(bridge.Apply(frame))(t)
			loss, _, err := model.MaskedLossAndInputGradient(sequence.tokens, sequence.trained, positions, mapped)
			if !kept || err != nil {
				t.Fatalf("the host could not score the %s run's bridge on %s: %v", side, sequence.vid, err)
			}
			losses[side] += loss / float64(len(sequences))
		}
	}
	// Muon's steps carry a gradient's weakest directions at full size, so the two bridges' weights part where the
	// two backward passes round differently; what each makes of the samples does not.
	if math.Abs(losses["card"]-losses["host"]) > 1e-3*losses["host"] {
		t.Fatalf("the host scores the card's bridge %.6f and its own %.6f; want the card to train the bridge the host does", losses["card"], losses["host"])
	}
	// A frame's tokens rewritten after the capture recorded them are not trained on.
	testcheck.NoError(t, writeTokens(tokens, "clip-0"+captionTokensSuffix, make([]float32, rows*encoderWidth)))
	if err := runRoute(t.Context(), run(filepath.Join(work, "rewritten"), "-host")); err == nil || !strings.Contains(err.Error(), "the capture recorded") {
		t.Fatalf("a run over a frame rewritten after its capture was not refused for it: %v", err)
	}
}
