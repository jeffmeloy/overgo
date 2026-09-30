package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestTrainRoutesReachTheMediaTrainers trains a tiny synthetic checkpoint
// of each media trainer through its route, the flow-matching image model on
// a manifest of one generated image and the joint speech model on a
// recorded synthesis trace, and holds each route to publishing the trained
// model as a checkpoint recorded in the store.
func TestTrainRoutesReachTheMediaTrainers(t *testing.T) {
	t.Run("diffusionimage", func(t *testing.T) {
		root := t.TempDir()
		model := writeTinyDiffusionImage(t, filepath.Join(root, "model"))
		picture := image.NewRGBA(image.Rect(0, 0, tinyImageSide, tinyImageSide))
		for x := range tinyImageSide {
			for y := range tinyImageSide {
				picture.Set(x, y, color.RGBA{R: uint8(x * tinyImageStride), G: uint8(y * tinyImageStride), B: uint8((x + y) * tinyImageStride), A: math.MaxUint8})
			}
		}
		file, err := os.Create(filepath.Join(root, "gradient.png"))
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(file, picture); err != nil {
			t.Fatal(err)
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
		manifest := filepath.Join(root, "manifest.txt")
		if err := os.WriteFile(manifest, []byte("gradient.png\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		requireRoutePublishes(t, "diffusionimage", "-model", model, "-images", manifest)
	})
	t.Run("speech", func(t *testing.T) {
		root := t.TempDir()
		model := writeTinySpeech(t, filepath.Join(root, "model"))
		trace := filepath.Join(root, "trace.json")
		recorded := `{"text":"a b","seed":7,"quantizer_calls":[{"in":{"values":[0.5,-0.25]}},{"in":{"values":[0.75,0.125]}}]}`
		if err := os.WriteFile(trace, []byte(recorded), 0o600); err != nil {
			t.Fatal(err)
		}
		requireRoutePublishes(t, "speech", "-model", model, "-trace", trace)
	})
}

// routeTestSteps is how many updates a route test trains.
const routeTestSteps = 2

// requireRoutePublishes trains route with args for routeTestSteps updates
// into a fresh store and holds it to publishing a checkpoint recorded there
// at its last step, which it returns.
func requireRoutePublishes(t *testing.T, route string, args ...string) (trainingprogram.Checkpoint, string) {
	t.Helper()
	root := t.TempDir()
	storePath, out := filepath.Join(root, "store"), filepath.Join(root, "checkpoint")
	args = append([]string{route, "-steps", fmt.Sprint(routeTestSteps), "-store", storePath, "-out", out}, args...)
	if err := runRoute(args); err != nil {
		t.Fatalf("%s route: %v", route, err)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(out)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, recorded, err := store.Artifact(t.Context(), checkpoint.ID()); err != nil || !recorded || checkpoint.Optimizer.Step != routeTestSteps {
		t.Fatalf("checkpoint %s at step %d recorded %t (%v), want the route's %d steps recorded", checkpoint.ID(), checkpoint.Optimizer.Step, recorded, err, routeTestSteps)
	}
	return checkpoint, out
}

// The generated training image is an 8x8 colour gradient.
const tinyImageSide, tinyImageStride = 8, 32

// writeTinyDiffusionImage writes the committed tiny u-vit golden model, its
// weights generated as the golden's generator does, as a checkpoint
// directory the diffusionimage loader reads, and returns the directory.
func writeTinyDiffusionImage(t *testing.T, directory string) string {
	t.Helper()
	var fixture struct {
		WeightKeys   []string
		WeightShapes map[string][]int
	}
	if err := jsonfile.Decode(testutil.FixturePath(t, "simplediffusion", "udit_tiny.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	// The golden generator's pattern: value i of key k is
	// ((i+(k+1)*7)%29-14)/128.
	const offsetStride, period, center, scale = 7, 29, 14, 128
	tensors := make(map[string][]float32, len(fixture.WeightKeys))
	for index, key := range fixture.WeightKeys {
		values := make([]float32, elementCount(fixture.WeightShapes[key]))
		for element := range values {
			values[element] = float32((element+(index+1)*offsetStride)%period-center) / scale
		}
		tensors[key] = values
	}
	writeModelDirectory(t, directory, "config.json", `{"model_type":"uvit"}`, tensors, fixture.WeightShapes)
	return directory
}

// Tiny speech geometry: model and flow width, attention heads, feed-forward
// width, text vocabulary, and latent width; every depth is one.
const (
	tinySpeechWidth, tinySpeechHeads, tinySpeechFF = 4, 2, 8
	tinySpeechVocab, tinySpeechLatent              = 6, 2
)

// writeTinySpeech writes a synthetic speech synthesis checkpoint, one
// backbone layer, one flow block, a one-layer codec transformer and a
// one-stage SEANet, with its SentencePiece tokenizer, as a model directory
// the speechsynth loader reads, and returns the directory.
func writeTinySpeech(t *testing.T, directory string) string {
	t.Helper()
	d, ff, latent, fused := tinySpeechWidth, tinySpeechFF, tinySpeechLatent, 3*tinySpeechWidth
	shapes := map[string][]int{
		"flow_lm.conditioner.embed.weight": {tinySpeechVocab, d}, "flow_lm.input_linear.weight": {d, latent},
		"flow_lm.out_norm.weight": {d}, "flow_lm.out_norm.bias": {d}, "flow_lm.out_eos.weight": {1, d}, "flow_lm.out_eos.bias": {1},
		"flow_lm.bos_emb": {latent}, "flow_lm.emb_mean": {latent}, "flow_lm.emb_std": {latent},
		"flow_lm.flow_net.input_proj.weight": {d, latent}, "flow_lm.flow_net.input_proj.bias": {d},
		"flow_lm.flow_net.cond_embed.weight": {d, d}, "flow_lm.flow_net.cond_embed.bias": {d},
		"flow_lm.flow_net.final_layer.adaLN_modulation.1.weight": {2 * d, d}, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias": {2 * d},
		"flow_lm.flow_net.final_layer.linear.weight": {latent, d}, "flow_lm.flow_net.final_layer.linear.bias": {latent},
		"mimi.quantizer.output_proj.weight": {d, latent, 1}, "mimi.upsample.convtr.convtr.weight": {d, 1, 2},
		"mimi.decoder.model.0.conv.weight": {2, d, 3}, "mimi.decoder.model.0.conv.bias": {2},
		"mimi.decoder.model.2.convtr.weight": {2, 1, 4}, "mimi.decoder.model.2.convtr.bias": {1},
		"mimi.decoder.model.3.block.1.conv.weight": {1, 1, 3}, "mimi.decoder.model.3.block.1.conv.bias": {1},
		"mimi.decoder.model.3.block.3.conv.weight": {1, 1, 1}, "mimi.decoder.model.3.block.3.conv.bias": {1},
		"mimi.decoder.model.5.conv.weight": {1, 1, 3}, "mimi.decoder.model.5.conv.bias": {1},
	}
	transformer := func(prefix string) {
		for name, shape := range map[string][]int{
			"norm1.weight": {d}, "norm1.bias": {d}, "norm2.weight": {d}, "norm2.bias": {d},
			"self_attn.in_proj.weight": {fused, d}, "self_attn.out_proj.weight": {d, d},
			"linear1.weight": {ff, d}, "linear2.weight": {d, ff},
		} {
			shapes[prefix+name] = shape
		}
	}
	transformer("flow_lm.transformer.layers.0.")
	transformer("mimi.decoder_transformer.transformer.layers.0.")
	shapes["mimi.decoder_transformer.transformer.layers.0.layer_scale_1.scale"] = []int{d}
	shapes["mimi.decoder_transformer.transformer.layers.0.layer_scale_2.scale"] = []int{d}
	for embedder := range 2 {
		prefix := fmt.Sprintf("flow_lm.flow_net.time_embed.%d.", embedder)
		for name, shape := range map[string][]int{
			"freqs": {2}, "mlp.0.weight": {d, 4}, "mlp.0.bias": {d}, "mlp.2.weight": {d, d}, "mlp.2.bias": {d}, "mlp.3.alpha": {d},
		} {
			shapes[prefix+name] = shape
		}
	}
	for name, shape := range map[string][]int{
		"in_ln.weight": {d}, "in_ln.bias": {d}, "mlp.0.weight": {d, d}, "mlp.0.bias": {d}, "mlp.2.weight": {d, d}, "mlp.2.bias": {d},
		"adaLN_modulation.1.weight": {3 * d, d}, "adaLN_modulation.1.bias": {3 * d},
	} {
		shapes["flow_lm.flow_net.res_blocks.0."+name] = shape
	}
	tensors := make(map[string][]float32, len(shapes))
	for index, name := range slices.Sorted(maps.Keys(shapes)) {
		values := make([]float32, elementCount(shapes[name]))
		for element := range values {
			values[element] = float32(math.Sin(float64(index*len(values)+element+1))) / 2
		}
		tensors[name] = values
	}
	// The latent statistics are the identity normalization.
	for element := range latent {
		tensors["flow_lm.emb_mean"][element], tensors["flow_lm.emb_std"][element] = 0, 1
	}
	config := `{"flow_lm":{"transformer":{"max_period":10000,"num_heads":2}},"mimi":{"frame_rate":12.5,"sample_rate":24000,` +
		`"seanet":{"channels":1,"compress":1,"dimension":4,"kernel_size":3,"last_kernel_size":3,"n_filters":1,"ratios":[2],"residual_kernel_size":3},` +
		`"transformer":{"context":4,"max_period":10000,"num_heads":2}}}`
	writeModelDirectory(t, directory, "pockettts_config.json", config, tensors, shapes)
	writeSentencePiece(t, filepath.Join(directory, "tokenizer.model"),
		[]string{"<pad>", "</s>", "<unk>", "▁a", "▁b", "▁c"}, 2)
	return directory
}

// writeModelDirectory writes a model directory: its declaration and its
// float32 weights.
func writeModelDirectory(t *testing.T, directory, declaration, content string, tensors map[string][]float32, shapes map[string][]int) {
	t.Helper()
	if err := os.MkdirAll(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, declaration), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights), tensors, shapes, nil); err != nil {
		t.Fatal(err)
	}
}

// elementCount is how many values a tensor of shape holds.
func elementCount(shape []int) int {
	count := 1
	for _, extent := range shape {
		count *= extent
	}
	return count
}
