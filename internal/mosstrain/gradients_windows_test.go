//go:build windows

package mosstrain

import (
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentimage"
	"overgo/internal/optimizer"
	"overgo/internal/peft"
	"overgo/internal/safetensors"
)

const (
	gradientGolden       = "adapter-gradients.safetensors"
	gradientGoldenDigest = "df7ff03beb8adce77ef5ac24b6146c04120c28038e7bd45b836240b8fe48e915"
	// goldenHeight, goldenWidth and goldenText are the example of the producer.
	goldenHeight, goldenWidth, goldenText = 3, 4, 5
	// goldenTimestep and goldenSigma are its point on the training grid.
	goldenTimestep = 412
	goldenSigma    = 0.412
	// goldenTolerance bounds an F32 device value against F32 autograd, relative
	// to the largest magnitude of the tensor it is compared in: two F32
	// orderings of the same sums.
	goldenTolerance = 2e-4
)

type goldenConfig struct {
	Dim      int     `json:"dim"`
	FFN      int     `json:"ffn_dim"`
	Freq     int     `json:"freq_dim"`
	Channels int     `json:"latent_channels"`
	Heads    int     `json:"num_heads"`
	Layers   int     `json:"num_layers"`
	RopeBase float64 `json:"rope_base"`
	Text     int     `json:"text_dim"`
}

// rows turns [channels, height, width] into [height*width, channels].
func rows(planar []float32, channels, height, width int) []float32 {
	out := make([]float32, len(planar))
	for token := range height * width {
		for channel := range channels {
			out[token*channels+channel] = planar[channel*height*width+token]
		}
	}
	return out
}

// TestMOSSAdapterGradients holds a device step to PyTorch autograd of the
// pinned transformer on a small multi-head model: the prediction, the loss and
// the gradient of every factor of every adapted projection.
func TestMOSSAdapterGradients(t *testing.T) {
	cudatest.Require(t)
	path := filepath.Join("..", "..", "fixtures", "moss", gradientGolden)
	identity, err := artifact.IdentifyFile(artifact.KindFile, path)
	if err != nil || identity.DigestHex() != gradientGoldenDigest {
		t.Fatal("golden changed identity", err)
	}
	source, err := safetensors.OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	read := func(name string) []float32 {
		t.Helper()
		item, ok := source.Tensors[name]
		if !ok {
			t.Fatalf("golden lacks %s", name)
		}
		values, err := safetensors.ReadF32(item)
		if err != nil {
			t.Fatal(err)
		}
		return values
	}
	var config goldenConfig
	for _, metadata := range source.Metadata {
		if err := json.Unmarshal([]byte(metadata["config"]), &config); err != nil {
			t.Fatal(err)
		}
	}
	spec := latentimage.TransformerSpec{Layers: config.Layers, Hidden: config.Dim, Heads: config.Heads, HeadDim: config.Dim / config.Heads, Intermediate: config.FFN, TextHidden: config.Text, InChannels: config.Channels, TimestepEmbed: config.Freq, RopeTheta: config.RopeBase}
	const rank, alpha = 3, 2.0
	adapter := map[string]peft.Module{}
	for layer := range spec.Layers {
		for _, projection := range projectionNames {
			name := moduleName(layer, projection)
			down, up := read("adapter."+name+".down"), read("adapter."+name+".up")
			adapter[name] = peft.Module{Rank: rank, In: len(down) / rank, Out: len(up) / rank, Alpha: alpha, Down: down, Up: up}
		}
	}
	optimizerConfig := optimizer.Config{BaseLearningRate: 1e-3, Momentum: 0.9, Schedule: optimizer.ScheduleConstant}
	trainer, err := New(t.Context(), spec, Options{Rank: rank, Alpha: alpha}, func(name string) ([]float32, error) {
		item, ok := source.Tensors["weight."+name]
		if !ok {
			return nil, fmt.Errorf("golden lacks %s", name)
		}
		return safetensors.ReadF32(item)
	}, adapter, optimizerConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	geometry := Geometry{Height: goldenHeight, Width: goldenWidth, Text: goldenText}
	if err := trainer.Reserve(t.Context(), geometry); err != nil {
		t.Fatal(err)
	}
	latent := rows(read("latent"), spec.InChannels, goldenHeight, goldenWidth)
	noise := rows(read("noise"), spec.InChannels, goldenHeight, goldenWidth)
	noised, target := FlowPath(latent, noise, goldenSigma)
	example := Example{Geometry: geometry, Latent: latent, Refined: read("refined")}
	if err := trainer.Clear(); err != nil {
		t.Fatal(err)
	}
	prediction, err := trainer.Predict(t.Context(), example, noised, goldenTimestep)
	if err != nil {
		t.Fatal(err)
	}
	compare := func(name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s holds %d values, want %d", name, len(got), len(want))
		}
		var largest, worst float64
		for index, value := range want {
			largest = max(largest, math.Abs(float64(value)))
			difference := math.Abs(float64(got[index]) - float64(value))
			if !(difference >= 0) {
				t.Fatalf("%s[%d] is %v", name, index, got[index])
			}
			worst = max(worst, difference)
		}
		t.Logf("%-34s worst %.3e of largest %.3e", name, worst, largest)
		if worst > goldenTolerance*max(largest, 1e-6) {
			t.Fatalf("%s differs by %g of largest %g", name, worst, largest)
		}
	}
	compare("prediction", prediction, rows(read("prediction"), spec.InChannels, goldenHeight, goldenWidth))
	loss, err := trainer.Accumulate(t.Context(), example, noised, target, goldenTimestep)
	if err != nil {
		t.Fatal(err)
	}
	if want := float64(read("loss")[0]); math.Abs(loss-want) > goldenTolerance*want {
		t.Fatalf("loss %g, autograd %g", loss, want)
	}
	if _, err := trainer.Gather(1); err != nil {
		t.Fatal(err)
	}
	group := 0
	for layer := range spec.Layers {
		for _, projection := range projectionNames {
			name := moduleName(layer, projection)
			compare(name+".down", trainer.pack.Gradients(group), read("gradient."+name+".down"))
			compare(name+".up", trainer.pack.Gradients(group+1), read("gradient."+name+".up"))
			group += 2
		}
	}
}
