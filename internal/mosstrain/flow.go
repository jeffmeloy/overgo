package mosstrain

import (
	"fmt"
	"math"
	"math/rand/v2"

	"overgo/internal/latentimage"
	"overgo/internal/peft"
)

// projectionNames are the projections of a joint block an adapter changes, as the
// checkpoint names them under layers.N.proj.
var projectionNames = []string{"q", "k", "v", "o", "ffn_up_gate", "ffn_down"}

// projections is the count of projectionNames; a block's arrays are indexed by their order.
const projections = 6

// gatedFeed widens a joint block's feed-forward: its up projection writes a value and its gate.
const gatedFeed = 2

// uniformSpan is the width of the [-1, 1) range an initial down factor draws from.
const uniformSpan = 2

const (
	projectionQ = iota
	projectionK
	projectionV
	projectionO
	projectionUpGate
	projectionDown
)

// Geometry is one training picture's token counts.
type Geometry struct {
	Height, Width, Text int
}

func (g Geometry) image() int  { return g.Height * g.Width }
func (g Geometry) tokens() int { return g.image() + g.Text }

// Example is what the frozen path gives one captioned picture: its packed
// latent rows [image tokens, latent channels] and refined text rows
// [text tokens, hidden width].
type Example struct {
	Geometry Geometry
	Latent   []float32
	Refined  []float32
}

// Options fix an adapter's shape.
type Options struct {
	Rank  int
	Alpha float64
}

// projectionShape is the [out, in] of a block's projection.
func projectionShape(spec latentimage.TransformerSpec, slot int) (out, in int) {
	switch slot {
	case projectionUpGate:
		return gatedFeed * spec.Intermediate, spec.Hidden
	case projectionDown:
		return spec.Hidden, spec.Intermediate
	}
	return spec.Hidden, spec.Hidden
}

// moduleName is the adapter module of a block's projection, the checkpoint's own path.
func moduleName(layer int, projection string) string {
	return fmt.Sprintf("layers.%d.proj.%s", layer, projection)
}

// InitialAdapter is PEFT's start for every projection: a down factor drawn
// uniformly from the width of its input and an up factor of zero, so an adapter
// starts as no change and only its up factors move at the first step.
func InitialAdapter(spec latentimage.TransformerSpec, options Options, seed uint64) map[string]peft.Module {
	stream := rand.New(rand.NewPCG(seed, uint64(options.Rank)))
	modules := make(map[string]peft.Module, spec.Layers*projections)
	for layer := range spec.Layers {
		for slot, projection := range projectionNames {
			out, in := projectionShape(spec, slot)
			down := make([]float32, options.Rank*in)
			bound := 1 / math.Sqrt(float64(in))
			for index := range down {
				down[index] = float32((uniformSpan*stream.Float64() - 1) * bound)
			}
			modules[moduleName(layer, projection)] = peft.Module{Rank: options.Rank, In: in, Out: out, Alpha: options.Alpha, Down: down, Up: make([]float32, out*options.Rank)}
		}
	}
	return modules
}

// FlowPath is the source's flow-matching path at sigma: the noised latent
// (1-sigma)*latent + sigma*noise and the velocity noise-latent the transformer
// predicts there.
func FlowPath(latent, noise []float32, sigma float64) (noised, target []float32) {
	noised, target = make([]float32, len(latent)), make([]float32, len(latent))
	kept, added := float32(1-sigma), float32(sigma)
	for index, value := range latent {
		noised[index] = kept*value + added*noise[index]
		target[index] = noise[index] - value
	}
	return noised, target
}

// ParameterCount is how many values an adapter of options trains.
func ParameterCount(spec latentimage.TransformerSpec, options Options) int {
	count := 0
	for layer := 0; layer < spec.Layers; layer++ {
		for slot := range projectionNames {
			out, in := projectionShape(spec, slot)
			count += options.Rank * (out + in)
		}
	}
	return count
}
