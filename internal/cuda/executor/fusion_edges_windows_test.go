//go:build windows

package executor

import (
	"context"
	"fmt"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// fusionChain is a MOSS fusion's chain as a graph builds it: its outputs, and the intermediates whose storing as
// outputs leaves the chain unfused, launched step by step.
type fusionChain struct {
	outputs, kept []*tensor.Tensor
}

// chainGraph builds a fusion's chain with its inputs: each F32 input fed a patterned value, each int8 ConvRot weight
// placed on the card once for every run that reads it (weights).
type chainGraph struct {
	*tensor.Builder
	feeds   map[*tensor.Tensor]reference.Value
	weights map[*tensor.Tensor]driver.DevicePtr
	placed  func(name string, inner, out uint64) driver.DevicePtr
	seed    int
}

// input is an F32 input of the chain, fed a patterned value of scale and bias.
func (g *chainGraph) input(name string, scale, bias float32, dimensions ...uint64) *tensor.Tensor {
	node := g.Input(name, dtype.F32, tensor.MustShape(dimensions...))
	g.seed++
	g.feeds[node] = patternedValue(node.Shape, g.seed, scale, bias)
	return node
}

// int8 is an int8 ConvRot weight of the chain, inner columns by out rows, read on the card.
func (g *chainGraph) int8(name string, inner, out uint64) *tensor.Tensor {
	node := g.Input(name, dtype.I8ConvRot, tensor.MustShape(inner, out))
	g.weights[node] = g.placed(name, inner, out)
	return node
}

// bounded is MOSS's rounded tanh bound of x (latentimage rotaryGraph.situ).
func (g *chainGraph) bounded(x *tensor.Tensor, bound float32) *tensor.Tensor {
	round := g.BF16RoundNative
	return round(g.Scale(round(g.Tanh(round(g.Scale(x, tensor.UnitScale/bound)))), bound))
}

// situ is MOSS's rounded SiTU of a gate and an up projection.
func (g *chainGraph) situ(gate, up *tensor.Tensor) (situ, boundedGate *tensor.Tensor) {
	const gateBound, hiddenBound = 4, 25
	round := g.BF16RoundNative
	boundedGate = g.bounded(gate, gateBound)
	return round(g.Multiply(round(g.Multiply(boundedGate, round(g.Sigmoid(gate)))), g.bounded(up, hiddenBound))), boundedGate
}

// rotate is MOSS's rounded rotary embedding of heads of width (latentimage rotaryGraph.rotate), and the half-turned
// heads it reads.
func (g *chainGraph) rotate(x, cosine, sine *tensor.Tensor, width, heads, tokens uint64) (rotated, turned *tensor.Tensor) {
	round := g.BF16RoundNative
	part := func(start uint64) *tensor.Tensor {
		return g.Reshape(g.GroupSlice(x, start, width/2, tensor.SingletonExtent, width/2), width/2, heads, tokens)
	}
	turned = g.Concat(g.Scale(part(width/2), -tensor.UnitScale), part(0), tensor.FirstOffset)
	return round(g.Add(round(g.Multiply(x, cosine)), round(g.Multiply(turned, sine)))), turned
}

// modulated is MOSS's rounded native-mean RMS norm of a stream with its weight, modulation scale and shift.
func (g *chainGraph) modulated(stream *tensor.Tensor, width uint64) (modulated, normed *tensor.Tensor) {
	const epsilon = 1e-6
	round := g.BF16RoundNative
	normed = round(g.RMSNormNativeMean(stream, epsilon))
	weighted := round(g.Multiply(normed, g.input("norm_weight", 0.9, 1, width)))
	return round(g.Add(round(g.Multiply(weighted, g.input("scale", 0.4, 1, width))), g.input("shift", 0.7, -0.2, width))), normed
}

// fusionEdgeCase is one MOSS fusion: what its rewrite folds (folds), and its chain built at the denoiser's shapes or
// at the kernel's edges.
type fusionEdgeCase struct {
	name  string
	folds func(*compiledFusion) bool
	build func(g *chainGraph, edge bool) fusionChain
}

// folding is the fold of a rewrite of kind.
func folding(kind compiledFusionKind) func(*compiledFusion) bool {
	return func(fusion *compiledFusion) bool { return fusion.kind == kind }
}

// mossFusionCases are the MOSS fusions, each chain built as the denoiser builds it (latentimage rotaryGraph) and as
// each fusion's own test builds it; at the edges, widths and head widths other than the kernels' and rows that no
// warp or vector width divides.
func mossFusionCases() []fusionEdgeCase {
	pick := func(edge bool, at, beyond uint64) uint64 {
		if edge {
			return beyond
		}
		return at
	}
	return []fusionEdgeCase{
		{"affine norm", folding(compiledFusionMOSSNorm), func(g *chainGraph, edge bool) fusionChain {
			const epsilon = 1e-6
			width, tokens, head, heads := pick(edge, 96, 131), pick(edge, 24, 5), pick(edge, 128, 64), uint64(3)
			round := g.BF16RoundNative
			modulated, normed := g.modulated(g.input("stream", 2.3, 0.1, width, tokens), width)
			queryNormed := round(g.RMSNormNativeMean(g.input("queries", 1.7, -0.3, head, heads, tokens), epsilon))
			query := round(g.Multiply(queryNormed, g.input("query_weight", 0.5, 1, head)))
			return fusionChain{outputs: []*tensor.Tensor{modulated, query}, kept: []*tensor.Tensor{normed, queryNormed}}
		}},
		{"gated add", folding(compiledFusionGatedAdd), func(g *chainGraph, edge bool) fusionChain {
			width, tokens := pick(edge, 96, 97), pick(edge, 24, 5)
			round := g.BF16RoundNative
			gated := round(g.Multiply(g.input("update", 1.9, -0.4, width, tokens), g.input("gate", 0.6, 0.2, width)))
			return fusionChain{outputs: []*tensor.Tensor{round(g.Add(g.input("stream", 2.3, 0.1, width, tokens), gated))}, kept: []*tensor.Tensor{gated}}
		}},
		{"norm and rotation", folding(compiledFusionMOSSNormRotary), func(g *chainGraph, edge bool) fusionChain {
			const epsilon = 1e-6
			width, heads, tokens := pick(edge, 128, 64), uint64(3), pick(edge, 12, 5)
			round := g.BF16RoundNative
			query := g.input("query", 1.7, -0.3, width, heads, tokens)
			normed := round(g.Multiply(round(g.RMSNormNativeMean(query, epsilon)), g.input("weight", 0.5, 1, width)))
			rotated, _ := g.rotate(normed, g.input("cosine", 0.11, 0.02, width, 1, tokens), g.input("sine", 0.09, -0.01, width, 1, tokens), width, heads, tokens)
			return fusionChain{outputs: []*tensor.Tensor{rotated}, kept: []*tensor.Tensor{normed}}
		}},
		{"rounded rotary", folding(compiledFusionMOSSRotary), func(g *chainGraph, edge bool) fusionChain {
			width, heads, tokens := pick(edge, 64, 96), pick(edge, 4, 3), pick(edge, 12, 5)
			value := g.input("value", 0.37, 0.01, width, heads, tokens)
			rotated, turned := g.rotate(value, g.input("cosine", 0.11, 0.02, width, 1, tokens), g.input("sine", 0.09, -0.01, width, 1, tokens), width, heads, tokens)
			return fusionChain{outputs: []*tensor.Tensor{g.BF16RoundNative(rotated)}, kept: []*tensor.Tensor{turned}}
		}},
		{"SiTU", folding(compiledFusionMOSSSitu), func(g *chainGraph, edge bool) fusionChain {
			width, tokens := pick(edge, 96, 97), pick(edge, 24, 5)
			situ, boundedGate := g.situ(g.input("gate", 0.83, 0.05, width, tokens), g.input("up", 3.7, -0.4, width, tokens))
			return fusionChain{outputs: []*tensor.Tensor{situ}, kept: []*tensor.Tensor{boundedGate}}
		}},
		{"SiTU of a product's halves", func(fusion *compiledFusion) bool {
			return fusion.kind == compiledFusionMOSSSitu && fusion.situ.halves != 0
		}, func(g *chainGraph, edge bool) fusionChain {
			width, tokens := pick(edge, 96, 97), pick(edge, 24, 5)
			product := g.input("product", 2.9, -0.2, 2*width, tokens)
			half := func(offset uint64) *tensor.Tensor {
				return g.Reshape(g.GroupSlice(product, offset, width, tensor.SingletonExtent, width), width, tokens)
			}
			gate, up := half(width), half(0)
			situ, _ := g.situ(gate, up)
			return fusionChain{outputs: []*tensor.Tensor{situ}, kept: []*tensor.Tensor{gate, up}}
		}},
		{"int8 prologues", folding(compiledFusionInt8Prologue), func(g *chainGraph, edge bool) fusionChain {
			const inner = 256
			out, tokens := pick(edge, 64, 40), pick(edge, 24, 5)
			modulated, _ := g.modulated(g.input("stream", 2.3, 0.1, inner, tokens), inner)
			fused := g.input("fused", 2.9, -0.2, 2*inner, tokens)
			half := func(offset uint64) *tensor.Tensor {
				return g.Reshape(g.GroupSlice(fused, offset, inner, tensor.SingletonExtent, inner), inner, tokens)
			}
			situ, _ := g.situ(half(inner), half(0))
			return fusionChain{outputs: []*tensor.Tensor{
				g.MulMat(g.int8("int8_a", inner, out), modulated), g.MulMat(g.int8("int8_b", inner, out), modulated), g.MulMat(g.int8("int8_c", inner, out), situ),
			}, kept: []*tensor.Tensor{modulated, situ}}
		}},
		{"int8 gated add", folding(compiledFusionInt8GatedAdd), func(g *chainGraph, edge bool) fusionChain {
			const inner = 256
			tokens := pick(edge, 24, 5)
			round := g.BF16RoundNative
			gates := g.input("gates", 0.6, 0.2, 2*inner)
			products := []*tensor.Tensor{
				g.MulMat(g.int8("output_weight", inner, inner), g.input("attended", 1.1, -0.1, inner, tokens)),
				g.MulMat(g.int8("down_weight", inner, inner), g.input("hidden", 2.9, -0.2, inner, tokens)),
			}
			hidden := round(g.Add(g.input("stream", 2.3, 0.1, inner, tokens), round(g.Multiply(products[0], g.FlatSlice(gates, 0, inner)))))
			sum := round(g.Add(hidden, round(g.Multiply(products[1], g.FlatSlice(gates, inner, inner)))))
			return fusionChain{outputs: []*tensor.Tensor{hidden, sum}, kept: products}
		}},
		{"int8 norm and rotation", folding(compiledFusionInt8NormRotary), func(g *chainGraph, edge bool) fusionChain {
			const inner, epsilon = 256, 1e-6
			head, heads, tokens := pick(edge, 128, 64), pick(edge, 2, 3), pick(edge, 24, 5)
			round := g.BF16RoundNative
			stream := g.input("stream", 2.3, 0.1, inner, tokens)
			cosine, sine := g.input("cosine", 0.11, 0.02, head, 1, tokens), g.input("sine", 0.09, -0.01, head, 1, tokens)
			var chain fusionChain
			for _, name := range []string{"a", "b"} {
				projected := g.Reshape(g.MulMat(g.int8("int8_"+name, inner, head*heads), stream), head, heads, tokens)
				normed := round(g.Multiply(round(g.RMSNormNativeMean(projected, epsilon)), g.input("norm_"+name, 0.5, 1, head)))
				rotated, _ := g.rotate(normed, cosine, sine, head, heads, tokens)
				chain.outputs, chain.kept = append(chain.outputs, rotated), append(chain.kept, projected)
			}
			return chain
		}},
		{"rounded elementwise", folding(compiledFusionRoundedElementwise), func(g *chainGraph, edge bool) fusionChain {
			width, tokens := pick(edge, 64, 65), pick(edge, 16, 3)
			left, right := g.input("left", 0.37, 0.01, width, tokens), g.input("right", 0.29, -0.02, width, tokens)
			row := g.input("row", 0.11, 0.5, width, 1)
			var chain fusionChain
			round := func(product *tensor.Tensor) *tensor.Tensor {
				chain.kept = append(chain.kept, product)
				chain.outputs = append(chain.outputs, g.BF16Round(product))
				return chain.outputs[len(chain.outputs)-1]
			}
			value := round(g.Multiply(round(g.Add(round(g.Multiply(round(g.Add(left, right)), right)), row)), row))
			round(g.RMSNormNativeMean(round(g.Sigmoid(round(g.Tanh(round(g.Scale(value, 0.7)))))), 1e-6))
			return chain
		}},
	}
}

// chainRun is how a fusion's chain is run: at the kernel's edges, with each kept intermediate read by a second reader
// as well (shared), with its intermediates kept as outputs, which leaves it unfused (keep), and with each F32 input fed
// as a device view at an address aligned to 4 bytes and not 16 (views).
type chainRun struct {
	edge, shared, keep, views bool
}

// fusionViewOffset places a device view of an input 4 bytes into its allocation, which the card aligns to 256.
const fusionViewOffset = 4

// TestEachMOSSFusionMatchesItsChainWhereTheyCouldPart holds each MOSS fusion to its chain launched step by step, bit
// for bit, as matchesUnfused does, at the conditions that can tell them apart (fusion-edges): as the denoiser builds
// it, where it folds; with each intermediate the fusion leaves unstored read by a second reader too; with every F32
// input fed as a device view at an address aligned to 4 bytes and not 16, as a session may bind a slice of another
// tensor; and at shapes beyond the kernels' own, head widths other than the warp kernel's and rows that no warp or
// vector width divides. The fusions' own tests had fed the one condition the denoiser makes, which let review find a
// norm two rotations read left unstored for one (SQA-433) and a kernel reading its tables four floats at a time from
// an address it does not check (SQA-436).
func TestEachMOSSFusionMatchesItsChainWhereTheyCouldPart(t *testing.T) {
	// Serial: device fixtures share the one card.
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	placed := map[string]driver.DevicePtr{}
	weight := func(name string, inner, out uint64) driver.DevicePtr {
		key := fmt.Sprintf("%s %dx%d", name, inner, out)
		if _, held := placed[key]; !held {
			placed[key] = copyFixtureDeviceBytes(t, worker, int8WeightBytes(len(placed)+1, int(inner), int(out)))
		}
		return placed[key]
	}
	run := func(t *testing.T, fusion fusionEdgeCase, how chainRun) [][]float32 {
		t.Helper()
		g := &chainGraph{Builder: tensor.NewBuilder(), feeds: map[*tensor.Tensor]reference.Value{}, weights: map[*tensor.Tensor]driver.DevicePtr{}, placed: weight}
		chain := fusion.build(g, how.edge)
		outputs := chain.outputs
		if how.shared {
			for _, kept := range chain.kept {
				outputs = append(outputs, g.Scale(kept, 0.5))
			}
		}
		compiledOutputs := outputs
		if how.keep {
			compiledOutputs = append(append([]*tensor.Tensor(nil), outputs...), chain.kept...)
		}
		testcheck.NoError(t, g.Err())
		compiled := testcheck.Must(Compile(compiledOutputs...))(t)
		folded := 0
		for _, frame := range compiled.nodes {
			if frame.fusion != nil && fusion.folds(frame.fusion) {
				folded++
			}
		}
		if how.keep && folded != 0 || !how.keep && !how.edge && !how.shared && folded == 0 {
			t.Fatalf("%d of the chain's steps fold as %s, keeping its intermediates %t", folded, fusion.name, how.keep)
		}
		inputs, host := compiled.NewDeviceInputs(), g.feeds
		for node, pointer := range g.weights {
			testcheck.NoError(t, inputs.Set(node, pointer))
		}
		if how.views {
			host = map[*tensor.Tensor]reference.Value{}
			for node, value := range g.feeds {
				view := copyFixtureDeviceBytes(t, worker, append(make([]byte, fusionViewOffset), driver.Bytes(value.Data)...)) + fusionViewOffset
				testcheck.NoError(t, inputs.Set(node, view))
			}
		}
		got := testcheck.Must(cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), compiled, host, inputs))(t)
		values := make([][]float32, len(outputs))
		for index, output := range outputs {
			values[index] = got[output].Data
		}
		return values
	}
	for _, fusion := range mossFusionCases() {
		for _, condition := range []struct {
			name string
			how  chainRun
		}{
			{"as the denoiser builds it", chainRun{}},
			{"its intermediates read again", chainRun{shared: true}},
			{"its inputs device views aligned to 4 bytes", chainRun{views: true}},
			{"at the kernels' edges", chainRun{edge: true}},
		} {
			t.Run(fusion.name+"/"+condition.name, func(t *testing.T) {
				unfused := condition.how
				unfused.keep, unfused.views = true, false
				fused, launched := run(t, fusion, condition.how), run(t, fusion, unfused)
				for index := range launched {
					compareBits(t, fused[index], launched[index])
				}
			})
		}
	}
	testcheck.NoError(t, worker.Do(context.WithoutCancel(t.Context()), func(state *device.State) error {
		return state.Driver.StreamSynchronize(state.Stream)
	}))
}
