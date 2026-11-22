//go:build windows

package executor

import (
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// modelFusionCases are the fusions the text models, Krea, the video models and MOSS's projections and attention make
// (fusion-edges-all), each chain built as its model builds it; at the edges, the kind's own: a batch of one token, a
// cache's last slot, rows and widths no warp, vector or quantized block divides, a row longer than a kernel holds.
func modelFusionCases() []fusionEdgeCase {
	const epsilon = 1e-6
	// quantized is a Q8_0 product of a decode step's or a verify span's columns reading x, as a text model's projection
	// reads its norm or activation, where x's rows hold whole blocks.
	quantized := func(g *chainGraph, name string, x *tensor.Tensor, rows uint64) []*tensor.Tensor {
		if x.Shape.Dims[0]%32 != 0 {
			return nil
		}
		return []*tensor.Tensor{g.MulMat(g.weight(name, dtype.Q8_0, x.Shape.Dims[0], rows), x)}
	}
	// cached is an F32 cache of capacity tokens of width by heads appended at active (tensor Builder.AppendCache).
	cached := func(g *chainGraph, name string, width, heads, capacity uint64) *tensor.Tensor {
		return g.input(name, 0.21, 0.03, width, heads, capacity)
	}
	return []fusionEdgeCase{
		{"weighted RMS norm", 2, folding(compiledFusionWeightedRMS), func(g *chainGraph, edge bool) fusionChain {
			// A last layer's output norm of its residual sum, which the norm alone reads (routedlm's decode graph), and its
			// Q8_0 projection; at the edge, a row longer than the kernel holds in registers, as the hyper draft's.
			width, tokens := pick(edge, 256, 4128), pick(edge, q8InputSpanColumns, 1)
			sum := g.Add(g.input("residual", 2.3, 0.1, width, tokens), g.input("update", 1.9, -0.4, width, tokens))
			normed := g.RMSNorm(sum, epsilon)
			weighted := g.Multiply(normed, g.input("weight", 0.5, 1, width))
			// A gated hyper-connection mix's norm (model block_gated_hyper), each stream weighed by its own row, its
			// projection reading it through a reshape.
			streamWidth, streams := pick(edge, 96, 97), uint64(4)
			streamNormed := g.RMSNorm(g.input("streams", 1.3, 0.2, streamWidth, streams, tokens), epsilon)
			streamWeighted := g.Multiply(streamNormed, g.Reshape(g.input("stream_weight", 0.5, 1, streamWidth*streams), streamWidth, streams))
			outputs := []*tensor.Tensor{weighted, streamWeighted}
			outputs = append(outputs, quantized(g, "projection", weighted, pick(edge, 64, 96))...)
			outputs = append(outputs, quantized(g, "mixer", g.Reshape(streamWeighted, streamWidth*streams, tokens), 64)...)
			return fusionChain{outputs: outputs, kept: []*tensor.Tensor{normed, sum, streamNormed}}
		}},
		{"activated gate", 2, folding(compiledFusionActivatedGate), func(g *chainGraph, edge bool) fusionChain {
			// A feed-forward's SwiGLU and its Q8_0 down product, and an attention's sigmoid output gate.
			width, tokens := pick(edge, 512, 97), pick(edge, q8InputSpanColumns, 1)
			silu := g.SiLU(g.input("gate", 0.83, 0.05, width, tokens))
			swiglu := g.Multiply(silu, g.input("up", 3.7, -0.4, width, tokens))
			sigmoid := g.Sigmoid(g.input("attention_gate", 0.61, -0.2, width, tokens))
			gated := g.Multiply(g.input("attention", 1.9, 0.3, width, tokens), sigmoid)
			outputs := append([]*tensor.Tensor{swiglu, gated}, quantized(g, "down", swiglu, 64)...)
			return fusionChain{outputs: outputs, kept: []*tensor.Tensor{silu, sigmoid}}
		}},
		{"weighted RMS gate", 2, folding(compiledFusionWeightedRMSGate), func(g *chainGraph, edge bool) fusionChain {
			// A gated delta layer's output norm over its value heads, gated by its z (model block, gatedDelta), the SiLU's
			// and the separate sigmoid's, reshaped for its Q8_0 output product.
			state, heads, tokens := pick(edge, 128, 97), uint64(3), pick(edge, q8InputSpanColumns, 1)
			var outputs, kept []*tensor.Tensor
			for _, activate := range []func(*tensor.Tensor) *tensor.Tensor{g.SiLU, g.Sigmoid} {
				weighted := g.Multiply(g.RMSNorm(g.input("core", 1.7, -0.3, state, heads, tokens), epsilon), g.input("norm", 0.5, 1, state))
				gated := g.Multiply(weighted, activate(g.input("z", 0.83, 0.05, state, heads, tokens)))
				outputs = append(append(outputs, gated), quantized(g, "output", g.Reshape(gated, state*heads, tokens), 64)...)
				kept = append(kept, weighted)
			}
			return fusionChain{outputs: outputs, kept: kept}
		}},
		{"tanh GELU", 2, folding(compiledFusionGELUTanh), func(g *chainGraph, edge bool) fusionChain {
			// A feed-forward's biased expansion, the bias folded (Wan's, Krea's timestep at one token), and an unbiased
			// activation (UMT5's) (tensor Builder.GELUTanhExact).
			width, tokens := pick(edge, 256, 97), pick(edge, 24, 1)
			biased := g.Add(g.input("projection", 1.1, -0.2, width, tokens), g.input("bias", 0.3, 0.05, width))
			activated, plain := g.GELUTanhExact(biased), g.GELUTanhExact(g.input("hidden", 1.7, 0.1, width, tokens))
			return fusionChain{outputs: []*tensor.Tensor{activated, plain}, kept: []*tensor.Tensor{biased, activated.Inputs[1], plain.Inputs[1]}}
		}},
		{"layer norm modulate", 2, folding(compiledFusionLayerNormModulate), func(g *chainGraph, edge bool) fusionChain {
			// A Wan block's adaptive shift and scale, each a slice of its modulation and conditioning's sum, and an affine
			// norm (tensor Builder.AdaptiveShiftScale, AffineLayerNorm).
			width, tokens := pick(edge, 256, 97), pick(edge, 24, 1)
			modulation := g.Add(g.input("modulation", 0.4, 0.1, 6*width), g.input("conditioning", 0.3, -0.05, 6*width))
			chunk := func(index uint64) *tensor.Tensor { return g.FlatSlice(modulation, index*width, width) }
			normed := g.LayerNorm(g.input("stream", 2.3, 0.1, width, tokens), epsilon)
			adaptive := g.AdaptiveShiftScale(normed, chunk(0), chunk(1))
			affineNormed := g.LayerNorm(g.input("hidden", 1.9, -0.4, width, tokens), epsilon)
			affine := g.Add(g.Multiply(affineNormed, g.input("weight", 0.5, 1, width)), g.input("bias", 0.3, 0.05, width))
			return fusionChain{outputs: []*tensor.Tensor{adaptive, affine}, kept: []*tensor.Tensor{normed, affineNormed}}
		}},
		{"scaled activation", 2, folding(compiledFusionScaledActivation), func(g *chainGraph, edge bool) fusionChain {
			// A gated hyper-connection mix's low-rank SiLU of its scaled down product, quantized for its up product, and a
			// sigmoid of a scaled value.
			width, tokens := pick(edge, 320, 96), pick(edge, q8InputSpanColumns, 1)
			scaled := g.Scale(g.input("low_rank", 1.3, -0.1, width, tokens), 0.25)
			silu := g.SiLU(scaled)
			scaledSigmoid := g.Scale(g.input("other", 0.9, 0.2, width, tokens), 2)
			sigmoid := g.Sigmoid(scaledSigmoid)
			outputs := append([]*tensor.Tensor{silu, sigmoid}, quantized(g, "up", silu, pick(edge, 64, 33))...)
			return fusionChain{outputs: outputs, kept: []*tensor.Tensor{scaled, scaledSigmoid}}
		}},
		{"stream sum", 1, folding(compiledFusionStreamSum), func(g *chainGraph, edge bool) fusionChain {
			// A gated hyper-connection mix's reduction over its streams of its sigmoid-gated norm (model
			// block_gated_hyper), the gate folded, and its Q8_0 mixer product.
			width, streams, tokens := pick(edge, 256, 96), uint64(4), pick(edge, q8InputSpanColumns, 1)
			gated := g.Multiply(g.input("normalized", 1.7, -0.4, width*streams, tokens), g.Sigmoid(g.input("gate", 0.83, 0.05, width*streams, tokens)))
			sum := g.Reshape(g.GroupSlice(gated, 0, width, tensor.SingletonExtent, width), width, tokens)
			for stream := uint64(1); stream < streams; stream++ {
				sum = g.Add(sum, g.Reshape(g.GroupSlice(gated, stream*width, width, tensor.SingletonExtent, width), width, tokens))
			}
			mixed := g.Scale(sum, 1/float32(streams))
			outputs := append([]*tensor.Tensor{mixed}, quantized(g, "mixer", mixed, 64)...)
			return fusionChain{outputs: outputs, kept: []*tensor.Tensor{sum, gated}}
		}},
		{"stream combine", 1, folding(compiledFusionStreamCombine), func(g *chainGraph, edge bool) fusionChain {
			// A hyper-connection combine (model graph_dispatch): the residual and the current row repeated over the streams,
			// gated by its scaled sigmoid injection.
			width, streams, tokens := pick(edge, 256, 97), uint64(4), pick(edge, q8InputSpanColumns, 1)
			repeated := g.RepeatHeads(g.Reshape(g.input("current", 1.9, -0.4, width, tokens), width, tensor.SingletonExtent, tokens), uint32(streams))
			injection := g.Reshape(g.Scale(g.Sigmoid(g.Scale(g.input("injection", 0.7, 0.1, streams, tokens), 1/float32(streams))), 2), tensor.SingletonExtent, streams, tokens)
			product := g.Multiply(repeated, injection)
			return fusionChain{outputs: []*tensor.Tensor{g.Add(g.input("residual", 2.3, 0.1, width, streams, tokens), product)}, kept: []*tensor.Tensor{product}}
		}},
		{"Q8_0 shared expert", 3, folding(compiledFusionQ8Pair, compiledFusionQ8PairPeer, compiledFusionQ8GatedAdd), func(g *chainGraph, edge bool) fusionChain {
			// A gated shared expert (model buildSharedSwiGLU, expert_composition): its gate and up products in one launch,
			// its down product gated by its router and added; at the edge one token, over rows long enough that warps
			// share each.
			width, hidden, tokens := pick(edge, 512, 8192), pick(edge, 256, 160), pick(edge, q8InputSpanColumns, 1)
			x := g.input("x", 1, 0.02, width, tokens)
			gateProduct, upProduct := g.MulMat(g.weight("gate", dtype.Q8_0, width, hidden), x), g.MulMat(g.weight("up", dtype.Q8_0, width, hidden), x)
			down := g.MulMat(g.weight("down", dtype.Q8_0, hidden, width), g.SwiGLU(gateProduct, upProduct))
			shared := g.Multiply(down, g.Sigmoid(g.MulMat(g.Reshape(g.input("router", 0.05, 0.01, width), width, tensor.SingletonExtent), x)))
			routed := g.Add(g.input("routed", 1, -0.03, width, tokens), shared)
			return fusionChain{outputs: []*tensor.Tensor{routed}, kept: []*tensor.Tensor{gateProduct, upProduct, shared}}
		}},
		{"Q8_0 greedy selection", 2, folding(compiledFusionQ8ArgmaxPartials, compiledFusionQ8ArgmaxReduction), func(g *chainGraph, edge bool) fusionChain {
			// A Q8_0 output head's logits for a decode step's token, selected greedily; at the edge a vocabulary no warp
			// block divides.
			width, vocabulary := pick(edge, 256, 160), pick(edge, 4096, 1001)
			logits := g.MulMat(g.weight("head", dtype.Q8_0, width, vocabulary), g.input("hidden", 1, 0.02, width, tensor.SingletonExtent))
			return fusionChain{outputs: []*tensor.Tensor{g.TopK(logits, 1)}, kept: []*tensor.Tensor{logits}}
		}},
		{"BF16 greedy selection", 2, folding(compiledFusionBF16ArgmaxPartials, compiledFusionBF16ArgmaxReduction), func(g *chainGraph, edge bool) fusionChain {
			// A BF16 output head's (model decode_output), as the Q8_0 head's.
			width, vocabulary := pick(edge, 256, 98), pick(edge, 4096, 1001)
			logits := g.MulMat(g.weight("head", dtype.BF16, width, vocabulary), g.input("hidden", 1, 0.02, width, tensor.SingletonExtent))
			return fusionChain{outputs: []*tensor.Tensor{g.TopK(logits, 1)}, kept: []*tensor.Tensor{logits}}
		}},
		{"BF16 gate", 1, folding(compiledFusionBF16Gate), func(g *chainGraph, edge bool) fusionChain {
			// A decode step's gated feed-forward over BF16 weights (routedlm's decode graph).
			width, hidden := pick(edge, 256, 98), pick(edge, 512, 97)
			x := g.input("x", 1, 0.02, width, tensor.SingletonExtent)
			gateProduct := g.MulMat(g.weight("gate", dtype.BF16, width, hidden), x)
			return fusionChain{outputs: []*tensor.Tensor{g.SwiGLU(gateProduct, g.MulMat(g.weight("up", dtype.BF16, width, hidden), x))}, kept: []*tensor.Tensor{gateProduct}}
		}},
		{"BF16 projection add", 2, folding(compiledFusionBF16ProjAdd), func(g *chainGraph, edge bool) fusionChain {
			// A decode step's output projection added to its residual (routedlm), and a biased one (Krea's timestep).
			width, rows := pick(edge, 256, 98), pick(edge, 256, 97)
			x := g.input("x", 1, 0.02, width, tensor.SingletonExtent)
			projection := g.MulMat(g.weight("output", dtype.BF16, width, rows), x)
			biasedProjection := g.MulMat(g.weight("biased", dtype.BF16, width, rows), x)
			return fusionChain{outputs: []*tensor.Tensor{
				g.Add(g.input("residual", 2.3, 0.1, rows, tensor.SingletonExtent), projection), g.Add(biasedProjection, g.input("bias", 0.3, 0.05, rows)),
			}, kept: []*tensor.Tensor{projection, biasedProjection}}
		}},
		{"BF16 append", 1, folding(compiledFusionBF16Append), func(g *chainGraph, edge bool) fusionChain {
			// A decode step's value projection written into its cache's slot (routedlm): the last one at the edge.
			width, heads, capacity, inner := pick(edge, 128, 64), uint64(2), uint64(9), pick(edge, 256, 98)
			g.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: uint32(pick(edge, 3, capacity-1)), CapacityTokens: uint32(capacity)})
			projection := g.MulMat(g.weight("value", dtype.BF16, inner, width*heads), g.input("x", 1, 0.02, inner, tensor.SingletonExtent))
			appended := g.AppendCache(cached(g, "cache", width, heads, capacity), g.Reshape(projection, width, heads, tensor.SingletonExtent), 2)
			return fusionChain{outputs: []*tensor.Tensor{appended}, kept: []*tensor.Tensor{projection}}
		}},
		{"rope append", 2, folding(compiledFusionRopeAppend), func(g *chainGraph, edge bool) fusionChain {
			// A step's keys rotated into their cache's slots (model layer_tensor_plan), NeoX over the whole head as Qwen
			// rotates and normal over part of it as Llama's partial rotary does: the last slots at the edge.
			width, heads, tokens, capacity := pick(edge, 128, 64), uint64(2), pick(edge, 3, 1), uint64(9)
			active := pick(edge, 2, capacity-tokens)
			g.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: uint32(active), CapacityTokens: uint32(capacity)})
			positions := make([]uint32, tokens)
			for index := range positions {
				positions[index] = uint32(active) + uint32(index)
			}
			var outputs, kept []*tensor.Tensor
			for _, layout := range []struct {
				name    string
				layout  tensor.RoPELayout
				rotated uint64
			}{{"neox", tensor.RoPELayoutNeoX, width}, {"normal", tensor.RoPELayoutNormal, width / 4}} {
				rotated := g.RoPEWithOptions(g.input("keys_"+layout.name, 1.3, -0.1, width, heads, tokens), tensor.RoPEOptions{
					Layout: layout.layout, Positions: positions, RotaryDimensions: uint32(layout.rotated), FrequencyBase: 10000, FrequencyScale: 1,
				})
				outputs = append(outputs, g.AppendCache(cached(g, "cache_"+layout.name, width, heads, capacity), rotated, 2))
				kept = append(kept, rotated)
			}
			return fusionChain{outputs: outputs, kept: kept}
		}},
		{"BF16 linear", 2, folding(compiledFusionBF16Linear), func(g *chainGraph, edge bool) fusionChain {
			// The Transformers BF16 arithmetic's projection (model ArithmeticPolicy.project) of its rounded norm's output
			// over a prompt's tokens, and MOSS's biased linear of its rounded input (latentimage rotaryGraph.linear), which its
			// time embedding takes at one token; both at one token at the edge.
			width, rows, tokens := pick(edge, 256, 96), pick(edge, 256, 72), pick(edge, 64, 1)
			x := g.BF16Round(g.input("x", 1, 0.02, width, tokens))
			product := g.MulMatWithCompute(g.weight("linear", dtype.BF16, width, rows), x, tensor.MulMatComputeNativeTensorCore)
			biased := g.Add(g.MulMatWithCompute(g.weight("biased", dtype.BF16, width, rows), x, tensor.MulMatComputeBF16TensorCore), g.BF16RoundNative(g.input("bias", 0.3, 0.05, rows)))
			return fusionChain{outputs: []*tensor.Tensor{g.BF16RoundNative(product), g.BF16RoundNative(biased)}, kept: []*tensor.Tensor{product, biased}}
		}},
		{"BF16 attention", 4, folding(compiledFusionBF16Attention), func(g *chainGraph, edge bool) fusionChain {
			// Wan's and Krea's attention over heads rounded for storage, Krea's with its key bias, and MOSS's over natively
			// rounded heads, its scores kept in registers or comfy_kitchen's int8 attention; at the edge a token count no
			// tile divides and a head narrower than the kernel's.
			heads, tokens := uint64(2), pick(edge, 256, 77)
			scale := func(width uint64) float32 { return float32(1 / math.Sqrt(float64(width))) }
			var outputs, kept []*tensor.Tensor
			attend := func(name string, round func(*tensor.Tensor) *tensor.Tensor, width uint64, options tensor.AttentionOptions) {
				headsOf := func(part string, magnitude, bias float32) *tensor.Tensor {
					rounded := round(g.input(name+"_"+part, magnitude, bias, width, heads, tokens))
					kept = append(kept, rounded)
					return rounded
				}
				options.Scale = scale(width)
				outputs = append(outputs, g.AttentionWithOptions(headsOf("query", 0.7, -0.1), headsOf("key", 0.6, 0.05), headsOf("value", 1.1, 0.02), options))
			}
			attend("storage", g.BF16Round, 128, tensor.AttentionOptions{})
			attend("biased", g.BF16Round, 128, tensor.AttentionOptions{KeyBias: g.input("key_bias", 0.5, -0.2, tokens)})
			attend("registers", g.BF16RoundNative, pick(edge, 128, 64), tensor.AttentionOptions{ScoresInRegisters: true})
			attend("int8", g.BF16RoundNative, 128, tensor.AttentionOptions{Int8: true})
			return fusionChain{outputs: outputs, kept: kept}
		}},
	}
}
