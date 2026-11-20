package latentimage

import (
	"fmt"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestAMOSSPassRoundsNoValueTwice holds moss-fusion/copies: a MOSS denoiser rounds no value that is already BF16, so
// no rounding in it reads a rounding, or an int8 product, whose epilogue rounds each entry to BF16
// (scale_int8_product_bf16_f32), except the rotated query and key an attention reads as roundings, which its rewrite
// (bf16-attention) takes into the attention's kernel. The rotary graph asked one rounding of each int8 product a block
// makes and one of each block's input, the last block's rounded sum: at the workflow's size, 401 bf16_round_f32
// launches took 19.1 ms of a depth control pass's 196 ms of kernels.
func TestAMOSSPassRoundsNoValueTwice(t *testing.T) {
	t.Parallel()
	spec := denoiserSpec()
	held := map[string]bool{}
	for layer := range spec.Layers {
		for _, projection := range jointProjections {
			held[fmt.Sprintf("layers.%d.%s.weight", layer, projection)] = true
		}
	}
	p := testcheck.Must(compileRotaryDenoiserProgram(spec, 1e-6, 2, 3, 4, nil, dtype.BF16, rotaryOptions{int8: held}))(t)
	seen, walk, attended := map[*tensor.Tensor]bool{}, []*tensor.Tensor{p.output}, map[*tensor.Tensor]bool{}
	for len(walk) > 0 {
		node := walk[len(walk)-1]
		walk = walk[:len(walk)-1]
		if !seen[node] {
			seen[node], walk = true, append(walk, node.Inputs...)
			if _, attention := node.Attrs.(tensor.AttentionAttributes); attention {
				for _, input := range node.Inputs {
					attended[input] = true
				}
			}
		}
	}
	product := func(node *tensor.Tensor) bool {
		return node.Op == tensor.OpMulMat && node.Inputs[0].Type == dtype.I8ConvRot
	}
	var products, twice int
	for node := range seen {
		if product(node) {
			products++
		}
		if node.Op == tensor.OpBF16Round && !attended[node] && (node.Inputs[0].Op == tensor.OpBF16Round || product(node.Inputs[0])) {
			twice++
		}
	}
	if products != spec.Layers*len(jointProjections) || twice != 0 {
		t.Fatalf("a denoiser of %d blocks, each projection in int8, made %d int8 products and rounded %d values already BF16 that no attention reads; want %d products and none rounded again",
			spec.Layers, products, twice, spec.Layers*len(jointProjections))
	}
}
