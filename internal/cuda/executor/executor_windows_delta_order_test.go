//go:build windows

package executor

import (
	"context"
	"math/big"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestThePromptDeltaRecurrenceMatchesStratas holds the key-head recurrence a prompt takes to the bits of Strata's order
// (gdn_rec_kh_kernel): a token's key product summed a row group at a time, each group's rows in one fused multiply-add
// chain, the groups added in order, then each state row updated by its decay's fused multiply-add onto the product of
// its key and the token's delta. Under unit decays, which the fast exponential gives exactly, every step is
// reproducible on the host, so the state after a prompt of three value heads a key head over a partial staged block is
// compared bit for bit; decaying the state in a pass of its own, as the kernel had, adds each update in one rounding
// and leaves other bits.
func TestThePromptDeltaRecurrenceMatchesStratas(t *testing.T) {
	// Serial: device fixtures share the one card.
	const width, keyHeads, heads, tokens, groups = 128, 1, 3, 17, 4
	builder := tensor.NewBuilder()
	q := builder.Input("q", dtype.F32, tensor.MustShape(width, keyHeads, tokens, 1))
	k := builder.Input("k", dtype.F32, tensor.MustShape(width, keyHeads, tokens, 1))
	v := builder.Input("v", dtype.F32, tensor.MustShape(width, heads, tokens, 1))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(1, heads, tokens, 1))
	beta := builder.Input("beta", dtype.F32, tensor.MustShape(1, heads, tokens, 1))
	state := builder.Input("state", dtype.F32, tensor.MustShape(width, width, heads, 1))
	output := builder.GatedDeltaNet(q, k, v, gate, beta, state)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		q:     patternedValue(q.Shape, 7, 0.008, -0.015),
		k:     patternedValue(k.Shape, 11, 0.006, -0.01),
		v:     patternedValue(v.Shape, 13, 0.09, 0.03),
		gate:  {Shape: gate.Shape, Data: make([]float32, heads*tokens)},
		beta:  patternedValue(beta.Shape, 3, 0.03, 0.4),
		state: patternedValue(state.Shape, 17, 0.004, -0.007),
	}
	got, err := newFixtureExecutor(t).Execute(context.WithoutCancel(t.Context()), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	// fused is one rounding of a*b+c, exact before it.
	fused := func(a, b, c float32) float32 {
		product := new(big.Float).SetPrec(big.MaxPrec).Mul(big.NewFloat(float64(a)), big.NewFloat(float64(b)))
		sum, _ := product.Add(product, big.NewFloat(float64(c))).Float32()
		return sum
	}
	keys, values, betas := feeds[k].Data, feeds[v].Data, feeds[beta].Data
	want := append([]float32(nil), feeds[state].Data...)
	rows := width / groups
	for head := range heads {
		for column := range width {
			s := want[(head*width+column)*width:][:width]
			for token := range tokens {
				key := keys[token*width:][:width]
				dot := float32(0)
				for group := range groups {
					product := float32(0)
					for row := group * rows; row < (group+1)*rows; row++ {
						product = fused(s[row], key[row], product)
					}
					dot += product
				}
				delta := (values[(token*heads+head)*width+column] - dot) * betas[token*heads+head]
				for row := range width {
					s[row] = fused(1, s[row], float32(key[row]*delta))
				}
			}
		}
	}
	attention := width * heads * tokens
	compareBits(t, got[output].Data[attention:], want)
}
