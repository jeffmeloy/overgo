//go:build windows

package executor

import (
	"cmp"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// TestStrataNativeRouterMatchesStrata holds Strata's native router top-10 (strata_native_router.cuh, from
// native_router.cu) to its contract over 512 experts: a multi-token launch gives each token the experts and weights a
// one-token launch gives it, bit for bit; the experts are a float64 softmax's ten largest, in rank order; and the weights
// are their probabilities over the ten's sum, within float rounding. The shared expert's sigmoid scaling multiplies each
// token's row by the sigmoid of its gate.
func TestStrataNativeRouterMatchesStrata(t *testing.T) {
	const (
		tokens = 11
		// bound is the relative L1 distance the weights keep from the transcription.
		bound = 1e-5
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(71, 73))
	logits := make([]float32, tokens*strataRouterExperts)
	for index := range logits {
		logits[index] = float32(draws.NormFloat64()) * 3
	}
	logitBuffer := buffers.upload(t, float32Bytes(logits))
	ids, weights := buffers.zeros(t, 4*tokens*strataRouterPicks), buffers.zeros(t, 4*tokens*strataRouterPicks)
	buffers.run(t, func(state *device.State, functions functionSet) error {
		return launchStrataNativeRouteMulti(state, functions, logitBuffer, ids, weights, tokens)
	})
	multiIDs, multiWeights := buffers.read(t, ids, 4*tokens*strataRouterPicks), float32s(buffers.read(t, weights, 4*tokens*strataRouterPicks))
	for token := range tokens {
		oneIDs, oneWeights := buffers.zeros(t, 4*strataRouterPicks), buffers.zeros(t, 4*strataRouterPicks)
		at := logitBuffer + driver.DevicePtr(4*token*strataRouterExperts)
		buffers.run(t, func(state *device.State, functions functionSet) error {
			return launchStrataNativeRoute(state, functions, at, oneIDs, oneWeights)
		})
		gotIDs := buffers.read(t, oneIDs, 4*strataRouterPicks)
		gotWeights := float32s(buffers.read(t, oneWeights, 4*strataRouterPicks))
		if !slices.Equal(gotIDs, multiIDs[4*token*strataRouterPicks:4*(token+1)*strataRouterPicks]) ||
			!slices.Equal(gotWeights, multiWeights[token*strataRouterPicks:(token+1)*strataRouterPicks]) {
			t.Fatalf("token %d: the multi-token route differs from its one-token route", token)
		}
		row := logits[token*strataRouterExperts : (token+1)*strataRouterExperts]
		peak := float64(slices.Max(row))
		probabilities := make([]float64, strataRouterExperts)
		var total float64
		for expert, logit := range row {
			probabilities[expert] = math.Exp(float64(logit) - peak)
			total += probabilities[expert]
		}
		order := make([]int, strataRouterExperts)
		for index := range order {
			order[index] = index
		}
		slices.SortStableFunc(order, func(a, b int) int { return cmp.Compare(probabilities[b], probabilities[a]) })
		var picked float64
		for _, expert := range order[:strataRouterPicks] {
			picked += probabilities[expert] / total
		}
		want := make([]float64, strataRouterPicks)
		for rank, expert := range order[:strataRouterPicks] {
			if got := int32(uint32(gotIDs[4*rank]) | uint32(gotIDs[4*rank+1])<<8 | uint32(gotIDs[4*rank+2])<<16 | uint32(gotIDs[4*rank+3])<<24); got != int32(expert) {
				t.Fatalf("token %d: rank %d picked expert %d, the softmax's is %d", token, rank, got, expert)
			}
			want[rank] = probabilities[expert] / total / picked
		}
		if d := strataRelativeL1(float32To64(gotWeights), want); d > bound {
			t.Fatalf("token %d: the weights are %.3g from the renormalized probabilities", token, d)
		}
	}
	// The shared expert's sigmoid scaling, in float4s and not.
	for _, embedding := range []int32{2560, 2558} {
		rows := make([]float32, 3*embedding)
		for index := range rows {
			rows[index] = float32(draws.NormFloat64())
		}
		gates := []float32{-1.5, 0.25, 3}
		out, g := buffers.upload(t, float32Bytes(rows)), buffers.upload(t, float32Bytes(gates))
		buffers.run(t, func(state *device.State, functions functionSet) error {
			return launchStrataSigmoidScaleRows(state, functions, out, g, embedding, 3)
		})
		want := make([]float64, len(rows))
		for index, value := range rows {
			want[index] = float64(value) / (1 + math.Exp(-float64(gates[index/int(embedding)])))
		}
		if d := strataRelativeL1(float32To64(float32s(buffers.read(t, out, 4*len(rows)))), want); d > bound {
			t.Fatalf("the sigmoid scaling over %d values is %.3g from its formula", embedding, d)
		}
	}
}
