package densecausal

import (
	"math"
	"slices"
	"testing"
)

// tinyMixture builds a deterministic small mixture exercising every gradient
// family: router, routed experts, shared expert.
func tinyMixture(seed uint32, hidden, inter, experts int, shared bool) moeWeights {
	next := func() float32 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return (float32(seed%2000)/1000 - 1) * 0.3
	}
	fill := func(n int) []float32 {
		values := make([]float32, n)
		for i := range values {
			values[i] = next()
		}
		return values
	}
	w := moeWeights{router: fill(experts * hidden), experts: make([]moeExpert, experts)}
	for e := range w.experts {
		w.experts[e] = moeExpert{gate: fill(inter * hidden), up: fill(inter * hidden), down: fill(hidden * inter)}
	}
	if shared {
		w.shared = &moeExpert{gate: fill(inter * hidden), up: fill(inter * hidden), down: fill(hidden * inter)}
		w.sharedInter = inter
	}
	return w
}

// mixtureLoss: scalar objective sum(out * probe) for finite differencing.
func mixtureLoss(t *testing.T, x []float32, w moeWeights, rows, hidden int, policy MoERouterPolicy, probe []float32) float64 {
	t.Helper()
	out, _, err := moeForward(x, w, rows, hidden, policy)
	if err != nil {
		t.Fatal(err)
	}
	var total float64
	for i := range out {
		total += float64(out[i]) * float64(probe[i])
	}
	return total
}

func TestMoEBackwardMatchesFiniteDifference(t *testing.T) {
	const (
		rows, hidden, inter, experts = 3, 6, 4, 5
		epsilon                      = 1e-3
		limit                        = 2e-2
	)
	for _, tc := range []struct {
		name   string
		policy MoERouterPolicy
		shared bool
	}{
		{"softmax-normalized-shared", MoERouterPolicy{TopK: 2, Scoring: MoEScoringSoftmax, NormalizeTopKProb: true, RoutedScaling: 1.5, ExpertInter: inter}, true},
		{"sigmoid-plain", MoERouterPolicy{TopK: 2, Scoring: MoEScoringSigmoid, RoutedScaling: 1, ExpertInter: inter}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := tinyMixture(2463534242, hidden, inter, experts, tc.shared)
			x := make([]float32, rows*hidden)
			probe := make([]float32, rows*hidden)
			seed := uint32(88172645)
			for i := range x {
				seed ^= seed << 13
				seed ^= seed >> 17
				seed ^= seed << 5
				x[i] = (float32(seed%2000)/1000 - 1) * 0.5
				probe[i] = (float32(seed%701)/350 - 1)
			}
			_, route, err := moeForward(x, w, rows, hidden, tc.policy)
			if err != nil {
				t.Fatal(err)
			}
			dx, grads, err := moeBackward(x, w, rows, hidden, tc.policy, route, probe, true)
			if err != nil {
				t.Fatal(err)
			}
			dxOnly, omitted, err := moeBackward(x, w, rows, hidden, tc.policy, route, probe, false)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(dxOnly, dx) || omitted.dRouter != nil || omitted.dExperts != nil || omitted.dShared != nil {
				t.Fatalf("activation-only VJP differs or retained parameter gradients")
			}

			check := func(label string, values, analytic []float32, count int) {
				t.Helper()
				checked := 0
				for i := 0; i < len(values) && checked < count; i += 1 + len(values)/count {
					original := values[i]
					values[i] = original + epsilon
					plus := mixtureLoss(t, x, w, rows, hidden, tc.policy, probe)
					values[i] = original - epsilon
					minus := mixtureLoss(t, x, w, rows, hidden, tc.policy, probe)
					values[i] = original
					numeric := (plus - minus) / (2 * epsilon)
					got := float64(analytic[i])
					scale := max(1, max(math.Abs(numeric), math.Abs(got)))
					if math.Abs(numeric-got)/scale > limit {
						t.Fatalf("%s[%d]: analytic=%.6g numeric=%.6g", label, i, got, numeric)
					}
					checked++
				}
				if checked == 0 {
					t.Fatalf("%s: nothing checked", label)
				}
			}
			check("dx", x, dx, 8)
			check("router", w.router, grads.dRouter, 8)
			selected := route.indices[0]
			check("expert.gate", w.experts[selected].gate, grads.dExperts[selected].gate, 6)
			check("expert.down", w.experts[selected].down, grads.dExperts[selected].down, 6)
			if tc.shared {
				check("shared.up", w.shared.up, grads.dShared.up, 6)
			}
		})
	}
}
