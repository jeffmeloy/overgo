// VJP of the routed mixture feed-forward. The top-k selection is fixed
// (piecewise-constant in the scores away from ties), so gradient flows to
// the magnitudes of the selected scores through the combine-weight Jacobian,
// the scoring activation, and the router matmul; expert and shared grads
// compose from the shared Linear/SiLU backwards. Gating intermediates stay
// in float64, matching the reference numerics.
package densecausal

import (
	"fmt"

	"overgo/internal/hostmath"
)

// moeGrads: parameter gradients of one routed layer's backward.
type moeGrads struct {
	dRouter  []float32
	dExperts []moeExpert
	dShared  *moeExpert
}

// expertBackward: VJP of expertForward for one row set; returns dx and
// accumulates weight grads into the provided expert-shaped slots.
func expertBackward(x []float32, expert moeExpert, dOut []float32, rows, hidden, inter int, into *moeExpert) []float32 {
	var dGateWeight, dUpWeight, dDownWeight []float32
	if into != nil {
		dGateWeight, dUpWeight, dDownWeight = into.gate, into.up, into.down
	}
	gate := make([]float32, rows*inter)
	up := make([]float32, rows*inter)
	hostmath.Linear(gate, x, expert.gate, rows, hidden, inter)
	hostmath.Linear(up, x, expert.up, rows, hidden, inter)
	activated := make([]float32, rows*inter)
	copy(activated, gate)
	hostmath.SiLUInPlace(activated)
	product := make([]float32, rows*inter)
	for i := range product {
		product[i] = activated[i] * up[i]
	}

	dProduct := make([]float32, rows*inter)
	hostmath.LinearBackward(dProduct, dDownWeight, nil, product, expert.down, dOut, rows, inter, hidden, false)
	dActivated := make([]float32, rows*inter)
	dUp := make([]float32, rows*inter)
	for i := range dProduct {
		dActivated[i] = dProduct[i] * up[i]
		dUp[i] = dProduct[i] * activated[i]
	}
	dGate := make([]float32, rows*inter)
	hostmath.SiLUBackward(dGate, gate, dActivated)
	dx := make([]float32, rows*hidden)
	hostmath.LinearBackward(dx, dGateWeight, nil, x, expert.gate, dGate, rows, hidden, inter, false)
	hostmath.LinearBackward(dx, dUpWeight, nil, x, expert.up, dUp, rows, hidden, inter, true)
	return dx
}

// accumulateMoEGrads adds one routed layer's mixture gradients into the
// named gradient map, so the flat pack sees every mixture tensor under its
// canonical name.
func accumulateMoEGrads(g Grads, names moeTensorNames, grads moeGrads) {
	if g == nil {
		return
	}
	addInPlace(hostmath.GradientSlot(g, names.router, len(grads.dRouter)), grads.dRouter)
	for e, expert := range grads.dExperts {
		addInPlace(hostmath.GradientSlot(g, names.experts[e].gate, len(expert.gate)), expert.gate)
		addInPlace(hostmath.GradientSlot(g, names.experts[e].up, len(expert.up)), expert.up)
		addInPlace(hostmath.GradientSlot(g, names.experts[e].down, len(expert.down)), expert.down)
	}
	if grads.dShared != nil && names.shared != nil {
		addInPlace(hostmath.GradientSlot(g, names.shared.gate, len(grads.dShared.gate)), grads.dShared.gate)
		addInPlace(hostmath.GradientSlot(g, names.shared.up, len(grads.dShared.up)), grads.dShared.up)
		addInPlace(hostmath.GradientSlot(g, names.shared.down, len(grads.dShared.down)), grads.dShared.down)
	}
}

// moeBackward: VJP of moeForward. dOut arrives at the mixture output; dx
// returns at the mixture input; parameter grads land in the returned slots.
func moeBackward(x []float32, w moeWeights, rows, hidden int, policy MoERouterPolicy, route moeRoute, dOut []float32, parameterGradients bool) ([]float32, moeGrads, error) {
	experts := len(w.experts)
	grads := moeGrads{}
	if len(route.indices) != rows*policy.TopK || len(route.weights) != rows*policy.TopK {
		return nil, grads, fmt.Errorf("densecausal: moe backward route len=%d/%d, want %d", len(route.indices), len(route.weights), rows*policy.TopK)
	}
	if parameterGradients {
		grads.dRouter = make([]float32, experts*hidden)
		grads.dExperts = make([]moeExpert, experts)
		for e := range grads.dExperts {
			grads.dExperts[e] = moeExpert{
				gate: make([]float32, policy.ExpertInter*hidden),
				up:   make([]float32, policy.ExpertInter*hidden),
				down: make([]float32, hidden*policy.ExpertInter),
			}
		}
		if w.shared != nil {
			grads.dShared = &moeExpert{
				gate: make([]float32, w.sharedInter*hidden),
				up:   make([]float32, w.sharedInter*hidden),
				down: make([]float32, hidden*w.sharedInter),
			}
		}
	}

	// Recompute activated router scores for the gating backward.
	_, scores, err := routeTopK(x, w.router, rows, hidden, experts, policy)
	if err != nil {
		return nil, grads, err
	}

	dx := make([]float32, rows*hidden)
	scaling := float64(policy.RoutedScaling)
	for r := 0; r < rows; r++ {
		xRow := x[r*hidden : (r+1)*hidden]
		dOutRow := dOut[r*hidden : (r+1)*hidden]
		dxRow := dx[r*hidden : (r+1)*hidden]
		sRow := scores[r*experts : (r+1)*experts]

		// Per selected expert: dw_k = <dOut, expertOut_k>, then expert VJP
		// with dy = w_k * dOut.
		dw := make([]float64, policy.TopK)
		for k := 0; k < policy.TopK; k++ {
			position := r*policy.TopK + k
			e := route.indices[position]
			expertOut := expertForward(xRow, w.experts[e], 1, hidden, policy.ExpertInter)
			var acc float64
			for i := range dOutRow {
				acc += float64(dOutRow[i]) * float64(expertOut[i])
			}
			dw[k] = acc
			scaled := make([]float32, hidden)
			weight := route.weights[position]
			for i := range dOutRow {
				scaled[i] = weight * dOutRow[i]
			}
			var destination *moeExpert
			if parameterGradients {
				destination = &grads.dExperts[e]
			}
			addInPlace(dxRow, expertBackward(xRow, w.experts[e], scaled, 1, hidden, policy.ExpertInter, destination))
		}
		if w.shared != nil {
			addInPlace(dxRow, expertBackward(xRow, *w.shared, dOutRow, 1, hidden, w.sharedInter, grads.dShared))
		}

		// Combine-weight Jacobian over the selected experts.
		ds := make([]float64, experts)
		if policy.NormalizeTopKProb && policy.TopK > 1 {
			var selectedSum, dotWG float64
			for k := 0; k < policy.TopK; k++ {
				selectedSum += float64(sRow[route.indices[r*policy.TopK+k]])
			}
			if selectedSum == 0 {
				return nil, grads, fmt.Errorf("densecausal: moe backward selected score sum is zero at row %d", r)
			}
			for k := 0; k < policy.TopK; k++ {
				dotWG += dw[k] * float64(sRow[route.indices[r*policy.TopK+k]]) / selectedSum
			}
			for k := 0; k < policy.TopK; k++ {
				e := route.indices[r*policy.TopK+k]
				ds[e] += (scaling / selectedSum) * (dw[k] - dotWG)
			}
		} else {
			for k := 0; k < policy.TopK; k++ {
				ds[route.indices[r*policy.TopK+k]] += scaling * dw[k]
			}
		}

		// Scoring-activation backward: softmax couples the row, sigmoid is
		// elementwise.
		dz := make([]float64, experts)
		switch policy.Scoring {
		case MoEScoringSigmoid:
			for e := 0; e < experts; e++ {
				se := float64(sRow[e])
				dz[e] = ds[e] * se * (1 - se)
			}
		case MoEScoringSoftmax:
			var dotSS float64
			for e := 0; e < experts; e++ {
				dotSS += ds[e] * float64(sRow[e])
			}
			for e := 0; e < experts; e++ {
				dz[e] = float64(sRow[e]) * (ds[e] - dotSS)
			}
		}

		// Router matmul backward: z[r,e] = <x[r], router[e]>.
		for e := 0; e < experts; e++ {
			if dz[e] == 0 {
				continue
			}
			dze := float32(dz[e])
			routerRow := w.router[e*hidden : (e+1)*hidden]
			var dRouterRow []float32
			if parameterGradients {
				dRouterRow = grads.dRouter[e*hidden : (e+1)*hidden]
			}
			for i := 0; i < hidden; i++ {
				if parameterGradients {
					dRouterRow[i] += dze * xRow[i]
				}
				dxRow[i] += dze * routerRow[i]
			}
		}
	}
	return dx, grads, nil
}
