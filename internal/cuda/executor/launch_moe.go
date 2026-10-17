package executor

import (
	"cmp"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func launchMoE(
	state *device.State,
	functions functionSet,
	blas *blasState,
	_ *q8InputState,
	node *tensor.Tensor,
	_ tensor.Attributes,
	pointers launchPointerFrame,
) error {
	output := pointers.output()
	switch node.Op {
	case tensor.OpMoE:
		attributes, ok := node.Attrs.(tensor.MoEAttributes)
		wantInputs := 5
		if attributes.Gated && !attributes.FusedGateUp {
			wantInputs = 6
		}
		expectedInputs := wantInputs
		if attributes.HasSelectionBias {
			expectedInputs++
		}
		if attributes.HasExpertScale {
			expectedInputs++
		}
		if attributes.HasRouterBias {
			expectedInputs++
		}
		if attributes.HasExpertBiases {
			expectedInputs += 3
		}
		if attributes.HasSelectedExperts {
			expectedInputs++
		}
		if !ok || len(node.Inputs) != expectedInputs ||
			attributes.HasRouterBias != attributes.HasExpertBiases ||
			(attributes.Activation == tensor.MoEActivationSwiGLUOAI) != attributes.HasExpertBiases ||
			(attributes.Routing == tensor.MoERoutingSelectedSoftmax) != (attributes.Activation == tensor.MoEActivationSwiGLUOAI) ||
			(attributes.Routing != tensor.MoERoutingSoftmax && attributes.Routing != tensor.MoERoutingSigmoid &&
				attributes.Routing != tensor.MoERoutingSelectedSoftmax && attributes.Routing != tensor.MoERoutingSqrtSoftplus) ||
			(attributes.Activation != tensor.MoEActivationSiLU && attributes.Activation != tensor.MoEActivationReLU &&
				attributes.Activation != tensor.MoEActivationGELU && attributes.Activation != tensor.MoEActivationSwiGLUOAI &&
				attributes.Activation != tensor.MoEActivationReLUSquared) ||
			attributes.SwiGLUClamp < 0 ||
			math.IsNaN(float64(attributes.SwiGLUClamp)) || math.IsInf(float64(attributes.SwiGLUClamp), 0) ||
			attributes.SwiGLUClamp > 0 && (attributes.Activation != tensor.MoEActivationSiLU || !attributes.Gated) {
			return errors.New("invalid MoE attributes")
		}
		hidden, err := uint32Checked(node.Shape.Dims[0], "MoE hidden width")
		if err != nil {
			return err
		}
		routerHidden, err := uint32Checked(node.Inputs[1].Shape.Dims[0], "MoE router hidden width")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[1], "MoE token count")
		if err != nil {
			return err
		}
		next := moeFirstBankInput
		var gate driver.DevicePtr
		if attributes.Gated && !attributes.FusedGateUp {
			gate = pointers.input(next)
			next++
		}
		upNode, downNode := node.Inputs[next], node.Inputs[next+1]
		intermediate, err := uint32Checked(upNode.Shape.Dims[1], "MoE intermediate width")
		if err != nil {
			return err
		}
		input := pointers.input(0)
		routerInput := pointers.input(1)
		router := pointers.input(2)
		up := pointers.input(next)
		down := pointers.input(next + 1)
		if attributes.FusedGateUp {
			gate = up
			intermediate /= 2
		}
		for _, bank := range node.Inputs[moeFirstBankInput:wantInputs] {
			if bank.Type != dtype.F32 && bank.Type != dtype.BF16 && !nativeQuantizedType(bank.Type) {
				return fmt.Errorf("MoE expert storage type %s is unsupported", bank.Type)
			}
		}
		gateStorage := uint32(node.Inputs[moeFirstBankInput].Type)
		upStorage, downStorage := uint32(upNode.Type), uint32(downNode.Type)
		var selectionBias driver.DevicePtr
		optionalIndex := wantInputs
		if attributes.HasSelectionBias {
			selectionBias = pointers.input(optionalIndex)
			optionalIndex++
		}
		var expertScale driver.DevicePtr
		if attributes.HasExpertScale {
			expertScale = pointers.input(optionalIndex)
			optionalIndex++
		}
		var routerBias, gateBias, upBias, downBias driver.DevicePtr
		if attributes.HasRouterBias {
			routerBias = pointers.input(optionalIndex)
			optionalIndex++
		}
		if attributes.HasExpertBiases {
			gateBias = pointers.input(optionalIndex)
			upBias = pointers.input(optionalIndex + 1)
			downBias = pointers.input(optionalIndex + 2)
			optionalIndex += 3
		}
		var selectedExperts driver.DevicePtr
		if attributes.HasSelectedExperts {
			selectedExperts = pointers.input(optionalIndex)
		}
		experts := attributes.Experts
		expertIndexDivisor := attributes.ExpertIndexDivisor
		expertIndexDivisor = cmp.Or(expertIndexDivisor, 1)
		topK := attributes.TopK
		if experts%expertIndexDivisor != 0 || upNode.Shape.Dims[2] != uint64(experts/expertIndexDivisor) ||
			downNode.Shape.Dims[2] != upNode.Shape.Dims[2] || uint64(topK) > upNode.Shape.Dims[2] {
			return errors.New("invalid grouped MoE dimensions")
		}
		normalize := kernelBool(attributes.NormalizeTopKProb)
		scale := attributes.Scale
		routing := uint32(attributes.Routing)
		activation := uint32(attributes.Activation)
		swigluClamp := attributes.SwiGLUClamp
		gated := kernelBool(attributes.Gated)
		fusedGateUp := kernelBool(attributes.FusedGateUp)
		group, tokenBytes, err := moeScratchGroup(node)
		if err != nil {
			return err
		}
		if blas == nil || blas.scores == 0 || uint64(group)*tokenBytes > blas.scoreBytes {
			return errors.New("MoE scratch is unavailable")
		}
		// A group's route words lead the scratch; its activations follow.
		routeState := blas.scores
		activations := routeState +
			driver.DevicePtr(uint64(group)*(uint64(topK)*moeSlotRouteWords+moeTokenRouteWords)*f32ScalarBytes)
		if uint64(experts) > math.MaxUint32/f32ScalarBytes || uint64(group)*uint64(hidden) > math.MaxUint32 {
			return errors.New("grouped MoE launch size exceeds uint32")
		}
		logitBytes := experts * uint32(f32ScalarBytes)
		indexed := uint32(0)
		banks := node.Inputs[moeFirstBankInput:wantInputs]
		if pointers.experts != nil && pointers.experts.IsExpertBank(banks[0].Name) {
			indexed = 1
		}
		for start := uint32(0); start < tokens; start += group {
			if pointers.ctx != nil {
				if err := pointers.ctx.Err(); err != nil {
					return err
				}
			}
			count := min(group, tokens-start)
			tokenOffset := uint64(start) * f32ScalarBytes
			groupInput := input + driver.DevicePtr(tokenOffset*uint64(hidden))
			groupRouterInput := routerInput + driver.DevicePtr(tokenOffset*uint64(routerHidden))
			groupOutput := output + driver.DevicePtr(tokenOffset*uint64(hidden))
			groupSelected := selectedExperts
			if attributes.HasSelectedExperts {
				groupSelected += driver.DevicePtr(tokenOffset * uint64(topK))
			}
			if err := launchGridSharedABI(
				state, functions[kernelMoeRouteF32],
				driver.Dim3{X: count, Y: 1, Z: 1},
				driver.Dim3{X: moeThreads, Y: 1, Z: 1}, logitBytes,
				&groupRouterInput, &router, &selectionBias, &routerBias, &groupSelected, &routeState,
				&routerHidden, &count, &experts, &topK, &routing,
			); err != nil {
				return err
			}
			if indexed != 0 {
				if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
					return err
				}
				routeWords := uint32(moeSlotRouteWords)*topK + uint32(moeTokenRouteWords)
				words := int(count) * int(routeWords)
				if cap(blas.moeRoutes) < words {
					blas.moeRoutes = make([]uint32, words)
				}
				routes := blas.moeRoutes[:words]
				if err := state.Driver.MemcpyDtoH(driver.Bytes(routes), routeState); err != nil {
					return err
				}
				blas.moeExperts = blas.moeExperts[:0]
				for token := uint32(0); token < count; token++ {
					for slot := uint32(0); slot < topK; slot++ {
						blas.moeExperts = append(blas.moeExperts, routes[token*routeWords+slot]/expertIndexDivisor)
					}
				}
				tables, err := pointers.experts.ResolveExpertBanks(pointers.ctx, state, banks, blas.moeExperts)
				if err != nil {
					return err
				}
				if len(tables) != len(banks) {
					return errors.New("expert resolver returned an incomplete bank group")
				}
				up, down = tables[next-moeFirstBankInput], tables[next-moeFirstBankInput+1]
				if attributes.Gated {
					gate = tables[0]
				}
			}
			hiddenCount := count * topK * intermediate
			if err := launchGridSharedABI(
				state, functions[kernelMoeHiddenF32],
				driver.Dim3{X: count * topK * ((intermediate-1)/moeThreads + 1), Y: 1, Z: 1},
				driver.Dim3{X: moeThreads, Y: 1, Z: 1},
				moeHiddenSharedBytes,
				&groupInput, &routeState, &gate, &up, &gateBias, &upBias, &activations,
				&hidden, &topK, &intermediate, &routing, &gateStorage, &upStorage, &gated, &fusedGateUp,
				&activation, &expertIndexDivisor, &swigluClamp, &indexed, &hiddenCount,
			); err != nil {
				return err
			}
			downCount := count * hidden
			partialLanes := uint32(1)
			partials := topK * ((intermediate-1)/moeDownTile + 1)
			for partialLanes < partials && partialLanes < uint32(kernel.WarpThreads()) {
				partialLanes <<= 1
			}
			rows := moeThreads / partialLanes
			if err := launchGridSharedABI(
				state, functions[kernelMoeDownF32],
				driver.Dim3{X: (downCount-1)/rows + 1, Y: 1, Z: 1},
				driver.Dim3{X: moeThreads, Y: 1, Z: 1}, moeThreads*uint32(f32ScalarBytes),
				&routeState, &activations, &down, &expertScale, &downBias, &groupOutput,
				&hidden, &topK, &intermediate, &normalize, &scale, &downStorage,
				&expertIndexDivisor, &indexed, &downCount,
			); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}

const (
	// moeFirstBankInput: a routed-expert node's inputs open with its
	// activations, its router input and its router; the expert banks follow.
	moeFirstBankInput = 3
	// moeThreads retains the generic 256-thread block for routing and hidden rows.
	moeThreads = uint32(256)
	// moeDownTile preserves the CUDA down projection's arithmetic boundary.
	// Changing it changes the ordered partial sums and requires a numerical contract.
	moeDownTile = uint32(256)
	// moeHiddenSharedBytes holds two banks of sector-wide F32 rows with one padding word.
	// Keep this ABI equal to the CUDA tile and reconsider it with measured occupancy.
	moeHiddenSharedBytes = 2 * moeThreads * (32/4 + 1) * 4
	// A token's route words: each routed slot's expert and route weight, then
	// the route weights' sum.
	moeSlotRouteWords  = uint64(2)
	moeTokenRouteWords = uint64(1)
	// moeScratchLimitBytes bounds the scratch one launch group holds; a
	// longer prompt runs as several groups.
	moeScratchLimitBytes = uint64(32 << 20)
)

// moeScratchGroup: the tokens one routed-expert launch group covers and the
// scratch bytes each of them holds -- its route words and one activation per
// routed intermediate row.
func moeScratchGroup(node *tensor.Tensor) (uint32, uint64, error) {
	attributes, ok := node.Attrs.(tensor.MoEAttributes)
	downInput := moeFirstBankInput + 1
	if attributes.Gated && !attributes.FusedGateUp {
		downInput++
	}
	if !ok || len(node.Inputs) <= downInput {
		return 0, 0, errors.New("invalid MoE attributes")
	}
	tokens, intermediate := node.Shape.Dims[1], node.Inputs[downInput].Shape.Dims[0]
	if tokens == 0 || tokens > math.MaxUint32 || attributes.TopK == 0 ||
		intermediate == 0 || intermediate > math.MaxUint32/uint64(attributes.TopK) {
		return 0, 0, errors.New("invalid grouped MoE dimensions")
	}
	activations := uint64(attributes.TopK) * intermediate
	tokenBytes := (uint64(attributes.TopK)*moeSlotRouteWords + moeTokenRouteWords + activations) * f32ScalarBytes
	return uint32(min(tokens, max(1, moeScratchLimitBytes/tokenBytes))), tokenBytes, nil
}
