package executor

import (
	"cmp"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
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
		// A group's activations lead the scratch, so a routed row of a block
		// storage starts 16-byte aligned for the block dots' vector reads.
		// Its route words and then its router logits follow.
		activations := blas.scores
		routeState := activations + driver.DevicePtr(uint64(group)*uint64(topK)*uint64(intermediate)*f32ScalarBytes)
		routerLogits := routeState +
			driver.DevicePtr(uint64(group)*(uint64(topK)*moeSlotRouteWords+moeTokenRouteWords)*f32ScalarBytes)
		if uint64(experts) > math.MaxUint32/f32ScalarBytes || uint64(group)*uint64(hidden) > math.MaxUint32 {
			return errors.New("grouped MoE launch size exceeds uint32")
		}
		logitBytes := experts * uint32(f32ScalarBytes)
		// Native execution splits each router logit's channels across a warp; strict keeps each logit's serial sum.
		routerKernel := functions[kernelMoeRouterLogitsF32]
		if attributes.Compute == tensor.MulMatComputeNativeTensorCore {
			routerKernel = functions[kernelMoeRouterDotF32]
		}
		// Indexed banks are tables of expert pointers; the route kernel marks
		// the routed experts for the residency to read between runs.
		indexed := uint32(0)
		var routed driver.DevicePtr
		var residency ExpertBanks
		banks := node.Inputs[moeFirstBankInput:wantInputs]
		if pointers.experts != nil && pointers.experts.IsExpertBank(banks[0].Name) {
			indexed, residency = 1, pointers.experts
			if routed, err = pointers.experts.ExpertRoutes(state, banks); err != nil {
				return err
			}
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
			if err := launchRouterLogits(state, functions, blas, routerKernel, attributes.Compute,
				groupRouterInput, router, routerBias, routerLogits, routerHidden, count, experts); err != nil {
				return err
			}
			if err := launchGridSharedABI(
				state, functions[kernelMoeRouteF32],
				driver.Dim3{X: count, Y: 1, Z: 1},
				driver.Dim3{X: moeThreads, Y: 1, Z: 1}, logitBytes,
				&routerLogits, &selectionBias, &groupSelected, &routeState, &routed, &count, &experts, &topK, &routing,
				&expertIndexDivisor,
			); err != nil {
				return err
			}
			if used, err := launchMoEMatrices(state, functions, blas, node,
				groupInput, routeState, activations, groupOutput, gate, up, down, gateBias, upBias, downBias, expertScale,
				count, residency); err != nil {
				return err
			} else if used {
				continue
			}
			hiddenCount := count * topK * intermediate
			downCount := count * hidden
			hiddenKernel, downKernel := functions[kernelMoeHiddenF32], functions[kernelMoeDownF32]
			hiddenGrid := count * topK * ((intermediate-1)/moeThreads + 1)
			sharedBytes := moeHiddenSharedBytes
			downThreads := downCount
			if attributes.Compute == tensor.MulMatComputeNativeTensorCore {
				// Native execution gives each routed row and each output channel a warp.
				hiddenThreads, hiddenErr := quantMulMatLaunchCount(dtype.BF16, intermediate, count*topK)
				nativeDown, downErr := quantMulMatLaunchCount(dtype.BF16, hidden, count)
				if hiddenErr == nil && downErr == nil {
					hiddenKernel, downKernel = functions[kernelMoeHiddenDotF32], functions[kernelMoeDownDotF32]
					hiddenGrid = (hiddenThreads-1)/moeThreads + 1
					sharedBytes = 0
					downThreads = nativeDown
				}
			}
			if err := launchGridSharedABI(
				state, hiddenKernel,
				driver.Dim3{X: hiddenGrid, Y: 1, Z: 1},
				driver.Dim3{X: moeThreads, Y: 1, Z: 1},
				sharedBytes,
				&groupInput, &routeState, &gate, &up, &gateBias, &upBias, &activations,
				&hidden, &topK, &intermediate, &routing, &gateStorage, &upStorage, &gated, &fusedGateUp,
				&activation, &expertIndexDivisor, &swigluClamp, &indexed, &hiddenCount,
			); err != nil {
				return err
			}
			if err := launch1DABI(
				state, downKernel, downThreads,
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

// launchRouterLogits writes a group's router logits, each token's experts
// together. A native group of a tile or more takes them as one F32 product of
// its router inputs with the router, then adds the router's bias: the kernel's
// warp a logit reads the router again for every token, 1.09 s of Flash-Next's
// 20,605-token prompt where the product takes milliseconds. A smaller group,
// and every strict one, keeps the kernel and its order of sums.
func launchRouterLogits(state *device.State, functions functionSet, blas *blasState, kernel boundKernel, compute tensor.MulMatCompute,
	input, router, bias, logits driver.DevicePtr, hidden, count, experts uint32) error {
	if compute != tensor.MulMatComputeNativeTensorCore || count < moeTileTokens {
		return launchGridSharedABI(
			state, kernel,
			driver.Dim3{X: (experts-1)/moeRouterExpertsPerBlock + 1, Y: count, Z: 1},
			driver.Dim3{X: moeThreads, Y: 1, Z: 1}, 0,
			&input, &router, &bias, &logits, &hidden, &count, &experts,
		)
	}
	if !traceExternalCall(state, traceTagSGEMM, uint64(router), uint64(input), uint64(logits), uint64(experts), uint64(count), uint64(hidden)) {
		if err := blas.library.SGEMM(
			blas.handle, cublas.OperationTranspose, cublas.OperationNone,
			int32(experts), int32(count), int32(hidden), gemmProductScale,
			router, int32(hidden), input, int32(hidden), gemmAccumulatorScale,
			logits, int32(experts),
		); err != nil {
			return err
		}
	}
	if bias == 0 {
		return nil
	}
	total, one := experts*count, uint32(1)
	return launch1DABI(
		state, functions[kernelBroadcastAddF32], total,
		&logits, &bias, &logits, &total,
		&experts, &count, &one, &one,
		&experts, &one, &one, &one,
		&experts, &count, &one,
	)
}

const (
	// moeFirstBankInput: a routed-expert node's inputs open with its
	// activations, its router input and its router; the expert banks follow.
	moeFirstBankInput = 3
	// moeThreads retains the generic 256-thread block for routing and hidden rows.
	moeThreads = uint32(256)
	// moeRouterExpertsPerBlock: the router logits give one warp of a block to each expert.
	moeRouterExpertsPerBlock = moeThreads / 32
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
	tokenLimit, scratchLimit := tokens, moeScratchLimitBytes
	if geometry, ok := nativeMoEMatrix(node); ok {
		tokenLimit = min(tokenLimit, geometry.tokenLimit())
		// A matrix group reads its experts once a tile, so a group takes as many tokens as the staging holds.
		scratchLimit = moeMatrixStagingBytes
	}
	activations := uint64(attributes.TopK) * intermediate
	tokenBytes := (uint64(attributes.TopK)*moeSlotRouteWords + moeTokenRouteWords + activations + uint64(attributes.Experts)) * f32ScalarBytes
	group := max(1, min(tokenLimit, scratchLimit/tokenBytes))
	// Groups of one size leave no short last group, which would fall below the matrix products' reuse.
	groups := (tokens-1)/group + 1
	return uint32((tokens-1)/groups + 1), tokenBytes, nil
}
