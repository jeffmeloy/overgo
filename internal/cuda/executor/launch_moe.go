package executor

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func launchMoE(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
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
		next := 3
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
		if upNode.Type != downNode.Type || attributes.Gated && !attributes.FusedGateUp && node.Inputs[3].Type != upNode.Type {
			return errors.New("MoE expert storage types differ")
		}
		var expertStorage uint32
		if upNode.Type == dtype.F32 {
			expertStorage = uint32(dtype.F32)
		} else if nativeQuantizedType(upNode.Type) {
			expertStorage = uint32(upNode.Type)
		} else {
			return fmt.Errorf("MoE expert storage type %s is unsupported", upNode.Type)
		}
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
		if expertIndexDivisor == 0 {
			expertIndexDivisor = 1
		}
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
		const (
			groupedMoEThreads          = uint32(256)
			groupedMoESharedScalars    = uint64(1)
			groupedMoESharedEntryBytes = uint64(4)
		)
		sharedEntries := uint64(topK)*2 + uint64(groupedMoEThreads) + groupedMoESharedScalars
		if sharedEntries > math.MaxUint32/groupedMoESharedEntryBytes {
			return errors.New("grouped MoE shared-memory size exceeds uint32")
		}
		sharedBytes := uint32(sharedEntries * groupedMoESharedEntryBytes)
		return launchGridSharedABI(
			state, functions[kernelMoeGroupedF32],
			driver.Dim3{X: tokens, Y: 1, Z: 1},
			driver.Dim3{X: groupedMoEThreads, Y: 1, Z: 1}, sharedBytes,
			&input, &routerInput, &router, &gate, &up, &down, &selectionBias, &expertScale,
			&routerBias, &gateBias, &upBias, &downBias, &selectedExperts, &output,
			&hidden, &routerHidden, &tokens, &experts, &topK, &intermediate, &normalize,
			&routing, &scale, &expertStorage, &gated, &fusedGateUp, &activation,
			&expertIndexDivisor, &swigluClamp,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
