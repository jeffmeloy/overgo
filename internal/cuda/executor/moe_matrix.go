package executor

import (
	"context"
	"errors"
	"math"
	"slices"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// moeMatrixGeometry bounds decoded weights and padded expert rows in existing staging.
type moeMatrixGeometry struct {
	hidden, intermediate, experts       uint64
	weightBytes, rowBytes, encodedBytes uint64
	rowsPerToken, slotBytes             uint64
}

func nativeMoEMatrix(node *tensor.Tensor) (moeMatrixGeometry, bool) {
	a, ok := node.Attrs.(tensor.MoEAttributes)
	if !ok || a.Compute != tensor.MulMatComputeNativeTensorCore || !a.Gated || node.Shape.Dims[1] <= 1 {
		return moeMatrixGeometry{}, false
	}
	down := moeFirstBankInput + 1
	if !a.FusedGateUp {
		down++
	}
	if len(node.Inputs) <= down {
		return moeMatrixGeometry{}, false
	}
	bankShape := node.Inputs[down].Shape
	g := moeMatrixGeometry{hidden: node.Shape.Dims[0], intermediate: bankShape.Dims[0], experts: bankShape.Dims[bankShape.Rank-1]}
	if g.hidden == 0 || g.intermediate == 0 || g.experts == 0 || a.TopK == 0 || g.experts > math.MaxInt32 || g.hidden > math.MaxInt32 || g.intermediate > math.MaxInt32/tensor.PairedExtent || g.hidden > quantStagedChunkBytes/(tensor.PairedExtent*f32ScalarBytes)/g.intermediate {
		return moeMatrixGeometry{}, false
	}
	// Gate and up share one staged matrix and down reuses that allocation.
	g.weightBytes = tensor.PairedExtent * g.hidden * g.intermediate * f32ScalarBytes
	g.rowBytes = tensor.PairedExtent*(g.hidden+g.intermediate)*f32ScalarBytes + f32ScalarBytes
	g.slotBytes = uint64(a.TopK) * g.hidden * f32ScalarBytes
	// Native selection is distinct before dividing router IDs into bank IDs.
	g.rowsPerToken = min(uint64(a.TopK), uint64(max(a.ExpertIndexDivisor, 1)))
	if a.HasSelectedExperts {
		g.rowsPerToken = uint64(a.TopK)
	}
	if g.weightBytes > quantStagedChunkBytes-f32ScalarBytes || g.rowBytes*g.rowsPerToken+g.slotBytes > quantStagedChunkBytes-g.weightBytes-f32ScalarBytes {
		return moeMatrixGeometry{}, false
	}
	for _, bank := range node.Inputs[moeFirstBankInput : down+1] {
		bytes, err := bank.Shape.Bytes(bank.Type)
		if err != nil || bytes%g.experts != 0 {
			return moeMatrixGeometry{}, false
		}
		g.encodedBytes += bytes / g.experts
	}
	// Even maximal routing reuse must repay weight expansion and its F32 read.
	decoded := (tensor.PairedExtent + 1) * g.hidden * g.intermediate * f32ScalarBytes
	if g.encodedBytes == 0 || node.Shape.Dims[1] <= 1+tensor.PairedExtent*decoded/g.encodedBytes {
		return moeMatrixGeometry{}, false
	}
	return g, true
}

func (g moeMatrixGeometry) stagingBytes(group uint32, topK uint32) uint64 {
	slots := uint64(group) * g.slotBytes
	perExpert := g.weightBytes + uint64(group)*g.rowsPerToken*g.rowBytes + f32ScalarBytes
	experts := min(g.experts, uint64(group)*uint64(topK))
	if perExpert > (quantStagedChunkBytes-slots)/experts {
		return quantStagedChunkBytes
	}
	return perExpert*experts + slots
}

func launchMoEMatrices(ctx context.Context, state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor,
	input, routeState, output, gate, up, down, gateBias, upBias, downBias, expertScale driver.DevicePtr,
	count, indexed uint32) (bool, error) {
	g, eligible := nativeMoEMatrix(node)
	if !eligible {
		return false, nil
	}
	a := node.Attrs.(tensor.MoEAttributes)
	hidden, intermediate, topK := uint32(g.hidden), uint32(g.intermediate), a.TopK
	divisor := max(a.ExpertIndexDivisor, 1)
	routeWords := uint32(moeSlotRouteWords)*topK + uint32(moeTokenRouteWords)
	if indexed == 0 {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return false, err
		}
		blas.moeRoutes = slices.Grow(blas.moeRoutes[:0], int(count*routeWords))[:count*routeWords]
		if err := state.Driver.MemcpyDtoH(driver.Bytes(blas.moeRoutes), routeState); err != nil {
			return false, err
		}
	}
	blas.moeCounts = slices.Grow(blas.moeCounts[:0], int(g.experts))[:g.experts]
	clear(blas.moeCounts)
	for token := range count {
		for slot := uint32(0); slot < topK; slot++ {
			expert := blas.moeRoutes[token*routeWords+slot] / divisor
			if uint64(expert) >= g.experts {
				return false, errors.New("MoE route exceeds expert bank")
			}
			blas.moeCounts[expert]++
		}
	}
	blas.moeExperts = blas.moeExperts[:0]
	for expert, rows := range blas.moeCounts {
		if rows > 0 {
			blas.moeExperts = append(blas.moeExperts, uint32(expert))
		}
	}
	decoded := (tensor.PairedExtent + 1) * g.hidden * g.intermediate * f32ScalarBytes
	paired := uint64(count) * uint64(topK)
	// Include expansion, its read, input gathering and two output passes.
	staged := uint64(len(blas.moeExperts))*(g.encodedBytes+tensor.PairedExtent*decoded) + (tensor.PairedExtent+1)*paired*g.hidden*f32ScalarBytes
	if paired*g.encodedBytes <= staged {
		return false, nil
	}
	if blas.staging == 0 || blas.stagingBytes < g.stagingBytes(count, topK) {
		return false, errors.New("grouped MoE staging is unavailable")
	}
	slices.SortFunc(blas.moeExperts, func(left, right uint32) int { return int(blas.moeCounts[right]) - int(blas.moeCounts[left]) })
	blas.moeOffsets = slices.Grow(blas.moeOffsets[:0], int(g.experts)+1)[:g.experts+1]
	blas.moeOffsets[0] = 0
	for expert, rows := range blas.moeCounts {
		blas.moeOffsets[expert+1] = blas.moeOffsets[expert] + rows
	}
	copy(blas.moeCounts, blas.moeOffsets)
	baseSlots := slices.Grow(blas.moeSlots[:0], int(paired))[:paired]
	blas.moeSlots = baseSlots
	for token := range count {
		for slot := uint32(0); slot < topK; slot++ {
			expert := blas.moeRoutes[token*routeWords+slot] / divisor
			baseSlots[blas.moeCounts[expert]] = token*topK + slot
			blas.moeCounts[expert]++
		}
	}
	gateType := uint32(node.Inputs[moeFirstBankInput].Type)
	upAt := moeFirstBankInput
	if !a.FusedGateUp {
		upAt++
	}
	upType, downType := uint32(node.Inputs[upAt].Type), uint32(node.Inputs[upAt+1].Type)
	fused, activation, routing := kernelBool(a.FusedGateUp), uint32(a.Activation), uint32(a.Routing)
	clamp := a.SwiGLUClamp
	batchBytes := min(blas.stagingBytes, quantStagedChunkBytes) - uint64(count)*g.slotBytes
	slots := blas.staging + driver.DevicePtr(batchBytes)
	blas.stagedNode = nil
	for begin := 0; begin < len(blas.moeExperts); {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return false, err
			}
		}
		rows := blas.moeOffsets[blas.moeExperts[begin]+1] - blas.moeOffsets[blas.moeExperts[begin]]
		perExpert := g.weightBytes + uint64(rows)*g.rowBytes + f32ScalarBytes
		batch := uint32(min(uint64(len(blas.moeExperts)-begin), batchBytes/perExpert))
		if batch == 0 {
			return false, errors.New("grouped MoE staging cannot hold one expert")
		}
		tableWords := batch * (rows + 1)
		blas.moeTable = slices.Grow(blas.moeTable[:0], int(tableWords))[:tableWords]
		for index := range blas.moeTable {
			blas.moeTable[index] = math.MaxUint32
		}
		for index := uint32(0); index < batch; index++ {
			expert := blas.moeExperts[begin+int(index)]
			blas.moeTable[index] = expert
			copy(blas.moeTable[batch+index*rows:batch+(index+1)*rows], baseSlots[blas.moeOffsets[expert]:blas.moeOffsets[expert+1]])
		}
		table := blas.staging
		weights := table + driver.DevicePtr(uint64(tableWords)*f32ScalarBytes)
		x := weights + driver.DevicePtr(uint64(batch)*g.weightBytes)
		gu := x + driver.DevicePtr(uint64(batch)*uint64(rows)*g.hidden*f32ScalarBytes)
		y := gu + driver.DevicePtr(uint64(batch)*uint64(rows)*tensor.PairedExtent*g.intermediate*f32ScalarBytes)
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return false, err
		}
		if err := state.Driver.MemcpyHtoD(table, driver.Bytes(blas.moeTable)); err != nil {
			return false, err
		}
		xCount := batch * rows * hidden
		if err := launch1DABI(state, functions[kernelMoeGatherF32], xCount, &input, &table, &x, &hidden, &rows, &batch, &topK, &xCount); err != nil {
			return false, err
		}
		guRows := uint32(tensor.PairedExtent) * intermediate
		weightCount := batch * guRows * hidden
		if err := launch1DABI(state, functions[kernelMoeStageF32], weightCount, &gate, &up, &table, &weights, &hidden, &intermediate, &gateType, &upType, &fused, &indexed, &weightCount); err != nil {
			return false, err
		}
		if err := moeMatrixProduct(state, blas, weights, x, gu, guRows, rows, hidden, hidden, guRows, batch); err != nil {
			return false, err
		}
		activationCount := batch * rows * intermediate
		if err := launch1DABI(state, functions[kernelMoeActivateF32], activationCount, &gu, &table, &gateBias, &upBias, &intermediate, &rows, &activation, &routing, &clamp, &activationCount); err != nil {
			return false, err
		}
		weightCount = batch * hidden * intermediate
		if err := launch1DABI(state, functions[kernelMoeStageDownF32], weightCount, &down, &table, &weights, &hidden, &intermediate, &downType, &indexed, &weightCount); err != nil {
			return false, err
		}
		if err := moeMatrixProduct(state, blas, weights, gu, y, hidden, rows, intermediate, guRows, hidden, batch); err != nil {
			return false, err
		}
		if err := launch1DABI(state, functions[kernelMoeScatterF32], xCount, &y, &table, &slots, &downBias, &hidden, &rows, &batch, &xCount); err != nil {
			return false, err
		}
		begin += int(batch)
	}
	normalize, scale := kernelBool(a.NormalizeTopKProb), a.Scale
	outputCount := count * hidden
	if err := launch1DABI(state, functions[kernelMoeCombineF32], outputCount, &routeState, &slots, &expertScale, &output, &hidden, &topK, &normalize, &scale, &divisor, &outputCount); err != nil {
		return false, err
	}
	blas.stagedNode = nil
	return true, nil
}

func moeMatrixProduct(state *device.State, blas *blasState, weights, input, output driver.DevicePtr, rows, columns, inner, inputLeading, outputLeading, batch uint32) error {
	if traceExternalCall(state, traceTagSGEMMStrided, uint64(weights), uint64(input), uint64(output), uint64(rows), uint64(columns), uint64(inner), uint64(batch)) {
		return nil
	}
	return blas.library.SGEMMStridedBatched(blas.handle, cublas.OperationTranspose, cublas.OperationNone,
		int32(rows), int32(columns), int32(inner), gemmProductScale,
		weights, int32(inner), int64(rows)*int64(inner), input, int32(inputLeading), int64(inputLeading)*int64(columns),
		gemmAccumulatorScale, output, int32(outputLeading), int64(outputLeading)*int64(columns), int32(batch))
}
