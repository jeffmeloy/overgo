package executor

import (
	"errors"
	"math"
	"slices"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

const (
	// moeTileTokens, moeTileRows and moeTileThreads mirror the grouped expert
	// kernels' MOE_TILE_TOKENS, MOE_TILE_ROWS and MOE_TILE_THREADS: a block
	// multiplies one tile of up to 32 routed rows of an expert by 64 of its
	// weight rows with four warps.
	moeTileTokens  = uint32(32)
	moeTileRows    = uint32(64)
	moeTileThreads = uint32(128)
	// moeTileWords: a tile's plan words, its expert, first position and rows.
	moeTileWords = uint32(3)
	// moePlanWords bounds a routed row's plan words: its position and at
	// most one tile it opens.
	moePlanWords = uint64(moeTileWords + 1)
	// moeSlotAlignment starts the routed outputs, the copied experts' tables
	// and each copied expert on a CUDA allocation's alignment.
	moeSlotAlignment = uint64(256)
	// moeBanks bounds a routed node's banks: gate, up and down.
	moeBanks = uint64(3)
	// moeCopyBytes: the staging a batch of routed experts copied from host
	// memory takes, 27 of Flash-Next's 2.33 MB experts. The copy runs at the
	// link's rate, where the kernels' own reads of host memory ran at 9 to
	// 33 GB/s, and once a group, where they ran once a tile.
	moeCopyBytes = uint64(64 << 20)
)

// moeMatrixGeometry sizes a prompt group's grouped expert products in the
// staging: the plan, each routed slot's output row, the copied experts'
// tables and the copied experts.
type moeMatrixGeometry struct {
	hidden, intermediate, experts uint64
	// tokenBytes: the staging one token takes, its slots' outputs and plan words.
	tokenBytes uint64
}

// nativeMoEMatrix reports the geometry of a routed node whose prompt groups
// take the grouped products: native compute over gated experts, in a prompt
// that fills a tile.
func nativeMoEMatrix(node *tensor.Tensor) (moeMatrixGeometry, bool) {
	a, ok := node.Attrs.(tensor.MoEAttributes)
	if !ok || a.Compute != tensor.MulMatComputeNativeTensorCore || !a.Gated || node.Shape.Dims[1] < uint64(moeTileTokens) {
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
	if g.hidden == 0 || g.intermediate == 0 || g.experts == 0 || a.TopK == 0 || g.experts > math.MaxUint32 ||
		g.hidden > math.MaxUint32 || g.intermediate > math.MaxUint32 {
		return moeMatrixGeometry{}, false
	}
	g.tokenBytes = uint64(a.TopK) * (g.hidden + moePlanWords) * f32ScalarBytes
	// A tile's outputs take the staging beside the copies, and its activations the scratch.
	activationBytes := uint64(a.TopK) * g.intermediate * f32ScalarBytes
	return g, g.copyBytes() < quantStagedChunkBytes &&
		max(g.tokenBytes, activationBytes)*uint64(moeTileTokens) <= quantStagedChunkBytes-g.copyBytes()
}

// moeAlign rounds bytes up to moeSlotAlignment.
func moeAlign(bytes uint64) uint64 {
	return (bytes + moeSlotAlignment - 1) / moeSlotAlignment * moeSlotAlignment
}

// copyBytes is the staging the copied experts and their tables take, with the plan's and the outputs' alignment.
func (g moeMatrixGeometry) copyBytes() uint64 {
	return moeAlign(moeBanks*g.experts*uint64(unsafe.Sizeof(driver.DevicePtr(0)))) + moeCopyBytes + moeBanks*moeSlotAlignment
}

// tokenLimit is the tokens whose plan and outputs the staging holds beside the copies.
func (g moeMatrixGeometry) tokenLimit() uint64 {
	return (quantStagedChunkBytes - g.copyBytes()) / g.tokenBytes
}

// planBytes is the staging a group's plan takes, aligned for its outputs.
func (g moeMatrixGeometry) planBytes(group uint32, topK uint32) uint64 {
	return moeAlign(uint64(group) * uint64(topK) * moePlanWords * f32ScalarBytes)
}

// copiesAt is the staging offset of the copied experts' tables, after the plan and the outputs.
func (g moeMatrixGeometry) copiesAt(group uint32, topK uint32) uint64 {
	return g.planBytes(group, topK) + moeAlign(uint64(group)*uint64(topK)*g.hidden*f32ScalarBytes)
}

func (g moeMatrixGeometry) stagingBytes(group uint32, topK uint32) uint64 {
	return g.copiesAt(group, topK) + moeAlign(moeBanks*g.experts*uint64(unsafe.Sizeof(driver.DevicePtr(0)))) + moeCopyBytes
}

// launchMoEMatrices computes a prompt group's routed experts as grouped
// products over the encoded experts.
//   - the group's routes come to the host, and its routed rows, ordered by expert, form tiles of up to moeTileTokens rows
//   - a tile reads each weight of its expert once for all its rows
//   - experts on the card go first; experts in host memory follow in batches copied into the staging, each once a group
//   - activations takes each routed row's activations in expert order, and the staging the plan and each slot's output
func launchMoEMatrices(state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor,
	input, routeState, activations, output, gate, up, down, gateBias, upBias, downBias, expertScale driver.DevicePtr,
	count uint32, residency ExpertBanks) (bool, error) {
	g, eligible := nativeMoEMatrix(node)
	if !eligible || count < moeTileTokens {
		return false, nil
	}
	a := node.Attrs.(tensor.MoEAttributes)
	upAt := moeFirstBankInput
	tables := []driver.DevicePtr{up, down}
	if !a.FusedGateUp {
		upAt++
		tables = []driver.DevicePtr{gate, up, down}
	}
	banks := node.Inputs[moeFirstBankInput : moeFirstBankInput+len(tables)]
	hidden, intermediate, topK, experts := uint32(g.hidden), uint32(g.intermediate), a.TopK, uint32(g.experts)
	divisor := max(a.ExpertIndexDivisor, 1)
	routeWords := uint32(moeSlotRouteWords)*topK + uint32(moeTokenRouteWords)
	if blas.staging == 0 || blas.stagingBytes < g.stagingBytes(count, topK) {
		return false, errors.New("grouped MoE staging is unavailable")
	}
	// A prompt group's tiles are planned on the host from its routes.
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		return false, err
	}
	blas.moeRoutes = slices.Grow(blas.moeRoutes[:0], int(count*routeWords))[:count*routeWords]
	if err := state.Driver.MemcpyDtoH(driver.Bytes(blas.moeRoutes), routeState); err != nil {
		return false, err
	}
	blas.moeOffsets = slices.Grow(blas.moeOffsets[:0], int(experts)+1)[:experts+1]
	clear(blas.moeOffsets)
	for token := range count {
		for slot := range topK {
			expert := blas.moeRoutes[token*routeWords+slot] / divisor
			if expert >= experts {
				return false, errors.New("MoE route exceeds expert bank")
			}
			blas.moeOffsets[expert+1]++
		}
	}
	for expert := range experts {
		blas.moeOffsets[expert+1] += blas.moeOffsets[expert]
	}
	// Each bank's expert addresses, and the pinned bytes of those in host memory.
	blas.moeSources = slices.Grow(blas.moeSources[:0], len(banks)*int(experts))[:len(banks)*int(experts)]
	blas.moeHosted = slices.Grow(blas.moeHosted[:0], len(banks)*int(experts))[:len(banks)*int(experts)]
	clear(blas.moeHosted)
	if residency != nil {
		for index, bank := range banks {
			if err := residency.ExpertSources(state, bank, blas.moeSources[index*int(experts):(index+1)*int(experts)],
				blas.moeHosted[index*int(experts):(index+1)*int(experts)]); err != nil {
				return false, err
			}
		}
	}
	// copyBytes is an expert's bytes in host memory, aligned as copied; an
	// expert larger than a batch is read where it lies.
	copyBytes := func(expert uint32) uint64 {
		var bytes uint64
		for index := range banks {
			if host := blas.moeHosted[index*int(experts)+int(expert)]; len(host) > 0 {
				bytes += moeAlign(uint64(len(host)))
			}
		}
		if bytes > moeCopyBytes {
			return 0
		}
		return bytes
	}
	// Tiles of the card's experts, then of each batch of host experts; a
	// batch's end is its last tile and its experts follow in moeCopied.
	blas.moeTable = blas.moeTable[:0]
	blas.moeBatches = blas.moeBatches[:0]
	blas.moeCopied = blas.moeCopied[:0]
	addTiles := func(expert uint32) {
		for start, end := blas.moeOffsets[expert], blas.moeOffsets[expert+1]; start < end; start += moeTileTokens {
			blas.moeTable = append(blas.moeTable, expert, start, min(moeTileTokens, end-start))
		}
	}
	for expert := range experts {
		if blas.moeOffsets[expert+1] > blas.moeOffsets[expert] && copyBytes(expert) == 0 {
			addTiles(expert)
		}
	}
	free := uint64(0)
	for expert := range experts {
		bytes := copyBytes(expert)
		if bytes == 0 || blas.moeOffsets[expert+1] == blas.moeOffsets[expert] {
			continue
		}
		if bytes > free {
			blas.moeBatches = append(blas.moeBatches, moeBatch{tile: uint32(len(blas.moeTable)) / moeTileWords, copied: len(blas.moeCopied)})
			free = moeCopyBytes
		}
		free -= bytes
		blas.moeCopied = append(blas.moeCopied, expert)
		addTiles(expert)
	}
	tiles := uint32(len(blas.moeTable)) / moeTileWords
	blas.moeBatches = append(blas.moeBatches, moeBatch{tile: tiles, copied: len(blas.moeCopied)})
	// Each slot takes the next position of its expert's rows.
	blas.moeCounts = append(blas.moeCounts[:0], blas.moeOffsets[:experts]...)
	paired := int(count * topK)
	blas.moeTable = slices.Grow(blas.moeTable, paired)[:len(blas.moeTable)+paired]
	positions := blas.moeTable[tiles*moeTileWords:]
	for token := range count {
		for slot := range topK {
			expert := blas.moeRoutes[token*routeWords+slot] / divisor
			positions[blas.moeCounts[expert]] = token*topK + slot
			blas.moeCounts[expert]++
		}
	}
	table := blas.staging
	positionTable := table + driver.DevicePtr(uint64(tiles)*uint64(moeTileWords)*f32ScalarBytes)
	slots := blas.staging + driver.DevicePtr(g.planBytes(count, topK))
	copyTables := blas.staging + driver.DevicePtr(g.copiesAt(count, topK))
	copies := copyTables + driver.DevicePtr(moeAlign(moeBanks*g.experts*uint64(unsafe.Sizeof(driver.DevicePtr(0)))))
	blas.stagedNode = nil
	if err := state.Driver.MemcpyHtoD(table, driver.Bytes(blas.moeTable)); err != nil {
		return false, err
	}
	gateType := uint32(node.Inputs[moeFirstBankInput].Type)
	upType, downType := uint32(node.Inputs[upAt].Type), uint32(node.Inputs[upAt+1].Type)
	fused, activation, routing := kernelBool(a.FusedGateUp), uint32(a.Activation), uint32(a.Routing)
	clamp, indexed := a.SwiGLUClamp, kernelBool(residency != nil)
	block := driver.Dim3{X: moeTileThreads, Y: 1, Z: 1}
	// launch multiplies tiles first to end over the banks' tables.
	launch := func(first, end uint32, tables []driver.DevicePtr) error {
		if first == end {
			return nil
		}
		tileTable := table + driver.DevicePtr(uint64(first)*uint64(moeTileWords)*f32ScalarBytes)
		gate, up, down := tables[0], tables[upAt-moeFirstBankInput], tables[upAt-moeFirstBankInput+1]
		if err := launchGridSharedABI(state, functions[kernelMoeGroupedHiddenF32],
			driver.Dim3{X: end - first, Y: (intermediate-1)/moeTileRows + 1, Z: 1}, block, 0,
			&input, &tileTable, &positionTable, &gate, &up, &gateBias, &upBias, &activations, &hidden, &intermediate, &topK,
			&gateType, &upType, &fused, &activation, &routing, &clamp, &indexed); err != nil {
			return err
		}
		return launchGridSharedABI(state, functions[kernelMoeGroupedDownF32],
			driver.Dim3{X: end - first, Y: (hidden-1)/moeTileRows + 1, Z: 1}, block, 0,
			&activations, &tileTable, &positionTable, &down, &downBias, &slots, &hidden, &intermediate, &downType, &indexed)
	}
	if err := launch(0, blas.moeBatches[0].tile, tables); err != nil {
		return false, err
	}
	// Each batch copies its experts' bytes in host memory into the staging
	// after the previous batch's kernels, and names the copies in tables of
	// its own that keep every other expert's address.
	copiedTables := make([]driver.DevicePtr, len(banks))
	for index := range copiedTables {
		copiedTables[index] = copyTables + driver.DevicePtr(uint64(index)*g.experts*uint64(unsafe.Sizeof(driver.DevicePtr(0))))
	}
	for batch := 1; batch < len(blas.moeBatches); batch++ {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return false, err
		}
		blas.moeCopyTable = append(blas.moeCopyTable[:0], blas.moeSources...)
		at := copies
		for _, expert := range blas.moeCopied[blas.moeBatches[batch-1].copied:blas.moeBatches[batch].copied] {
			for index := range banks {
				source := index*int(experts) + int(expert)
				if host := blas.moeHosted[source]; len(host) > 0 {
					if err := state.Driver.MemcpyHtoD(at, host); err != nil {
						return false, err
					}
					blas.moeCopyTable[source] = at
					at += driver.DevicePtr(moeAlign(uint64(len(host))))
				}
			}
		}
		if err := state.Driver.MemcpyHtoD(copyTables, driver.Bytes(blas.moeCopyTable)); err != nil {
			return false, err
		}
		if err := launch(blas.moeBatches[batch-1].tile, blas.moeBatches[batch].tile, copiedTables); err != nil {
			return false, err
		}
	}
	normalize, scale := kernelBool(a.NormalizeTopKProb), a.Scale
	outputCount := count * hidden
	return true, launch1DABI(state, functions[kernelMoeCombineF32], outputCount,
		&routeState, &slots, &expertScale, &output, &hidden, &topK, &normalize, &scale, &divisor, &outputCount)
}

// moeBatch ends a batch of a group's tiles: the tile that follows it and
// the end of its experts copied from host memory.
type moeBatch struct {
	tile   uint32
	copied int
}
