package executor

import (
	"errors"
	"math"
	"runtime"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/hostexpert"
	"overgo/internal/tensor"
)

// hostExpertSlots is the most slots of a prompt group an expert wholly in
// host memory may have and be computed on the CPU where it lies, beside the
// card's products, rather than copied to the card: a Flash-Next expert's
// three matrices take about 160 us over the link and about 5 us a slot on
// the 9950X's 32 workers (hostexpert's BenchmarkLayer), so up to about 32
// slots the CPU costs less.
const hostExpertSlots = 32

// gatedBanks is a gated expert layer's banks, gate, up and down in order, and
// fusedHalves the halves a fused layer's first bank holds, gate's then up's.
const (
	gatedBanks  = 3
	fusedHalves = 2
)

// hostExperts is a group's experts computed on the CPU: whether each expert
// is, the experts in order, and their slots' positions in expert order.
type hostExperts struct {
	onCPU     []bool
	experts   []uint32
	slots     uint32
	gateIndex int
	upIndex   int
	downIndex int
	fused     bool
	storage   hostexpert.Formats
}

// planHostExperts chooses a group's experts to compute on the CPU: every
// bank of the expert in host memory, at most hostExpertSlots slots, in a
// format and with an activation the CPU computes, and no bias; the staging
// reserves their outputs (hostRowsReserve). It returns false where none is
// chosen.
func (b *blasState) planHostExperts(a tensor.MoEAttributes, banks []*tensor.Tensor, g moeMatrixGeometry,
	count uint32, biased bool, residency ExpertBanks) (hostExperts, bool) {
	plan := hostExperts{fused: a.FusedGateUp, upIndex: 1, downIndex: gatedBanks - 1}
	if a.FusedGateUp {
		plan.upIndex, plan.downIndex = 0, plan.downIndex-1
	}
	if residency == nil || biased || !a.Gated || a.Activation != tensor.MoEActivationSiLU || len(banks) != plan.downIndex+1 {
		return plan, false
	}
	plan.storage = hostexpert.Formats{Gate: banks[plan.gateIndex].Type, Up: banks[plan.upIndex].Type, Down: banks[plan.downIndex].Type}
	if !hostexpert.Supported(plan.storage, int(g.hidden), int(g.intermediate)) {
		return plan, false
	}
	experts := uint32(g.experts)
	plan.onCPU = make([]bool, experts)
	for expert := range experts {
		slots := b.moeOffsets[expert+1] - b.moeOffsets[expert]
		if slots == 0 || slots > hostExpertSlots {
			continue
		}
		hosted := true
		for index := range banks {
			hosted = hosted && len(b.moeHosted[index*int(experts)+int(expert)]) > 0
		}
		if hosted {
			plan.onCPU[expert] = true
			plan.experts = append(plan.experts, expert)
			plan.slots += slots
		}
	}
	if plan.slots == 0 || hostRowsBytes(g, plan.slots) > g.hostRowsReserve() || g.stagingBytes(count, a.TopK) > b.stagingBytes {
		return plan, false
	}
	return plan, true
}

// hostRowsBytes is the staging the CPU's outputs of slots slots and their
// positions take.
func hostRowsBytes(g moeMatrixGeometry, slots uint32) uint64 {
	return moeAlign(uint64(slots)*g.hidden*f32ScalarBytes) + moeAlign(uint64(slots)*uint64(unsafe.Sizeof(uint32(0))))
}

// hostActivation is moe_activation for the gated SiLU the CPU computes: the
// up product clamped to the SwiGLU limit where one is set, the gate too under
// square-root-softplus routing and its activation otherwise, then gate's
// SiLU times up.
func hostActivation(a tensor.MoEAttributes) func(gate, up float32) float32 {
	limit, clampGate := a.SwiGLUClamp, a.Routing == tensor.MoERoutingSqrtSoftplus
	return func(gate, up float32) float32 {
		if limit > 0 {
			up = min(limit, max(-limit, up))
			if clampGate {
				gate = min(limit, gate)
			}
		}
		activated := float32(gate / float32(1+math.Exp(-float64(gate))))
		if limit > 0 && !clampGate {
			activated = min(limit, activated)
		}
		return activated * up
	}
}

// pinned grows a pinned host buffer to at least bytes, freeing the one it
// replaces.
func pinned(state *device.State, buffer *[]byte, bytes uint64) error {
	if uint64(len(*buffer)) >= bytes {
		return nil
	}
	if len(*buffer) > 0 {
		if err := state.Driver.MemFreeHost(*buffer); err != nil {
			return err
		}
		*buffer = nil
	}
	data, _, err := state.Driver.MemHostAlloc(bytes)
	if err != nil {
		return err
	}
	*buffer = data
	return nil
}

// gatherHostInputs brings to host memory the input rows of the tokens the
// plan's experts read, gathered on the card into the staging and copied back
// together, each token once: the whole group's rows took a 5,099-token
// prompt about 4 ms a layer. It runs before the card's products are queued,
// so its copy waits for nothing else.
func (b *blasState) gatherHostInputs(state *device.State, functions functionSet, plan hostExperts, g moeMatrixGeometry,
	a tensor.MoEAttributes, input driver.DevicePtr, count uint32, positions []uint32) error {
	topK := a.TopK
	b.hostTokens = b.hostTokens[:0]
	if cap(b.hostQuantizedAt) < int(count) {
		b.hostQuantizedAt = make([]int, count)
	}
	quantizedAt := b.hostQuantizedAt[:count]
	for index := range quantizedAt {
		quantizedAt[index] = -1
	}
	for _, expert := range plan.experts {
		for _, position := range positions[b.moeOffsets[expert]:b.moeOffsets[expert+1]] {
			if token := position / topK; quantizedAt[token] < 0 {
				quantizedAt[token] = len(b.hostTokens)
				b.hostTokens = append(b.hostTokens, token)
			}
		}
	}
	tokens := uint32(len(b.hostTokens))
	gatheredBytes := uint64(tokens) * g.hidden * f32ScalarBytes
	if err := pinned(state, &b.hostInput, gatheredBytes); err != nil {
		return err
	}
	// The tokens' indices and their rows take the staging the CPU's outputs and positions take later.
	rowsAt := b.staging + driver.DevicePtr(g.hostRowsAt(count, topK))
	indicesAt := rowsAt + driver.DevicePtr(moeAlign(uint64(plan.slots)*g.hidden*f32ScalarBytes))
	if err := state.Driver.MemcpyHtoD(indicesAt, driver.Bytes(b.hostTokens)); err != nil {
		return err
	}
	width, total := uint32(g.hidden), tokens*uint32(g.hidden)
	if err := launch1DABI(state, functions[kernelGetRowsF32], total, &input, &indicesAt, &rowsAt, &width, &total); err != nil {
		return err
	}
	return state.Driver.MemcpyDtoH(b.hostInput[:gatheredBytes], rowsAt)
}

// computeHostExperts computes the plan's experts on the CPU from the rows
// gatherHostInputs gathered and the slot positions in expert order, and
// queues their outputs into the slots at slots on the compute stream. The
// card's products are already queued, so the two run together.
func (b *blasState) computeHostExperts(state *device.State, functions functionSet, plan hostExperts, g moeMatrixGeometry,
	a tensor.MoEAttributes, slots driver.DevicePtr, count uint32, positions []uint32) error {
	hidden, intermediate, topK, experts := int(g.hidden), int(g.intermediate), a.TopK, int(g.experts)
	if b.hostPool == nil {
		b.hostPool = hostexpert.NewPool(runtime.NumCPU())
	}
	rowsBytes := uint64(plan.slots) * g.hidden * f32ScalarBytes
	positionBytes := uint64(plan.slots) * uint64(unsafe.Sizeof(uint32(0)))
	if err := pinned(state, &b.hostRows, rowsBytes+positionBytes); err != nil {
		return err
	}
	tokens := len(b.hostTokens)
	inputs := unsafe.Slice((*float32)(unsafe.Pointer(&b.hostInput[0])), tokens*hidden)
	rows := unsafe.Slice((*float32)(unsafe.Pointer(&b.hostRows[0])), int(plan.slots)*hidden)
	packed := unsafe.Slice((*uint32)(unsafe.Pointer(&b.hostRows[rowsBytes])), int(plan.slots))
	quantizedAt := b.hostQuantizedAt[:count]
	// Each token a CPU expert reads is quantized once.
	quantizedBytes := hostexpert.Q8KBytes(hidden)
	if len(b.hostQuantized) < tokens*quantizedBytes {
		b.hostQuantized = make([]byte, tokens*quantizedBytes)
	}
	values, blocks := make([][]float32, tokens), make([][]byte, tokens)
	for index := range tokens {
		values[index] = inputs[index*hidden : (index+1)*hidden]
		blocks[index] = b.hostQuantized[index*quantizedBytes : (index+1)*quantizedBytes]
	}
	if err := b.hostPool.QuantizeQ8K(values, blocks); err != nil {
		return err
	}
	work := make([]hostexpert.Expert, len(plan.experts))
	row := 0
	for index, expert := range plan.experts {
		hosted := func(bank int) []byte { return b.moeHosted[bank*experts+int(expert)] }
		gate, up := hosted(plan.gateIndex), hosted(plan.upIndex)
		if plan.fused {
			half := len(gate) / fusedHalves
			gate, up = gate[:half], gate[half:]
		}
		work[index] = hostexpert.Expert{Gate: gate, Up: up, Down: hosted(plan.downIndex)}
		for _, position := range positions[b.moeOffsets[expert]:b.moeOffsets[expert+1]] {
			work[index].Activations = append(work[index].Activations, blocks[quantizedAt[position/topK]])
			work[index].Out = append(work[index].Out, rows[row*hidden:(row+1)*hidden])
			packed[row] = position
			row++
		}
	}
	if row != int(plan.slots) {
		return errors.New("host experts' slots disagree with their plan")
	}
	if err := b.hostPool.Layer(work, plan.storage, hidden, intermediate, hostActivation(a)); err != nil {
		return err
	}
	rowsAt := b.staging + driver.DevicePtr(g.hostRowsAt(count, topK))
	positionsAt := rowsAt + driver.DevicePtr(moeAlign(rowsBytes))
	if err := state.Driver.MemcpyHtoDAsync(rowsAt, b.hostRows[:rowsBytes], state.Stream); err != nil {
		return err
	}
	if err := state.Driver.MemcpyHtoDAsync(positionsAt, b.hostRows[rowsBytes:rowsBytes+positionBytes], state.Stream); err != nil {
		return err
	}
	hiddenWidth, total := uint32(hidden), plan.slots*uint32(hidden)
	return launch1DABI(state, functions[kernelMoeHostSlotsF32], total, &rowsAt, &positionsAt, &slots, &hiddenWidth, &total)
}
