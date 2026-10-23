package executor

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync/atomic"
	"time"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/hostexpert"
	"overgo/internal/tensor"
)

// moeDoorbell computes a captured decode step's routed experts that lie in
// host memory on the CPU, beside the card, as Strata's doorbell does
// (src/core/layer.cpp, THE DOORBELL; ops_f32.cu, The expert doorbell). Each
// routed node of the step rings as its hidden products start, handing over
// the slots of experts on host with its tokens' inputs; the card computes the
// other slots while the worker that launched the step computes those and
// answers, and the card adds them. The link reads Flash-Next's 2.3 MB expert
// at about 13 GB/s, 0.17 ms; the CPU computes it from host memory in about
// 0.06 ms (hostexpert's BenchmarkDecodeExpert), the card's own slots covering
// about half of that.
//
// The words, slot ids, input rows and answer rows are one pinned allocation
// the card maps; descriptor is card memory, the CUDA moe_doorbell: the rings
// so far, the ring the next down products wait for and their blocks
// finished, then the addresses of the mapped parts. The group's hidden and
// down products ring and wait themselves, so a ring takes no launch.
type moeDoorbell struct {
	host       []byte
	mapped     driver.DevicePtr
	descriptor driver.DevicePtr
	slots      uint64
	width      uint64
	words      []uint32
	ids        []uint32
	rows       []float32
	answer     []float32
	pool       *hostexpert.Pool
	// capture collects the routed nodes a capture rings for.
	capture []*tensor.Tensor
	// pointers and hosted hold a routed node's banks' sources; quantized and
	// activations the Q8_K activations of its tokens; work its experts.
	pointers    []driver.DevicePtr
	hosted      [][]byte
	quantized   []byte
	activations [][]byte
	values      [][]float32
	work        []hostexpert.Expert
}

// The doorbell's words, the CUDA MOE_DOORBELL_*: the ring, the CPU's answer,
// the last ring that handed slots over, the ring an add gave up on and the
// host's abandonment of the run; doorbellWordCount pads them to 32 bytes.
const (
	doorbellRing = iota
	doorbellAnswer
	doorbellWork
	doorbellFailed
	doorbellAbandoned
	doorbellWordCount = 8
)

const (
	// doorbellHostSlot marks a handed-over slot of an expert the CPU computes, the CUDA MOE_DOORBELL_HOST_SLOT.
	doorbellHostSlot = uint32(1) << 31
	// doorbellSlots is the most routed slots a group hands over, the CUDA MOE_DOORBELL_SLOTS.
	doorbellSlots = 128
	// doorbellVector is the values the last block of the down products adds at once, a float4.
	doorbellVector = 4
	// doorbellPatience bounds the worker's wait for a ring: a step rings within milliseconds of its launch.
	doorbellPatience = 2 * time.Second
	// doorbellCPUShare divides the logical CPUs into the CPU's workers: the worker spins beside them, and half read
	// memory at about the rate of all.
	doorbellCPUShare = 2
	// doorbellPolls is the polls of a ring between reads of the clock.
	doorbellPolls = 1 << 12
)

// ensure allocates the doorbell for slots routed slots of tokens width wide,
// once, and again when a graph needs more, and starts the CPU's workers.
func (d *moeDoorbell) ensure(state *device.State, slots, width uint64) error {
	if d.host != nil && slots <= d.slots && width <= d.width {
		return nil
	}
	if err := d.free(state); err != nil {
		return err
	}
	slots, width = max(slots, d.slots), max(width, d.width)
	// The answer rows start aligned, as the card reads them a float4 at a time.
	idBytes := moeAlign(slots * uint64(unsafe.Sizeof(uint32(0))))
	rowBytes := moeAlign(slots * width * f32ScalarBytes)
	host, mapped, err := state.Driver.MemHostAlloc(doorbellWordCount*uint64(unsafe.Sizeof(uint32(0))) + idBytes + rowBytes + rowBytes)
	if err != nil {
		return err
	}
	idsAt := mapped + doorbellWordCount*driver.DevicePtr(unsafe.Sizeof(uint32(0)))
	rowsAt := idsAt + driver.DevicePtr(idBytes)
	// The CUDA moe_doorbell: four counts, each zero at first, in two eight-byte fields, then the four addresses.
	fields := []uint64{0, 0, uint64(mapped), uint64(idsAt), uint64(rowsAt), uint64(rowsAt) + rowBytes}
	descriptor, err := state.Driver.MemAlloc(uint64(len(fields)) * uint64(unsafe.Sizeof(uint64(0))))
	if err == nil {
		if err = state.Driver.MemcpyHtoD(descriptor, driver.Bytes(fields)); err != nil {
			err = errors.Join(err, state.Driver.MemFree(descriptor))
		}
	}
	if err != nil {
		return errors.Join(err, state.Driver.MemFreeHost(host))
	}
	clear(host)
	d.host, d.mapped, d.descriptor, d.slots, d.width = host, mapped, descriptor, slots, width
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&host[0])), len(host)/int(unsafe.Sizeof(uint32(0))))
	d.words = words[:doorbellWordCount]
	d.ids = words[doorbellWordCount : doorbellWordCount+int(slots)]
	floats := unsafe.Slice((*float32)(unsafe.Pointer(&host[0])), len(words))
	rowsIndex := doorbellWordCount + int(idBytes/f32ScalarBytes)
	answerIndex := rowsIndex + int(rowBytes/f32ScalarBytes)
	d.rows = floats[rowsIndex : rowsIndex+int(slots*width)]
	d.answer = floats[answerIndex : answerIndex+int(slots*width)]
	if d.pool == nil {
		d.pool = hostexpert.NewPool(max(1, runtime.NumCPU()/doorbellCPUShare))
	}
	return nil
}

// rings reports whether a routed node's decode group of count tokens rings:
// a group of a captured run whose doorbell holds its slots, whose products
// are the q8 decode products, and whose experts the CPU computes. A nil
// doorbell, an uncaptured run's, rings for none.
func (d *moeDoorbell) rings(node *tensor.Tensor, q8Input *q8InputState, count uint32) bool {
	if d == nil || d.host == nil || uint64(count) > q8InputSpanColumns {
		return false
	}
	a := node.Attrs.(tensor.MoEAttributes)
	g, _, plan, ok := doorbellGeometry(node)
	slots := uint64(count) * uint64(a.TopK)
	if _, matrix := nativeMoEMatrix(node); !ok || matrix || slots > min(d.slots, doorbellSlots) || g.hidden > d.width ||
		g.hidden%doorbellVector != 0 {
		return false
	}
	hiddenQ8, downQ8 := moeQ8Products(a, plan.storage.Gate, plan.storage.Up, plan.storage.Down, uint32(g.hidden),
		uint32(g.intermediate))
	_, q8Bytes := moeQ8Bytes(uint32(g.hidden), uint32(g.intermediate), a.TopK, count)
	return hiddenQ8 && downQ8 && q8Input != nil && q8Input.staging != 0 && q8Bytes <= q8Input.stagingBytes
}

// free releases the doorbell's memory; the CPU's workers stay.
func (d *moeDoorbell) free(state *device.State) error {
	var errs []error
	if d.host != nil {
		errs = append(errs, state.Driver.MemFreeHost(d.host))
	}
	if d.descriptor != 0 {
		errs = append(errs, state.Driver.MemFree(d.descriptor))
	}
	d.host, d.mapped, d.descriptor, d.words, d.ids, d.rows, d.answer = nil, 0, 0, nil, nil, nil, nil
	return errors.Join(errs...)
}

// close releases the doorbell and stops its workers.
func (d *moeDoorbell) close(state *device.State) error {
	if d.pool != nil {
		d.pool.Close()
		d.pool = nil
	}
	return d.free(state)
}

// prepare clears the words a run's failure sets and returns the ring before the run, before it is launched.
func (d *moeDoorbell) prepare() uint32 {
	atomic.StoreUint32(&d.words[doorbellAbandoned], 0)
	atomic.StoreUint32(&d.words[doorbellFailed], 0)
	return atomic.LoadUint32(&d.words[doorbellRing])
}

// serve answers each ring of a launched run whose rings follow base, one a
// routed node in order, computing the slots it hands over. A failure
// abandons the run, whose adds then write NaN at once.
func (d *moeDoorbell) serve(ctx context.Context, state *device.State, experts ExpertBanks, rings []*tensor.Tensor, base uint32) error {
	d.pool.Spin(true)
	defer d.pool.Spin(false)
	for index, node := range rings {
		ring := base + uint32(index) + 1
		err := d.await(ctx, state, ring)
		// A ring that hands slots over holds the card until its answer, so the last such ring is this one.
		if err == nil && atomic.LoadUint32(&d.words[doorbellWork]) == ring {
			err = d.compute(state, experts, node)
		}
		if err != nil {
			d.abandon()
			return fmt.Errorf("the doorbell of tensor %d: %w", node.ID, err)
		}
		atomic.StoreUint32(&d.words[doorbellAnswer], ring)
	}
	return nil
}

// await spins until the card rings ring. It queries the stream between
// reads of the clock: Windows' driver batches submissions, and a run it has
// not submitted rings only once the worker calls into it (Strata's round
// 195: a spin that only reads memory saw the ring when the stream was
// synchronized, never before).
func (d *moeDoorbell) await(ctx context.Context, state *device.State, ring uint32) error {
	start := time.Now()
	for polls := 0; int32(atomic.LoadUint32(&d.words[doorbellRing])-ring) < 0; polls++ {
		if polls%doorbellPolls == 0 {
			if _, err := state.Driver.StreamQuery(state.Stream); err != nil {
				return err
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if time.Since(start) > doorbellPatience {
				return fmt.Errorf("ring %d did not come within %s", ring, doorbellPatience)
			}
		}
	}
	return nil
}

// failure reports the ring a run's add gave up waiting on, once the run is done.
func (d *moeDoorbell) failure() error {
	if ring := atomic.LoadUint32(&d.words[doorbellFailed]); ring != 0 {
		return fmt.Errorf("the card gave up waiting for the CPU's experts of ring %d", ring)
	}
	return nil
}

// compute computes the slots a ring of node handed over into the answer
// rows, each expert once for its tokens, as computeHostExperts computes a
// prompt's.
func (d *moeDoorbell) compute(state *device.State, residency ExpertBanks, node *tensor.Tensor) error {
	a := node.Attrs.(tensor.MoEAttributes)
	g, banks, plan, ok := doorbellGeometry(node)
	if !ok {
		return errors.New("a routed node the CPU cannot compute rang")
	}
	experts, hidden, intermediate := int(g.experts), int(g.hidden), int(g.intermediate)
	tokens, topK, divisor := int(node.Shape.Dims[1]), int(a.TopK), max(a.ExpertIndexDivisor, 1)
	if len(d.pointers) < len(banks)*experts {
		d.pointers, d.hosted = make([]driver.DevicePtr, len(banks)*experts), make([][]byte, len(banks)*experts)
	}
	for index, bank := range banks {
		if err := residency.ExpertSources(state, bank, d.pointers[index*experts:(index+1)*experts], d.hosted[index*experts:(index+1)*experts]); err != nil {
			return err
		}
	}
	quantizedBytes := hostexpert.Q8KBytes(hidden)
	if len(d.quantized) < tokens*quantizedBytes {
		d.quantized = make([]byte, tokens*quantizedBytes)
	}
	d.activations, d.values, d.work = d.activations[:0], d.values[:0], d.work[:0]
	quantizedAt := make([]int, tokens)
	for token := range quantizedAt {
		quantizedAt[token] = -1
	}
	var order []uint32
	for slot, id := range d.ids[:tokens*topK] {
		if id&doorbellHostSlot == 0 {
			continue
		}
		token, expert := slot/topK, (id&^doorbellHostSlot)/divisor
		if int(expert) >= experts {
			return fmt.Errorf("a handed-over slot names expert %d of %d", expert, experts)
		}
		if quantizedAt[token] < 0 {
			quantizedAt[token] = len(d.values)
			d.values = append(d.values, d.rows[token*hidden:(token+1)*hidden])
			d.activations = append(d.activations, d.quantized[len(d.activations)*quantizedBytes:(len(d.activations)+1)*quantizedBytes])
		}
		at := slices.Index(order, expert)
		if at < 0 {
			hosted := func(bank int) []byte { return d.hosted[bank*experts+int(expert)] }
			gate, up, down := hosted(plan.gateIndex), hosted(plan.upIndex), hosted(plan.downIndex)
			if gate == nil || up == nil || down == nil {
				return fmt.Errorf("expert %d on host has no host bytes", expert)
			}
			if plan.fused {
				half := len(gate) / fusedHalves
				gate, up = gate[:half], gate[half:]
			}
			at, order = len(order), append(order, expert)
			d.work = append(d.work, hostexpert.Expert{Gate: gate, Up: up, Down: down})
		}
		d.work[at].Activations = append(d.work[at].Activations, d.activations[quantizedAt[token]])
		d.work[at].Out = append(d.work[at].Out, d.answer[slot*hidden:(slot+1)*hidden])
	}
	if len(d.work) == 0 {
		return errors.New("a ring handed over no slot")
	}
	if err := d.pool.QuantizeQ8K(d.values, d.activations); err != nil {
		return err
	}
	return d.pool.Layer(d.work, plan.storage, hidden, intermediate, hostActivation(a))
}

// doorbellGeometry returns a routed node's geometry, banks and the CPU's
// plan of them where the CPU computes its experts on host for a ring: gated
// SiLU experts without biases, in formats it reads; a fused first bank holds
// gate and up.
func doorbellGeometry(node *tensor.Tensor) (moeMatrixGeometry, []*tensor.Tensor, hostExperts, bool) {
	a, ok := node.Attrs.(tensor.MoEAttributes)
	plan := hostExperts{fused: a.FusedGateUp, upIndex: 1, downIndex: gatedBanks - 1}
	if a.FusedGateUp {
		plan.upIndex, plan.downIndex = 0, plan.downIndex-1
	}
	if !ok || !a.Gated || a.Activation != tensor.MoEActivationSiLU || a.HasExpertBiases || a.HasRouterBias ||
		len(node.Inputs) <= moeFirstBankInput+plan.downIndex {
		return moeMatrixGeometry{}, nil, plan, false
	}
	banks := node.Inputs[moeFirstBankInput : moeFirstBankInput+plan.downIndex+1]
	down := banks[plan.downIndex]
	plan.storage = hostexpert.Formats{Gate: banks[plan.gateIndex].Type, Up: banks[plan.upIndex].Type, Down: down.Type}
	// A bank's last axis counts its experts.
	g := moeMatrixGeometry{hidden: node.Shape.Dims[0], intermediate: down.Shape.Dims[0], experts: down.Shape.Dims[down.Shape.Rank-1]}
	if !hostexpert.Supported(plan.storage, int(g.hidden), int(g.intermediate)) {
		return moeMatrixGeometry{}, nil, plan, false
	}
	return g, banks, plan, true
}

// abandon releases a launched run's adds from their waits; each writes NaN.
func (d *moeDoorbell) abandon() {
	atomic.StoreUint32(&d.words[doorbellAbandoned], 1)
}
