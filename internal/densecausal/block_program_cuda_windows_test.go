package densecausal_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/densecausal"
	"overgo/internal/densecausaltrain"
	"overgo/internal/loratrain"
	"overgo/internal/testutil"
)

// TestDenseBlockAdapterTrainsOnTheDevice puts the block adapter's programs,
// the forward graph written once and the backward graph derived from it, on
// a device, and sets a step of theirs beside the trainers it would replace.
//
// Agreement. Two adapters over one block, one executed on the host and one
// on the device, are stepped toward one target from one state: every
// update's loss is the same to a part in a hundred thousand, and the loss
// falls, so what agrees is training and not a standing value.
//
// Measurement, on one block of Qwen2.5-0.5B's shape over a sequence of 128
// tokens, an adapter of rank eight on all seven projections. The block
// program's step is timed on the device and on the host, with the device
// memory it peaks at. The LoRA trainer, which runs on the host, and the
// resident trainer, which trains every weight on the device, train whole
// models, so each is run on models of one and of five such blocks and a
// block's share is a quarter of the difference. The figures are logged as the
// record docs/verification/dense-block-program-device.txt keeps them. Two
// are held, since they are what the program is for: its step on the device
// takes less time than the LoRA trainer's share for a block, and it peaks at
// less device memory than the resident trainer does on one block. The
// resident trainer's forward and backward for a block is logged beside the
// program's and not held: the record says which is faster.
func TestDenseBlockAdapterTrainsOnTheDevice(t *testing.T) {
	cudatest.Require(t)
	if cudatest.MeasurementProcess(t, 0) {
		return
	}
	ctx := t.Context()
	build := func(spec testutil.DenseCausalSpec) *densecausal.Model {
		t.Helper()
		weights, shapes := testutil.DenseCausalWeights(t, spec)
		model, err := densecausal.NewModel(weights, shapes, spec.Heads, spec.HeadDim, 10000, 1e-6)
		if err != nil {
			t.Fatal(err)
		}
		return model
	}
	random := rand.New(rand.NewPCG(5, 9))
	drawn := func(count int) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(random.NormFloat64())
		}
		return values
	}
	every := func(layer int) []string {
		names := densecausal.DenseLayerTensorNames(layer)
		return []string{names.Q, names.K, names.V, names.O, names.Gate, names.Up, names.Down}
	}
	const momentum = 0.9

	// Agreement, on a small block with grouped attention.
	{
		const seq, rank, alpha, layer, rate, updates = 6, 2, 4.0, 1, 0.01, 10
		spec := testutil.DenseCausalSpec{Vocab: 32, Hidden: 16, Heads: 2, HeadDim: 8, KVHeads: 1, Intermediate: 32, Layers: 2, Seed: 3}
		model := build(spec)
		state, target := drawn(seq*spec.Hidden), drawn(seq*spec.Hidden)
		host, err := densecausal.NewBlockAdapter(model, layer, seq, rank, alpha, every(layer), 7)
		if err != nil {
			t.Fatal(err)
		}
		placed, err := densecausal.NewBlockAdapter(model, layer, seq, rank, alpha, every(layer), 7)
		if err != nil {
			t.Fatal(err)
		}
		session, err := placed.OnDevice(ctx, 0)
		if err != nil {
			t.Fatal(err)
		}
		var first, last float64
		for update := range updates {
			want, err := host.Step(state, target, rate, momentum)
			if err != nil {
				t.Fatal(err)
			}
			got, err := placed.Step(state, target, rate, momentum)
			if err != nil {
				t.Fatal(err)
			}
			if math.IsNaN(got) || math.Abs(got-want) > 1e-5*math.Abs(want) {
				t.Fatalf("update %d: the loss is %.9g on the device and %.9g on the host, want them one training", update, got, want)
			}
			if update == 0 {
				first = got
			}
			last = got
		}
		if err := session.Close(ctx); err != nil {
			t.Fatal(err)
		}
		t.Logf("on the device as on the host, the loss fell from %.9g to %.9g over %d updates", first, last, updates)
		if last >= first {
			t.Fatalf("the loss went from %v to %v: the updates that agree train nothing", first, last)
		}
	}

	// Measurement, on a block of Qwen2.5-0.5B's shape.
	const seq, rank, alpha, rate, steps, hostSteps = 128, 8, 16.0, 0.01, 64, 3
	shaped := func(layers int) testutil.DenseCausalSpec {
		return testutil.DenseCausalSpec{Vocab: 64, Hidden: 896, Heads: 14, HeadDim: 64, KVHeads: 2, Intermediate: 4864, Layers: layers, Seed: 3}
	}
	const one, many = 1, 5
	t.Logf("shape: hidden %d, %d heads of %d over %d shared, intermediate %d, sequence %d, rank %d on all seven projections",
		shaped(one).Hidden, shaped(one).Heads, shaped(one).HeadDim, shaped(one).KVHeads, shaped(one).Intermediate, seq, rank)
	// measured logs one figure as the record keeps it.
	measured := func(name string, value int64) {
		t.Helper()
		t.Logf("measured: %s: %d", name, value)
	}
	single := build(shaped(one))
	state, target := drawn(seq*single.Dims.Hidden), drawn(seq*single.Dims.Hidden)
	// stepped is the mean time of a step of an adapter, after one that is not timed, and the first timed loss.
	stepped := func(adapter interface {
		Step(state, target []float32, learningRate, momentum float64) (float64, error)
	}, count int) (time.Duration, float64) {
		t.Helper()
		opening, err := adapter.Step(state, target, rate, momentum)
		if err != nil || math.IsNaN(opening) {
			t.Fatalf("loss %v (%v)", opening, err)
		}
		started := time.Now()
		for range count {
			if loss, err := adapter.Step(state, target, rate, momentum); err != nil || math.IsNaN(loss) {
				t.Fatalf("loss %v (%v)", loss, err)
			}
		}
		return time.Since(started) / time.Duration(count), opening
	}
	reset := func(state *device.State) error {
		state.Driver.ResetPeakBytes()
		return nil
	}
	placed, err := densecausal.NewBlockAdapter(single, 0, seq, rank, alpha, every(0), 7)
	if err != nil {
		t.Fatal(err)
	}
	session, err := placed.OnDevice(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}
	programStep, onDevice := stepped(placed, steps)
	started := time.Now()
	for range steps {
		if err := placed.Passes(state, target); err != nil {
			t.Fatal(err)
		}
	}
	programPasses := time.Since(started) / steps
	stats, err := session.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(ctx); err != nil {
		t.Fatal(err)
	}
	programPeak := stats.Device.PeakBytes
	measured("program-device-step-microseconds", programStep.Microseconds())
	measured("program-device-passes-microseconds", programPasses.Microseconds())
	measured("program-device-peak-bytes", int64(programPeak))
	measured("program-device-weights-bytes", int64(stats.StaticBytes))
	hosted, err := densecausal.NewBlockAdapter(single, 0, seq, rank, alpha, every(0), 7)
	if err != nil {
		t.Fatal(err)
	}
	hostStep, onHost := stepped(hosted, hostSteps)
	measured("program-host-step-microseconds", hostStep.Microseconds())
	if math.Abs(onDevice-onHost) > 1e-4*math.Abs(onHost) {
		t.Fatalf("at the measured shape the first loss is %.9g on the device and %.9g on the host", onDevice, onHost)
	}

	tokens := make([]int, seq)
	for index := range tokens {
		tokens[index] = random.IntN(shaped(one).Vocab)
	}
	lora := map[int]time.Duration{}
	for _, layers := range []int{one, many} {
		trainer, err := loratrain.New(build(shaped(layers)), loratrain.Config{Rank: rank, Alpha: alpha, Seed: 7})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := trainer.Step(tokens, nil, rate, momentum); err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		for range hostSteps {
			if _, err := trainer.Step(tokens, nil, rate, momentum); err != nil {
				t.Fatal(err)
			}
		}
		lora[layers] = time.Since(started) / hostSteps
		measured(fmt.Sprintf("lora-host-step-%d-blocks-microseconds", layers), lora[layers].Microseconds())
	}
	batches := make([][]int, steps)
	for index := range batches {
		batches[index] = tokens
	}
	type resident struct {
		passes, loop time.Duration
		peak         uint64
	}
	trained := map[int]resident{}
	for _, layers := range []int{one, many} {
		worker, err := device.New(0)
		if err != nil {
			t.Fatal(err)
		}
		// The first run loads the kernels; the second, over a model of its own, is the one measured.
		const residentRate = 1e-4
		if _, err := densecausaltrain.TrainDeviceResident(build(shaped(layers)), worker, batches[:1], residentRate, momentum, densecausaltrain.Options{FrozenLexical: true}); err != nil {
			t.Fatal(err)
		}
		if err := worker.Do(ctx, reset); err != nil {
			t.Fatal(err)
		}
		result, err := densecausaltrain.TrainDeviceResident(build(shaped(layers)), worker, batches, residentRate, momentum, densecausaltrain.Options{FrozenLexical: true, Measure: true})
		if err != nil {
			t.Fatal(err)
		}
		memory, err := worker.MemoryStats(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if err := worker.Close(); err != nil {
			t.Fatal(err)
		}
		trained[layers] = resident{passes: result.Measurement.ForwardBackward / steps, loop: result.Measurement.Loop / steps, peak: memory.PeakBytes}
		measured(fmt.Sprintf("resident-passes-%d-blocks-microseconds", layers), trained[layers].passes.Microseconds())
		measured(fmt.Sprintf("resident-step-%d-blocks-microseconds", layers), trained[layers].loop.Microseconds())
		measured(fmt.Sprintf("resident-peak-%d-blocks-bytes", layers), int64(trained[layers].peak))
	}
	const between = many - one
	loraBlock := (lora[many] - lora[one]) / between
	t.Logf("a block's share: the LoRA trainer's step %s; the resident trainer's forward and backward %s, its step %s, its peak %d bytes",
		loraBlock, (trained[many].passes-trained[one].passes)/between, (trained[many].loop-trained[one].loop)/between, (trained[many].peak-trained[one].peak)/between)
	if programStep >= loraBlock {
		t.Fatalf("the program's step on the device takes %s and the LoRA trainer's share for a block %s, want the program's the shorter", programStep, loraBlock)
	}
	if programPeak >= trained[one].peak {
		t.Fatalf("the program peaks at %d bytes of device memory and the resident trainer on one block at %d, want the program's the less", programPeak, trained[one].peak)
	}
}
