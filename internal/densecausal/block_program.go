package densecausal

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"strings"

	"overgo/internal/optimizer"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tensorgrad"
)

// The names of an adapter's two factors, after the weight they adapt: the
// down factor takes the projection's input to the rank, the up factor the
// rank to its output.
const (
	blockDown = ".down"
	blockUp   = ".up"
	// blockHalf is the half a squared error is taken at, so that its
	// gradient is the error itself.
	blockHalf = 0.5
)

// blockAdapter trains a low-rank adapter on the projections of one dense
// block through a tensor program. The block's forward pass is written once,
// as a graph of the operations that serve it: the weighted normalizations,
// the attention's projections, rotary encoding and grouped causal attention,
// and the gated feed-forward half, each projection adapted being its weight's
// product and the scaled product of the adapter's two factors. Its backward
// pass is not written: the gradients of half the squared error against a
// target, by the factors, are derived from that graph by the operations' own
// rules (tensorgrad). A step executes both and takes Muon's update of the
// factors, which are the trainer's own values, as the optimizer's state is.
type blockAdapter struct {
	state, target *tensor.Tensor
	output, miss  *tensor.Tensor
	// fixed are the model's weights, which both programs read; constants
	// are the backward graph's, which only a step reads. An executor that
	// holds them itself empties both.
	fixed, constants map[*tensor.Tensor][]float32
	factors          map[string][]float32
	nodes            map[string]*tensor.Tensor
	gradients        map[string]*tensor.Tensor
	// outputs are the block's output, then what a step reads beside it;
	// execute runs the first of them alone or all of them, on the host
	// until the adapter is put on a device.
	outputs   []*tensor.Tensor
	execute   func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error)
	pack      *optimizer.TensorPack
	optimizer *optimizer.Optimizer
	config    optimizer.Config
}

// newBlockAdapter builds the program of layer index of a model over a
// sequence of seq tokens, with an adapter of the given rank and alpha on the
// projections named, by their weights' names. The down factors are drawn as
// a LoRA adapter's are and the up factors start at nothing, so the adapted
// block starts as the block.
func newBlockAdapter(m *Model, index, seq, rank int, alpha float64, adapted []string, seed uint64) (*blockAdapter, error) {
	if m == nil || index < 0 || index >= len(m.layers) || seq <= 0 || rank <= 0 || alpha <= 0 || len(adapted) == 0 || m.layers[index].moe != nil {
		return nil, errors.New("densecausal: a block adapter is a dense layer of a model, a sequence length, a rank, an alpha and the projections it adapts")
	}
	d, names := m.Dims, m.layers[index].names
	b := tensor.NewBuilder()
	adapter := &blockAdapter{fixed: map[*tensor.Tensor][]float32{}, constants: map[*tensor.Tensor][]float32{}, factors: map[string][]float32{}, nodes: map[string]*tensor.Tensor{}, gradients: map[string]*tensor.Tensor{}}
	input := func(name string, dimensions ...uint64) *tensor.Tensor {
		shape, err := tensor.NewShape(dimensions...)
		if err != nil {
			return nil
		}
		return b.Input(name, dtype.F32, shape)
	}
	held := func(name string, dimensions ...uint64) *tensor.Tensor {
		node := input(name, dimensions...)
		adapter.fixed[node] = m.Weights[name]
		return node
	}
	random := rand.New(rand.NewPCG(seed, seed))
	scale := float32(alpha / float64(rank))
	// project is rows through a projection of the layer, its bias added where the model has one, and the adapter's update where it is adapted.
	project := func(weight, bias string, rows *tensor.Tensor, in, out int) *tensor.Tensor {
		projected := b.MulMatWithCompute(held(weight, uint64(in), uint64(out)), rows, tensor.MulMatComputeExact)
		for _, name := range adapted {
			if name != weight {
				continue
			}
			down, bound := make([]float32, rank*in), 1/math.Sqrt(float64(in))
			for at := range down {
				down[at] = float32((random.Float64() - blockHalf) / blockHalf * bound)
			}
			adapter.factors[weight+blockDown], adapter.factors[weight+blockUp] = down, make([]float32, out*rank)
			adapter.nodes[weight+blockDown], adapter.nodes[weight+blockUp] = input(weight+blockDown, uint64(in), uint64(rank)), input(weight+blockUp, uint64(rank), uint64(out))
			update := b.MulMatWithCompute(adapter.nodes[weight+blockUp], b.MulMatWithCompute(adapter.nodes[weight+blockDown], rows, tensor.MulMatComputeExact), tensor.MulMatComputeExact)
			projected = b.Add(projected, b.Scale(update, scale))
		}
		if bias != "" && m.Weights[bias] != nil {
			projected = b.Add(projected, held(bias, uint64(out)))
		}
		return projected
	}
	hidden, width, shared := uint64(d.Hidden), d.Heads*d.HeadDim, d.KVHeads*d.HeadDim
	positions := make([]uint32, seq)
	for position := range positions {
		positions[position] = uint32(position)
	}
	rotary := tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: uint32(d.HeadDim), FrequencyBase: float32(d.RopeTheta), FrequencyScale: 1}
	heads := func(rows *tensor.Tensor, count int) *tensor.Tensor {
		return b.Reshape(rows, uint64(d.HeadDim), uint64(count), uint64(seq))
	}
	adapter.state = input("block.state", hidden, uint64(seq))
	normed := b.Multiply(b.RMSNorm(adapter.state, float32(d.RMSEps)), held(names.InLN, hidden))
	mixed := b.AttentionWithOptions(
		b.RoPEWithOptions(heads(project(names.Q, names.QB, normed, d.Hidden, width), d.Heads), rotary),
		b.RoPEWithOptions(heads(project(names.K, names.KB, normed, d.Hidden, shared), d.KVHeads), rotary),
		heads(project(names.V, names.VB, normed, d.Hidden, shared), d.KVHeads),
		tensor.AttentionOptions{Scale: float32(1 / math.Sqrt(float64(d.HeadDim))), Causal: true})
	attended := b.Add(adapter.state, project(names.O, "", b.Reshape(mixed, uint64(width), uint64(seq)), width, d.Hidden))
	settled := b.Multiply(b.RMSNorm(attended, float32(d.RMSEps)), held(names.PostLN, hidden))
	gated := b.Multiply(b.SiLU(project(names.Gate, "", settled, d.Hidden, d.Intermediate)), project(names.Up, "", settled, d.Hidden, d.Intermediate))
	adapter.output = b.Add(attended, project(names.Down, "", gated, d.Intermediate, d.Hidden))
	if len(adapter.factors) != len(adapted)+len(adapted) {
		return nil, fmt.Errorf("densecausal: layer %d has no projection among %v", index, adapted)
	}
	adapter.target = input("block.target", hidden, uint64(seq))
	adapter.miss = b.Add(adapter.output, b.Scale(adapter.target, -1))
	var wanted []*tensor.Tensor
	var order []string
	for name, node := range adapter.nodes {
		order, wanted = append(order, name), append(wanted, node)
	}
	derived, constants, err := tensorgrad.Backward(b, adapter.output, b.Scale(adapter.miss, 1/float32(seq*d.Hidden)), wanted...)
	if err != nil {
		return nil, err
	}
	adapter.outputs = []*tensor.Tensor{adapter.output, adapter.miss}
	for at, name := range order {
		adapter.gradients[name] = derived[at]
		adapter.outputs = append(adapter.outputs, derived[at])
	}
	for _, constant := range constants {
		adapter.constants[constant.Tensor] = constant.Data
	}
	forward, err := reference.Compile(adapter.output)
	if err != nil {
		return nil, err
	}
	both, err := reference.Compile(adapter.outputs...)
	if err != nil {
		return nil, err
	}
	adapter.execute = func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
		program := both
		if len(outputs) == 1 {
			program = forward
		}
		inputs := program.NewInputs()
		for node, value := range feeds {
			if err := inputs.Set(node, value); err != nil {
				return nil, err
			}
		}
		return program.Execute(inputs, program.NewWorkspace())
	}
	adapter.pack, err = optimizer.NewTensorPack(adapter.factors, func(name string, length int) (int, int, error) {
		if strings.HasSuffix(name, blockDown) {
			return rank, length / rank, nil
		}
		return length / rank, rank, nil
	})
	return adapter, err
}

// run executes the block's output alone, or everything a step reads, on a
// state, and a target where a step reads one, with the factors as they stand
// and what the adapter's executor is not holding of the weights and constants.
func (adapter *blockAdapter) run(outputs []*tensor.Tensor, feeds map[*tensor.Tensor][]float32) (map[*tensor.Tensor]reference.Value, error) {
	for name, node := range adapter.nodes {
		feeds[node] = adapter.factors[name]
	}
	maps.Copy(feeds, adapter.fixed)
	if len(outputs) > 1 {
		maps.Copy(feeds, adapter.constants)
	}
	read := make(map[*tensor.Tensor]reference.Value, len(feeds))
	for node, data := range feeds {
		value, err := reference.BorrowedValue(node.Shape, data)
		if err != nil {
			return nil, fmt.Errorf("densecausal: block input %q: %w", node.Name, err)
		}
		read[node] = value
	}
	return adapter.execute(outputs, read)
}

// kept is what a later adapter over the same block resumes from: the factors
// as they stand, copied, and the optimizer's state after the updates taken.
func (adapter *blockAdapter) kept() (map[string][]float32, optimizer.State, error) {
	if adapter.optimizer == nil {
		return nil, optimizer.State{}, errors.New("densecausal: a block adapter that has taken no update has nothing to resume from")
	}
	factors := make(map[string][]float32, len(adapter.factors))
	for name, values := range adapter.factors {
		factors[name] = append([]float32(nil), values...)
	}
	return factors, adapter.optimizer.Snapshot(), nil
}

// resume makes the adapter the one that kept these factors and this state,
// under the rate and momentum it took its updates with, so that its next
// update is the one that adapter would have taken next.
func (adapter *blockAdapter) resume(factors map[string][]float32, state optimizer.State, learningRate, momentum float64) error {
	if err := adapter.pack.RestoreWeights(factors); err != nil {
		return err
	}
	adapter.pack.Scatter()
	config := optimizer.Config{BaseLearningRate: learningRate, Momentum: momentum, Schedule: optimizer.ScheduleConstant}
	created, err := adapter.pack.NewOptimizer(config)
	if err != nil {
		return err
	}
	adapter.optimizer, adapter.config = created, config
	return adapter.optimizer.Restore(state)
}

// outputOf is the adapted block's output for a state: a sequence's rows of
// the model's width.
func (adapter *blockAdapter) outputOf(state []float32) ([]float32, error) {
	values, err := adapter.run(adapter.outputs[:1], map[*tensor.Tensor][]float32{adapter.state: state})
	if err != nil {
		return nil, err
	}
	return values[adapter.output].Data, nil
}

// step takes one update of the factors toward a target for a state and
// returns the loss before it: half the mean squared error of the adapted
// block's output against the target.
func (adapter *blockAdapter) step(state, target []float32, learningRate, momentum float64) (float64, error) {
	config := optimizer.Config{BaseLearningRate: learningRate, Momentum: momentum, Schedule: optimizer.ScheduleConstant}
	if adapter.optimizer == nil {
		created, err := adapter.pack.NewOptimizer(config)
		if err != nil {
			return 0, err
		}
		adapter.optimizer, adapter.config = created, config
	} else if config != adapter.config {
		return 0, errors.New("densecausal: a block adapter's learning rate and momentum are fixed by its first update")
	}
	values, err := adapter.run(adapter.outputs, map[*tensor.Tensor][]float32{adapter.state: state, adapter.target: target})
	if err != nil {
		return 0, err
	}
	var loss float64
	for _, miss := range values[adapter.miss].Data {
		loss += blockHalf * float64(miss) * float64(miss)
	}
	gradients := make(map[string][]float32, len(adapter.gradients))
	for name, node := range adapter.gradients {
		gradients[name] = values[node].Data
	}
	if err := adapter.pack.GatherGradients(gradients); err != nil {
		return 0, err
	}
	adapter.optimizer.Step()
	adapter.pack.Scatter()
	return loss / float64(len(values[adapter.miss].Data)), nil
}
