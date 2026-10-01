// Package e4btrain trains every parameter of a Gemma-4 E4B decoder on its
// own next-token objective: the sliding and full attention layers, the
// layers that attend an earlier layer's keys and values, the feed-forward
// blocks, the per-layer input adapters with their shared projection, the
// layer output scales and the final norm, through the tied soft-capped
// head. The lexical tables (token and per-layer token embeddings) stay
// frozen, as every trainer here freezes lexical tensors. Weights are host
// f32 masters in one slab the optimizer plan spans; the windowed attention
// VJP runs on the device, and each Muon step streams one parameter group at
// a time across it.
package e4btrain

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"overgo/internal/optimizer"
)

// Tensor roles, named as the artifact names them, so a checkpoint's tensors
// carry the names of the tensors they train.
const (
	roleAttentionNorm      = "attn_norm"
	roleQuery              = "attn_q"
	roleQueryNorm          = "attn_q_norm"
	roleKey                = "attn_k"
	roleKeyNorm            = "attn_k_norm"
	roleValue              = "attn_v"
	roleAttentionOutput    = "attn_output"
	rolePostAttentionNorm  = "post_attention_norm"
	roleFeedForwardNorm    = "ffn_norm"
	roleGate               = "ffn_gate"
	roleUp                 = "ffn_up"
	roleDown               = "ffn_down"
	rolePostFeedForward    = "post_ffw_norm"
	roleAdapterGate        = "per_layer_inp_gate"
	roleAdapterProjection  = "per_layer_proj"
	roleAdapterNorm        = "per_layer_post_norm"
	roleOutputScale        = "layer_output_scale"
	rolePerLayerProjection = "per_layer_model_proj"
	rolePerLayerNorm       = "per_layer_proj_norm"
	roleOutputNorm         = "output_norm"
	layerPrefix            = "blk."
	weightSuffix           = ".weight"
)

const (
	// scalarWidth is the width of a layer output scale: one trained scalar.
	scalarWidth = 1
	// shortestWindow is the fewest tokens a next-token objective trains
	// on: one input and the target after it.
	shortestWindow = 2
	// pairWidth is the rotary pair: one inverse frequency rotates two values.
	pairWidth = 2
)

// layerGeometry is one decoder layer's declared shape. Keys and values have
// one width, headDim. The rotary inverse frequencies cover the rotary width
// (their count is half of it) with the layer's frequency factors folded in.
// A layer whose source is another layer projects no keys or values of its
// own and attends that layer's.
type layerGeometry struct {
	heads, kvHeads, headDim int
	inter                   int
	window                  int // keys a query attends back to; 0 is the whole causal prefix
	source                  int
	invFreq                 []float64
	scaled                  bool // the layer output is multiplied by a trained scalar
}

// geometry is the decoder's declared shape and scalar policy. A zero
// perLayer declares no per-layer inputs, and a zero softcap no logit cap.
type geometry struct {
	hidden, vocab, perLayer int
	eps                     float64
	attentionScale          float64
	embeddingScale          float64
	logitScale              float64
	softcap                 float64
	perLayerScale           float64 // weight of the per-layer inputs' projected + selected sum
	layers                  []layerGeometry
}

// layerParameters are one layer's tensor views, in the slab or the gradient
// slab. Tensors the layer does not declare stay nil.
type layerParameters struct {
	attentionNorm, query, queryNorm, key, keyNorm, value []float32
	attentionOutput, postAttentionNorm                   []float32
	feedForwardNorm, gate, up, down, postFeedForward     []float32
	adapterGate, adapterProjection, adapterNorm          []float32
	outputScale                                          []float32
}

// globalParameters are the views that belong to no one layer.
type globalParameters struct {
	perLayerProjection, perLayerNorm, outputNorm []float32
}

// Model is a Gemma-4 E4B decoder bound for full-parameter training on one
// token window: every trainable tensor is a view of the flat f32 slab its
// optimizer plan spans, with a matching view of the gradient slab. The tied
// output table and the window's per-layer embedding rows are frozen inputs.
type Model struct {
	shape                   geometry
	plan                    optimizer.Plan
	weights, gradients      []float32
	layers, layerGradients  []layerParameters
	global, globalGradients globalParameters
	head                    []float32 // [vocab, hidden]; the input rows are its rows
	perLayerRows            []float32 // [tokens, layers*perLayer]
	inputs, targets         []int
}

// parameter is one trainable tensor: its checkpoint name, its [rows, cols]
// geometry, and the views it binds in the weight and gradient slabs.
type parameter struct {
	name             string
	rows, cols       int
	weight, gradient *[]float32
}

// newModel validates the geometry and the frozen inputs, lays every
// trainable tensor out in one slab in layer order and binds the views. The
// weights start at zero; the caller fills them. ids is the token window:
// every token but the last is an input, every token but the first a target.
func newModel(shape geometry, head, perLayerRows []float32, ids []int) (*Model, error) {
	if len(ids) < shortestWindow {
		return nil, errors.New("e4b model: a token window needs an input and a target")
	}
	for _, token := range ids {
		if token < 0 || token >= shape.vocab {
			return nil, fmt.Errorf("e4b model: token %d is outside the %d-token vocabulary", token, shape.vocab)
		}
	}
	m := &Model{
		shape: shape, head: head, perLayerRows: perLayerRows,
		inputs: ids[:len(ids)-1], targets: ids[1:],
		layers: make([]layerParameters, len(shape.layers)), layerGradients: make([]layerParameters, len(shape.layers)),
	}
	if err := m.validate(); err != nil {
		return nil, err
	}
	parameters := m.parameters()
	specs := make([]optimizer.GroupSpec, len(parameters))
	total := 0
	for index, parameter := range parameters {
		size := parameter.rows * parameter.cols
		specs[index] = optimizer.GroupSpec{Name: parameter.name, Start: total, End: total + size, Rows: parameter.rows, Cols: parameter.cols}
		total += size
	}
	plan, err := optimizer.CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	m.plan = plan
	m.weights = make([]float32, total)
	m.gradients = make([]float32, total)
	for index, parameter := range parameters {
		spec := specs[index]
		*parameter.weight = m.weights[spec.Start:spec.End:spec.End]
		*parameter.gradient = m.gradients[spec.Start:spec.End:spec.End]
	}
	return m, nil
}

// validate holds the geometry and the frozen inputs to what the forward
// reads.
func (m *Model) validate() error {
	g := m.shape
	if g.hidden <= 0 || g.vocab <= 0 || g.perLayer < 0 || len(g.layers) == 0 || !(g.eps > 0) ||
		!(g.attentionScale > 0) || !(g.embeddingScale > 0) || !(g.logitScale > 0) || g.softcap < 0 {
		return errors.New("e4b model: invalid decoder geometry")
	}
	if len(m.head) != g.vocab*g.hidden {
		return fmt.Errorf("e4b model: output table has %d values, need %d", len(m.head), g.vocab*g.hidden)
	}
	if want := len(m.inputs) * len(g.layers) * g.perLayer; len(m.perLayerRows) != want {
		return fmt.Errorf("e4b model: per-layer embedding rows have %d values, need %d", len(m.perLayerRows), want)
	}
	if g.perLayer > 0 && !(g.perLayerScale > 0) {
		return errors.New("e4b model: per-layer input scale is invalid")
	}
	for index, layer := range g.layers {
		rotary := len(layer.invFreq) * pairWidth
		if layer.heads <= 0 || layer.kvHeads <= 0 || layer.heads%layer.kvHeads != 0 || layer.headDim <= 0 ||
			rotary <= 0 || rotary > layer.headDim || layer.inter <= 0 || layer.window < 0 || layer.source < 0 || layer.source > index {
			return fmt.Errorf("e4b model: layer %d geometry is invalid", index)
		}
		if layer.source == index {
			continue
		}
		source := g.layers[layer.source]
		if source.source != layer.source || source.kvHeads != layer.kvHeads || source.headDim != layer.headDim ||
			!equalFrequencies(source.invFreq, layer.invFreq) {
			return fmt.Errorf("e4b model: layer %d attends layer %d, whose keys and values differ in shape or rotation", index, layer.source)
		}
	}
	return nil
}

func equalFrequencies(left, right []float64) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if math.Float64bits(left[index]) != math.Float64bits(right[index]) {
			return false
		}
	}
	return true
}

// parameters lists every trainable tensor in slab order: each layer's in
// forward order, then the per-layer projection and its norm, then the final
// norm.
func (m *Model) parameters() []parameter {
	g := m.shape
	var parameters []parameter
	add := func(name string, rows, cols int, weight, gradient *[]float32) {
		parameters = append(parameters, parameter{name: name, rows: rows, cols: cols, weight: weight, gradient: gradient})
	}
	vector := func(name string, width int, weight, gradient *[]float32) {
		add(name, scalarWidth, width, weight, gradient)
	}
	for index, layer := range g.layers {
		w, d := &m.layers[index], &m.layerGradients[index]
		name := func(role string) string { return layerPrefix + strconv.Itoa(index) + "." + role + weightSuffix }
		queryWidth, keyWidth := layer.heads*layer.headDim, layer.kvHeads*layer.headDim
		vector(name(roleAttentionNorm), g.hidden, &w.attentionNorm, &d.attentionNorm)
		add(name(roleQuery), queryWidth, g.hidden, &w.query, &d.query)
		vector(name(roleQueryNorm), layer.headDim, &w.queryNorm, &d.queryNorm)
		if layer.source == index {
			add(name(roleKey), keyWidth, g.hidden, &w.key, &d.key)
			vector(name(roleKeyNorm), layer.headDim, &w.keyNorm, &d.keyNorm)
			add(name(roleValue), keyWidth, g.hidden, &w.value, &d.value)
		}
		add(name(roleAttentionOutput), g.hidden, queryWidth, &w.attentionOutput, &d.attentionOutput)
		vector(name(rolePostAttentionNorm), g.hidden, &w.postAttentionNorm, &d.postAttentionNorm)
		vector(name(roleFeedForwardNorm), g.hidden, &w.feedForwardNorm, &d.feedForwardNorm)
		add(name(roleGate), layer.inter, g.hidden, &w.gate, &d.gate)
		add(name(roleUp), layer.inter, g.hidden, &w.up, &d.up)
		add(name(roleDown), g.hidden, layer.inter, &w.down, &d.down)
		vector(name(rolePostFeedForward), g.hidden, &w.postFeedForward, &d.postFeedForward)
		if g.perLayer > 0 {
			add(name(roleAdapterGate), g.perLayer, g.hidden, &w.adapterGate, &d.adapterGate)
			add(name(roleAdapterProjection), g.hidden, g.perLayer, &w.adapterProjection, &d.adapterProjection)
			vector(name(roleAdapterNorm), g.hidden, &w.adapterNorm, &d.adapterNorm)
		}
		if layer.scaled {
			vector(name(roleOutputScale), scalarWidth, &w.outputScale, &d.outputScale)
		}
	}
	w, d := &m.global, &m.globalGradients
	if g.perLayer > 0 {
		add(rolePerLayerProjection+weightSuffix, len(g.layers)*g.perLayer, g.hidden, &w.perLayerProjection, &d.perLayerProjection)
		vector(rolePerLayerNorm+weightSuffix, g.perLayer, &w.perLayerNorm, &d.perLayerNorm)
	}
	vector(roleOutputNorm+weightSuffix, g.hidden, &w.outputNorm, &d.outputNorm)
	return parameters
}

// ParameterCount is the number of trained values.
func (m *Model) ParameterCount() int { return len(m.weights) }

// OptimizerPlan is the plan over the weight slab: one group per trainable
// tensor, named as the artifact names it.
func (m *Model) OptimizerPlan() optimizer.Plan { return m.plan }

// TrainedTensor is the trained values of the tensor the artifact names
// name, a view of the master slab, when the model trains it.
func (m *Model) TrainedTensor(name string) ([]float32, bool) {
	for _, group := range m.plan.Groups() {
		if group.Name == name {
			return m.weights[group.Start:group.End], true
		}
	}
	return nil, false
}

// SaveWeights writes every trainable tensor at path as a named F32
// safetensors tensor with its [rows, cols] geometry: the weights a Snapshot
// of the trainer stepping the model pairs with.
func (m *Model) SaveWeights(path string) error {
	return optimizer.SavePlanWeights(path, m.plan, m.weights)
}

// gradientL2 is the L2 norm of the gradient slab.
func (m *Model) gradientL2() float64 {
	var squared float64
	for _, value := range m.gradients {
		squared += float64(value) * float64(value)
	}
	return math.Sqrt(squared)
}
