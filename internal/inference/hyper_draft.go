package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modeldevice"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// hyperDraft is the draft layer a draft file attaches to a hyper-connection trunk (Flash-Next's MTP layer).
type hyperDraft struct {
	file    *gguf.File
	weights model.HyperDraftWeights
	program model.CompiledLayerProgram
	// step is the compiled draft step, nil until the first draft.
	step *hyperDraftStep
}

// hyperDraftStep is the compiled draft step, replayed as one captured graph with the token, the position and the streams of each draft.
//   - a round drafts one token from a fresh cell, so the step attends over that cell alone
type hyperDraftStep struct {
	graph              *executor.IndexedGraph
	attributes         *executor.RuntimeAttributes
	rows               *tensor.GetRowsAttributes
	positions          [][]uint32
	feeds              map[*tensor.Tensor]reference.Value
	streams, selection *tensor.Tensor
}

// openHyperDraft binds a draft file to the trunk the plan compiled.
func openHyperDraft(path string, plan model.ModelPlan, spec model.Spec) (*hyperDraft, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("inference: open draft %s: %w", path, err)
	}
	weights, err := model.BindHyperDraftWeights(file, spec)
	if err != nil {
		return nil, errors.Join(fmt.Errorf("inference: draft %s: %w", path, err), file.Close())
	}
	program, err := plan.HyperDraftProgram()
	if err != nil {
		return nil, errors.Join(fmt.Errorf("inference: draft %s: %w", path, err), file.Close())
	}
	return &hyperDraft{file: file, weights: weights, program: program}, nil
}

func (d *hyperDraft) close() error {
	if d == nil || d.file == nil {
		return nil
	}
	err := d.file.Close()
	d.file, d.step = nil, nil
	return err
}

// loadHyperDraft uploads an attached draft layer: its routed-expert banks as stored, every other tensor as F32.
//   - it runs before the trunk's pass, so an expert residency sizes itself to the memory the draft leaves
func loadHyperDraft(ctx context.Context, draft *hyperDraft, raw *modeldevice.DeviceWeights, converted *modeldevice.DeviceConvertedWeights) error {
	if draft == nil {
		return nil
	}
	loader := modeldevice.NewLoader(draft.file)
	banks := map[string]bool{
		draft.weights.Layer.FeedForwardGateUpExperts.Name: true, draft.weights.Layer.FeedForwardDownExperts.Name: true,
	}
	var native, f32 []gguf.TensorInfo
	for _, info := range draft.weights.Tensors() {
		if raw != nil && banks[info.Name] && info.Type != dtype.F32 {
			native = append(native, info)
		} else {
			f32 = append(f32, info)
		}
	}
	if len(native) != 0 {
		loader.Raw(raw, native)
	}
	loader.Converted(converted, f32)
	return loader.Load(ctx, nil)
}

// hyperDraftWidth is the trunk residual row a hyper draft reads: every stream of one position.
func (r *Runner) hyperDraftWidth() uint64 {
	return uint64(r.spec.EmbeddingLength) * uint64(r.spec.HyperConnectionCount)
}

// compileHyperDraftStep builds the draft step.
//   - the draft file stores the checkpoint's norms, scaled by one plus the weight, so the step adds the unit offset
func (r *Runner) compileHyperDraftStep() (*hyperDraftStep, error) {
	draft, builder := r.hyperDraft, r.newGraphBuilder()
	step := &hyperDraftStep{feeds: map[*tensor.Tensor]reference.Value{}}
	var deviceFeeds tensor.InputBindings[driver.DevicePtr]
	var bindErr error
	weight := func(info gguf.TensorInfo) *tensor.Tensor {
		node, pointer, err := r.deviceInput(builder, info)
		bindErr = errors.Join(bindErr, err)
		deviceFeeds.Add(node, pointer)
		return node
	}
	host := func(name string, shape tensor.Shape, fill float32) *tensor.Tensor {
		node := builder.Input(name, dtype.F32, shape)
		elements, err := shape.Elements()
		bindErr = errors.Join(bindErr, err)
		data := make([]float32, elements)
		for index := range data {
			data[index] = fill
		}
		step.feeds[node] = reference.Value{Shape: shape, Data: data}
		return node
	}
	offset := func(node *tensor.Tensor) *tensor.Tensor {
		return builder.Add(node, host("hyper_draft.unit_offset", node.Shape, 1))
	}
	w := draft.weights
	embedding := builder.GetRows(weight(r.weights.TokenEmbedding), []uint32{0})
	step.streams = host("hyper_draft.streams", tensor.MustShape(r.hyperDraftWidth(), tensor.SingletonExtent), 0)
	current, err := model.BuildHyperDraftInput(
		builder, r.spec, embedding, step.streams, offset(weight(w.EmbeddingNorm)), offset(weight(w.HiddenNorm)),
		weight(w.EmbeddingProjection), weight(w.HiddenProjection),
	)
	if err = errors.Join(err, bindErr); err != nil {
		return nil, err
	}
	layer, layerFeeds, err := modeldevice.BindDeviceLayerGraphInputs(builder, w.Layer, r.deviceInput)
	if err != nil {
		return nil, err
	}
	deviceFeeds = append(deviceFeeds, layerFeeds...)
	layer.AttentionQNorm, layer.AttentionKNorm = offset(layer.AttentionQNorm), offset(layer.AttentionKNorm)
	layer.HyperAttentionNorm, layer.HyperFeedForwardNorm = offset(layer.HyperAttentionNorm), offset(layer.HyperFeedForwardNorm)
	plan := draft.program.Layer()
	block, err := draft.program.Build(model.CachedBlockContext{
		Builder: builder, Input: current, Positions: []uint32{0}, Layer: plan.Layer, Recurrent: plan.Recurrent,
	}, layer)
	if err != nil {
		return nil, err
	}
	mixed, err := model.BuildHyperDraftHeadInput(builder, r.spec, block.Output, offset(weight(w.MixerNorm)), weight(w.MixerDown), weight(w.MixerUp))
	if err != nil {
		return nil, err
	}
	step.selection = builder.TopK(builder.MulMat(weight(r.outputTensor()), mixed), 1)
	if err := errors.Join(builder.Err(), bindErr); err != nil {
		return nil, err
	}
	if step.graph, err = executor.CompileIndexed(step.selection); err != nil {
		return nil, err
	}
	if err := step.graph.Inputs.Bind(deviceFeeds); err != nil {
		return nil, err
	}
	// The token row and every rotary position are per-draft attributes, so one capture serves every draft of this shape.
	step.attributes = step.graph.Graph.NewRuntimeAttributes()
	rows := embedding.Attrs.(tensor.GetRowsAttributes)
	rows.Rows = []uint32{0}
	step.rows = &rows
	if err := step.attributes.Set(embedding, step.rows); err != nil {
		return nil, err
	}
	nodes, err := tensor.Topological(step.selection)
	if err != nil {
		return nil, err
	}
	for _, node := range nodes {
		var value tensor.Attributes
		switch attributes := node.Attrs.(type) {
		case tensor.RoPEAttributes:
			attributes.Positions = []uint32{0}
			step.positions = append(step.positions, attributes.Positions)
			value = &attributes
		case tensor.RoPEMultiAttributes:
			for axis := range attributes.Positions {
				if len(attributes.Positions[axis]) != 0 {
					attributes.Positions[axis] = []uint32{0}
					step.positions = append(step.positions, attributes.Positions[axis])
				}
			}
			value = &attributes
		default:
			continue
		}
		if err := step.attributes.Set(node, value); err != nil {
			return nil, err
		}
	}
	return step, nil
}

// advanceHyperDraftLocked drafts the token after tokenID from the session's streams with the attached draft layer.
//   - the session's cell sits at the position of the streams it read, one before the token it embeds
//   - a round drafts one token, so no session follows it
func (r *Runner) advanceHyperDraftLocked(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *MTPSession,
) (tokenizer.TokenID, *MTPSession, error) {
	draft := r.hyperDraft
	if draft == nil || session == nil || !r.hasPreloadedWeights() {
		return 0, nil, errors.New("inference: hyper-connection draft is unavailable")
	}
	if tokenID < 0 || int(tokenID) >= r.vocab.Len() {
		return 0, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	if !tensor.HasDimensions(session.PendingHidden.Shape, r.hyperDraftWidth(), tensor.SingletonExtent) {
		return 0, nil, errors.New("inference: hyper-connection draft streams are incompatible")
	}
	if draft.step == nil {
		step, err := r.compileHyperDraftStep()
		if err != nil {
			return 0, nil, err
		}
		draft.step = step
	}
	step := draft.step
	step.rows.Rows[0] = uint32(tokenID)
	for _, positions := range step.positions {
		positions[0] = session.Position
	}
	step.feeds[step.streams] = session.PendingHidden
	retained, err := r.cuda.ExecuteRetainedCompiled(ctx, step.graph.Graph, step.feeds, step.graph.Inputs, nil, step.attributes)
	if err != nil {
		return 0, nil, err
	}
	selection, err := retained.CopyToHost(ctx, step.selection)
	if err = errors.Join(err, retained.Release(context.Background())); err != nil {
		return 0, nil, err
	}
	if len(selection.Data) != 1 || selection.Data[0] < 0 || int(selection.Data[0]) >= r.vocab.Len() {
		return 0, nil, errors.New("inference: hyper-connection draft selection is invalid")
	}
	return tokenizer.TokenID(selection.Data[0]), nil, nil
}
