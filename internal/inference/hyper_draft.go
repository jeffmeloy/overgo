package inference

import (
	"context"
	"encoding/binary"
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
//   - vocabulary, when a file names it, is the token ids whose rows of the trunk's head the draft's head reads, head
//     those rows on the card, so a draft step reads that share of the head (Flash-Next: 106,299 of 248,320 tokens)
type hyperDraft struct {
	file       *gguf.File
	weights    model.HyperDraftWeights
	program    model.CompiledLayerProgram
	vocabulary []uint32
	head       gguf.TensorInfo
	// steps are the compiled draft rounds by their rows, over the history's capacity.
	steps map[uint32]*hyperDraftStep
	// history is the draft layer's own keys and values of the request's decoded cells, nil until the first draft.
	history *hyperDraftHistory
}

// hyperDraftHistory is the draft layer's keys and values of a request's cells, as Strata's drafter keeps its layer's:
// cell c pairs the trunk's streams at position start+c with the token after them, at rope position start+c, so a
// draft attends over every cell before its own, as the layer was trained, where a cell alone drafted 73% of Flash-Next's
// greedy answers' next-next tokens and the answer's cells 80%.
//   - keys and values are [key length, kv heads, capacity] each in one buffer, cells of them written
type hyperDraftHistory struct {
	buffer       *executor.DeviceBuffer
	keys, values executor.DeviceValue
	capacity     uint32
	start, cells uint32
}

// hyperDraftHistoryCells is the fewest cells a history holds: a request's own decoded tokens, rounded up to a power of
// two so a capacity serves many requests' compiled rounds.
const hyperDraftHistoryCells = 1024

// hyperDraftHead names the draft's head over its vocabulary among the device weights.
const hyperDraftHead = "hyper_draft.head"

// draftVocabularyIDBytes is a draft vocabulary's token id, a little-endian int32.
const draftVocabularyIDBytes = 4

// hyperDraftStep is a compiled draft round: rows cells, each a row of the trunk's streams with the token after it,
// appended to the history at an offset and attending over it, the last row drafting; replayed as one captured graph
// with each round's tokens, positions, streams and offset.
type hyperDraftStep struct {
	graph              *executor.IndexedGraph
	attributes         *executor.RuntimeAttributes
	targets            *executor.RetainedTargets
	rows               *tensor.GetRowsAttributes
	positions          [][]uint32
	attention          []*tensor.AttentionAttributes
	appends            []*tensor.CacheAppendAttributes
	feeds              map[*tensor.Tensor]reference.Value
	streams, selection *tensor.Tensor
	cells              uint32
}

// openHyperDraft binds a draft file to the trunk the plan compiled, with the vocabulary its head reads when the file
// holds one.
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
	vocabulary, err := readDraftVocabulary(file)
	if err != nil {
		return nil, errors.Join(fmt.Errorf("inference: draft %s: %w", path, err), file.Close())
	}
	return &hyperDraft{file: file, weights: weights, program: program, vocabulary: vocabulary}, nil
}

// readDraftVocabulary reads the token ids of the draft file's vocabulary tensor, none when it holds none; placing the
// head refuses one past its rows.
func readDraftVocabulary(file *gguf.File) ([]uint32, error) {
	info, ok := file.Tensor(model.HyperDraftVocabulary)
	if !ok {
		return nil, nil
	}
	if info.Type != dtype.I32 || info.Dimensions != 1 || info.Shape[0] == 0 || info.Size != info.Shape[0]*draftVocabularyIDBytes {
		return nil, fmt.Errorf("vocabulary tensor %s is not a list of int32 token ids", info.Name)
	}
	data := make([]byte, info.Size)
	if err := file.ReadTensorData(info, data); err != nil {
		return nil, err
	}
	vocabulary := make([]uint32, info.Shape[0])
	for index := range vocabulary {
		vocabulary[index] = binary.LittleEndian.Uint32(data[index*draftVocabularyIDBytes:])
	}
	return vocabulary, nil
}

func (d *hyperDraft) close() error {
	if d == nil || d.file == nil {
		return nil
	}
	err := d.file.Close()
	d.file = nil
	return err
}

// loadHyperDraft uploads an attached draft layer: its routed-expert banks and its BF16 matrices as stored, every other
// tensor as F32.
//   - a BF16 matrix takes the trunk's raw route, read as stored by the draft step's products at half an F32 copy's bytes
//     (its F32 copies took about 1.1 ms of mul_mat_f32 a Flash-Next draft step); the draft holds no convolution kernel
//     that the trunk keeps in F32
//   - a draft with a vocabulary places its rows of the trunk's head, as the file stores them, read from the trunk's file
//   - it runs before the trunk's pass, so an expert residency sizes itself to the memory the draft leaves
func loadHyperDraft(
	ctx context.Context, draft *hyperDraft, trunk *gguf.File, head gguf.TensorInfo, raw *modeldevice.DeviceWeights,
	converted *modeldevice.DeviceConvertedWeights,
) error {
	if draft == nil {
		return nil
	}
	if len(draft.vocabulary) != 0 {
		if raw == nil {
			return errors.New("inference: a draft vocabulary needs the trunk's head as stored on the card")
		}
		var err error
		if draft.head, err = raw.PlaceRows(ctx, trunk, head, hyperDraftHead, draft.vocabulary); err != nil {
			return err
		}
	}
	loader := modeldevice.NewLoader(draft.file)
	layer := draft.weights.Layer
	banks := map[string]bool{layer.FeedForwardGateUpExperts.Name: true, layer.FeedForwardDownExperts.Name: true}
	// The routers stay F32, as the routed product reads them (the trunk's file stores its routers so).
	routers := map[string]bool{}
	for _, router := range []*gguf.TensorInfo{layer.FeedForwardRouter, layer.FeedForwardSharedRouter} {
		if router != nil {
			routers[router.Name] = true
		}
	}
	var native, f32 []gguf.TensorInfo
	for _, info := range draft.weights.Tensors() {
		matrix := info.Type == dtype.BF16 && info.Dimensions == tensor.PairedExtent && !routers[info.Name]
		if raw != nil && (banks[info.Name] && info.Type != dtype.F32 || matrix) {
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

// ensureHyperDraftHistory gives the draft a history of at least cells, empty: a larger one drops the rounds compiled
// over the old one, whose captured graphs address its buffer.
func (r *Runner) ensureHyperDraftHistory(ctx context.Context, cells uint32) error {
	draft := r.hyperDraft
	capacity := uint32(hyperDraftHistoryCells)
	for capacity < cells {
		capacity *= tensor.PairedExtent
	}
	if history := draft.history; history != nil && history.capacity >= capacity {
		history.start, history.cells = 0, 0
		return nil
	}
	if err := r.releaseHyperDraftHistory(ctx); err != nil {
		return err
	}
	keyShape := tensor.MustShape(uint64(r.spec.KeyLength), uint64(r.spec.HeadCountKV), uint64(capacity))
	valueShape := tensor.MustShape(uint64(r.spec.ValueLength), uint64(r.spec.HeadCountKV), uint64(capacity))
	keyElements, keyErr := keyShape.Elements()
	valueElements, valueErr := valueShape.Elements()
	if err := errors.Join(keyErr, valueErr); err != nil {
		return err
	}
	buffer, err := r.cuda.AllocateDeviceBuffer(ctx, (keyElements+valueElements)*cacheScalarBytes)
	if err != nil {
		return err
	}
	whole, err := buffer.Value(tensor.MustShape(keyElements + valueElements))
	if err == nil {
		history := &hyperDraftHistory{buffer: buffer, capacity: capacity}
		if history.keys, err = deviceCacheStorageView(whole, 0, keyShape); err == nil {
			history.values, err = deviceCacheStorageView(whole, keyElements, valueShape)
		}
		if err == nil {
			draft.history, draft.steps = history, map[uint32]*hyperDraftStep{}
			return nil
		}
	}
	return errors.Join(err, buffer.Release(ctx))
}

// releaseHyperDraftHistory frees the draft's history and the rounds compiled over it.
func (r *Runner) releaseHyperDraftHistory(ctx context.Context) error {
	if r.hyperDraft == nil || r.hyperDraft.history == nil {
		return nil
	}
	err := r.hyperDraft.history.buffer.Release(ctx)
	r.hyperDraft.history, r.hyperDraft.steps = nil, nil
	return err
}

// compileHyperDraftStep builds a draft round of rows cells over the history.
//   - the draft file stores the checkpoint's norms, scaled by one plus the weight, so the step adds the unit offset
//   - the cells append to the history in place at the round's offset, and attend over it and over the cells before
//     them in the round, the last cell's mixed streams drafting
func (r *Runner) compileHyperDraftStep(rows uint32) (*hyperDraftStep, error) {
	draft, builder, history := r.hyperDraft, r.newGraphBuilder(), r.hyperDraft.history
	builder.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: 0, SourceCapacityTokens: history.capacity, CapacityTokens: history.capacity})
	step := &hyperDraftStep{feeds: map[*tensor.Tensor]reference.Value{}, cells: rows}
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
	past := func(name string, value executor.DeviceValue) *tensor.Tensor {
		node := builder.Input(name, dtype.F32, value.Shape)
		deviceFeeds.Add(node, value.Pointer)
		return node
	}
	w := draft.weights
	embedding := builder.GetRows(weight(r.weights.TokenEmbedding), make([]uint32, rows))
	step.streams = host("hyper_draft.streams", tensor.MustShape(r.hyperDraftWidth(), uint64(rows)), 0)
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
		Builder: builder, Input: current, Positions: make([]uint32, rows), Layer: plan.Layer, Recurrent: plan.Recurrent,
		PastKey: past("hyper_draft.history.key", history.keys), PastValue: past("hyper_draft.history.value", history.values),
		CacheWrite: tensor.CacheWriteAppend,
	}, layer)
	if err != nil {
		return nil, err
	}
	mixed, err := model.BuildHyperDraftHeadInput(builder, r.spec, block.Output, offset(weight(w.MixerNorm)), weight(w.MixerDown), weight(w.MixerUp))
	if err != nil {
		return nil, err
	}
	width := uint64(r.spec.EmbeddingLength)
	last := builder.FlatSlice(mixed, width*uint64(rows-1), width, 1)
	head := r.outputTensor()
	if len(draft.vocabulary) != 0 {
		head = draft.head
	}
	step.selection = builder.TopK(builder.MulMat(weight(head), last), 1)
	if err := errors.Join(builder.Err(), bindErr); err != nil {
		return nil, err
	}
	if step.graph, err = executor.CompileIndexed(step.selection, block.Key, block.Value); err != nil {
		return nil, err
	}
	// The draft's token comes down with its run; the history's keys and values are written in place.
	if err := step.graph.Graph.ReadBack(step.selection); err != nil {
		return nil, err
	}
	if err := step.graph.Inputs.Bind(deviceFeeds); err != nil {
		return nil, err
	}
	step.targets = step.graph.Graph.NewRetainedTargets()
	for output, value := range map[*tensor.Tensor]executor.DeviceValue{block.Key: history.keys, block.Value: history.values} {
		slot, ok := step.graph.Graph.OutputSlot(output)
		if !ok {
			return nil, errors.New("inference: hyper-connection draft history output is not compiled")
		}
		if err := step.targets.SetSlot(slot, value); err != nil {
			return nil, err
		}
	}
	// The token rows, the rotary positions, the attention's extent and the append's offset are per-round attributes,
	// so one capture serves every round of these rows over this history.
	step.attributes = step.graph.Graph.NewRuntimeAttributes()
	rowAttributes := embedding.Attrs.(tensor.GetRowsAttributes)
	rowAttributes.Rows = make([]uint32, rows)
	step.rows = &rowAttributes
	if err := step.attributes.Set(embedding, step.rows); err != nil {
		return nil, err
	}
	nodes, err := tensor.Topological(step.selection, block.Key, block.Value)
	if err != nil {
		return nil, err
	}
	for _, node := range nodes {
		var value tensor.Attributes
		switch attributes := node.Attrs.(type) {
		case tensor.RoPEAttributes:
			attributes.Positions = make([]uint32, rows)
			step.positions = append(step.positions, attributes.Positions)
			value = &attributes
		case tensor.RoPEMultiAttributes:
			for axis := range attributes.Positions {
				if len(attributes.Positions[axis]) != 0 {
					attributes.Positions[axis] = make([]uint32, rows)
					step.positions = append(step.positions, attributes.Positions[axis])
				}
			}
			value = &attributes
		case tensor.AttentionAttributes:
			step.attention = append(step.attention, &attributes)
			value = &attributes
		case tensor.CacheAppendAttributes:
			step.appends = append(step.appends, &attributes)
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

// draftHyperFromHistoryLocked drafts the token after the round's last cell: cells are the trunk's streams of
// consecutive positions from position, rows of streams, each with the token after it in tokens; they are written to the
// history at their place, and the draft attends over the history's cells before them and the round's own.
//   - a round whose cells do not follow the history's last starts it again at them
func (r *Runner) draftHyperFromHistoryLocked(
	ctx context.Context,
	streams reference.Value,
	tokens []tokenizer.TokenID,
	position uint32,
) (tokenizer.TokenID, error) {
	draft := r.hyperDraft
	rows := uint32(len(tokens))
	if draft == nil || draft.history == nil || rows == 0 || !r.hasPreloadedWeights() {
		return 0, errors.New("inference: hyper-connection draft is unavailable")
	}
	if uint64(len(streams.Data)) != uint64(rows)*r.hyperDraftWidth() {
		return 0, errors.New("inference: hyper-connection draft streams are incompatible")
	}
	for _, token := range tokens {
		if token < 0 || int(token) >= r.vocab.Len() {
			return 0, fmt.Errorf("inference: token ID %d is out of range", token)
		}
	}
	history := draft.history
	if history.cells == 0 || position < history.start || position > history.start+history.cells {
		history.start, history.cells = position, 0
	}
	at := position - history.start
	if at+rows > history.capacity {
		history.start, history.cells, at = position, 0, 0
		if rows > history.capacity {
			return 0, errors.New("inference: a hyper-connection draft round outgrows its history")
		}
	}
	step := draft.steps[rows]
	if step == nil {
		var err error
		if step, err = r.compileHyperDraftStep(rows); err != nil {
			return 0, err
		}
		draft.steps[rows] = step
	}
	for index, token := range tokens {
		step.rows.Rows[index] = uint32(token)
	}
	for _, positions := range step.positions {
		for index := range positions {
			positions[index] = position + uint32(index)
		}
	}
	for _, attention := range step.attention {
		attention.QueryStart, attention.KeyValueTokens = at, at+rows
	}
	for _, write := range step.appends {
		write.Offset = at
	}
	step.feeds[step.streams] = reference.Value{Shape: tensor.MustShape(r.hyperDraftWidth(), uint64(rows)), Data: streams.Data}
	retained, err := r.cuda.ExecuteRetainedCompiled(ctx, step.graph.Graph, step.feeds, step.graph.Inputs, step.targets, step.attributes)
	if err != nil {
		history.cells = 0
		return 0, err
	}
	selection, err := retained.CopyToHost(ctx, step.selection)
	if err = errors.Join(err, retained.Release(context.Background())); err != nil {
		history.cells = 0
		return 0, err
	}
	history.cells = at + rows
	// A head over a vocabulary selects its row, which names the token.
	vocabulary := r.vocab.Len()
	if len(draft.vocabulary) != 0 {
		vocabulary = len(draft.vocabulary)
	}
	if len(selection.Data) != 1 || selection.Data[0] < 0 || int(selection.Data[0]) >= vocabulary {
		return 0, errors.New("inference: hyper-connection draft selection is invalid")
	}
	if len(draft.vocabulary) != 0 {
		return tokenizer.TokenID(draft.vocabulary[int(selection.Data[0])]), nil
	}
	return tokenizer.TokenID(selection.Data[0]), nil
}

// advanceHyperDraftLocked drafts the token after tokenID from the session's streams, a round of one cell at the
// session's position over the history.
//   - the session's cell sits at the position of the streams it read, one before the token it embeds
//   - a round drafts one token, so no session follows it
func (r *Runner) advanceHyperDraftLocked(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *MTPSession,
) (tokenizer.TokenID, *MTPSession, error) {
	if r.hyperDraft == nil || session == nil {
		return 0, nil, errors.New("inference: hyper-connection draft is unavailable")
	}
	if !tensor.HasDimensions(session.PendingHidden.Shape, r.hyperDraftWidth(), tensor.SingletonExtent) {
		return 0, nil, errors.New("inference: hyper-connection draft streams are incompatible")
	}
	if r.hyperDraft.history == nil {
		if err := r.ensureHyperDraftHistory(ctx, hyperDraftHistoryCells); err != nil {
			return 0, nil, err
		}
	}
	drafted, err := r.draftHyperFromHistoryLocked(ctx, session.PendingHidden, []tokenizer.TokenID{tokenID}, session.Position)
	return drafted, nil, err
}
