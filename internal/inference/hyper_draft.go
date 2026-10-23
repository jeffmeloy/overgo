package inference

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math/bits"

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
	// steps are the compiled draft rounds by their rows, over the history's capacity, and writes the rounds that only
	// write their cells, a prompt's.
	steps, writes map[uint32]*hyperDraftStep
	// history is the draft layer's own keys and values of the request's cells, nil until the first draft.
	history *hyperDraftHistory
	// prompt is the request's prompt read into the history, nil where the request reads none into it.
	prompt *hyperDraftPrompt
}

// hyperDraftPrompt is a prompt read's place in the draft's history: from is the first position the window keeps, and
// last the streams of the last row read, at position, whose cell waits for the token after it (pending).
//   - streams is the card memory a read writes its rows' streams into for the history's rounds, held while the prompt
//     is read: beside the outputs a read's cache keeps, an 8,192-token read's 335 MB went with the cache
type hyperDraftPrompt struct {
	from, position uint32
	last           reference.Value
	pending        bool
	streams        *executor.DeviceBuffer
}

// promptStreamsLocked answers card memory for a prompt read's streams of shape: the prompt's buffer, grown to fit.
func (r *Runner) promptStreamsLocked(ctx context.Context, shape tensor.Shape) (executor.DeviceValue, error) {
	prompt := r.hyperDraft.prompt
	if prompt.streams != nil {
		if value, err := prompt.streams.Value(shape); err == nil {
			return value, nil
		}
	}
	bytes, err := shape.Bytes(dtype.F32)
	if err = errors.Join(err, prompt.releaseStreams(ctx)); err != nil {
		return executor.DeviceValue{}, err
	}
	if prompt.streams, err = r.cuda.AllocateDeviceBuffer(ctx, bytes); err != nil {
		return executor.DeviceValue{}, err
	}
	return prompt.streams.Value(shape)
}

// releaseStreams frees the prompt's streams buffer, once the prompt is read.
func (p *hyperDraftPrompt) releaseStreams(ctx context.Context) error {
	if p == nil || p.streams == nil {
		return nil
	}
	err := p.streams.Release(ctx)
	p.streams = nil
	return err
}

// promptRead is the prompt the request read into the history, with its last row waiting for the token after it; nil
// where it read none.
func (d *hyperDraft) promptRead() *hyperDraftPrompt {
	if d == nil || d.prompt == nil || !d.prompt.pending {
		return nil
	}
	return d.prompt
}

// hyperDraftWindowCells is the most cells of a prompt the draft attends over, its last. Strata's drafter attends over
// the last 32,768 (--mtp-window); the draft's attention here reads every cell in one pass, and on Flash-Next a round
// took 1.27 ms over 16 cells, 1.46 over 4,096 and 2.26 over 16,384, so a window past an agent turn's newest document
// costs the decode more than its drafts return.
const hyperDraftWindowCells = 4096

// hyperDraftStreamBytes is a stream element's bytes on the card: the trunk keeps its streams as F32, which a read checks.
const hyperDraftStreamBytes = 4

// hyperDraftPromptRound is the most cells a round of a prompt writes at once; a read's cells go in rounds of powers of
// two up to it, so a few compiled rounds serve every read.
const hyperDraftPromptRound = 512

// hyperDraftHistory is the draft layer's keys and values of a request's cells, as Strata's drafter keeps its layer's:
// cell c pairs the trunk's streams at position start+c with the token after them, at rope position start+c, so a
// draft attends over every cell before its own, as the layer was trained, where a cell alone drafted 73% of Flash-Next's
// greedy answers' next-next tokens and the answer's cells 80%. A request that reads its prompt into it attends over the
// prompt's cells too, as Strata's drafter does: over two documents and two word lists its answers' first 24 cells drafted
// 20 to 23 where the answer's own cells drafted 18 to 19.
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
			draft.history, draft.steps, draft.writes = history, map[uint32]*hyperDraftStep{}, map[uint32]*hyperDraftStep{}
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
	r.hyperDraft.history, r.hyperDraft.steps, r.hyperDraft.writes = nil, nil, nil
	return err
}

// compileHyperDraftStep builds a round of rows cells over the history, which drafts where drafts is set and otherwise
// only writes its cells, as Strata's drafter writes a prompt's: their keys and values without the attention, the
// feed-forward or the head.
//   - the draft file stores the checkpoint's norms, scaled by one plus the weight, so the step adds the unit offset
//   - the cells append to the history in place at the round's offset, and attend over it and over the cells before
//     them in the round, the last cell's mixed streams drafting
func (r *Runner) compileHyperDraftStep(rows uint32, drafts bool) (*hyperDraftStep, error) {
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
	// A drafting round's streams come from the host; a writing round's lie on the card, where a prompt read left them.
	if drafts {
		step.streams = host("hyper_draft.streams", tensor.MustShape(r.hyperDraftWidth(), uint64(rows)), 0)
	} else {
		step.streams = builder.Input("hyper_draft.streams", dtype.F32, tensor.MustShape(r.hyperDraftWidth(), uint64(rows)))
	}
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
	outputs := []*tensor.Tensor{block.Key, block.Value}
	if drafts {
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
		outputs = append([]*tensor.Tensor{step.selection}, outputs...)
	}
	if err := errors.Join(builder.Err(), bindErr); err != nil {
		return nil, err
	}
	if step.graph, err = executor.CompileIndexed(outputs...); err != nil {
		return nil, err
	}
	// The draft's token comes down with its run; the history's keys and values are written in place.
	if drafts {
		if err := step.graph.Graph.ReadBack(step.selection); err != nil {
			return nil, err
		}
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
	nodes, err := tensor.Topological(outputs...)
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
func (r *Runner) draftHyperFromHistoryLocked(
	ctx context.Context,
	streams reference.Value,
	tokens []tokenizer.TokenID,
	position uint32,
) (tokenizer.TokenID, error) {
	retained, step, err := r.runHyperDraftRoundLocked(ctx, streams, 0, tokens, position, true)
	if err != nil {
		return 0, err
	}
	draft := r.hyperDraft
	selection, err := retained.CopyToHost(ctx, step.selection)
	if err = errors.Join(err, retained.Release(context.Background())); err != nil {
		draft.history.cells = 0
		return 0, err
	}
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

// runHyperDraftRoundLocked writes a round's cells to the history at their place and answers its run, which drafts
// where drafts is set: cells are the trunk's streams of consecutive positions from position, rows of streams, each with
// the token after it in tokens; a drafting round reads them from streams, a writing one from the card at device.
//   - a round whose cells do not follow the history's last starts it again at them
//   - a writing round's streams change place each run, so it runs uncaptured
func (r *Runner) runHyperDraftRoundLocked(
	ctx context.Context,
	streams reference.Value,
	device driver.DevicePtr,
	tokens []tokenizer.TokenID,
	position uint32,
	drafts bool,
) (*executor.RetainedOutputs, *hyperDraftStep, error) {
	draft := r.hyperDraft
	rows := uint32(len(tokens))
	if draft == nil || draft.history == nil || rows == 0 || !r.hasPreloadedWeights() {
		return nil, nil, errors.New("inference: hyper-connection draft is unavailable")
	}
	if drafts && uint64(len(streams.Data)) != uint64(rows)*r.hyperDraftWidth() || !drafts && device == 0 {
		return nil, nil, errors.New("inference: hyper-connection draft streams are incompatible")
	}
	for _, token := range tokens {
		if token < 0 || int(token) >= r.vocab.Len() {
			return nil, nil, fmt.Errorf("inference: token ID %d is out of range", token)
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
			return nil, nil, errors.New("inference: a hyper-connection draft round outgrows its history")
		}
	}
	steps := draft.steps
	if !drafts {
		steps = draft.writes
	}
	step := steps[rows]
	if step == nil {
		var err error
		if step, err = r.compileHyperDraftStep(rows, drafts); err != nil {
			return nil, nil, err
		}
		steps[rows] = step
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
	execute := r.cuda.ExecuteRetainedCompiled
	if drafts {
		step.feeds[step.streams] = reference.Value{Shape: tensor.MustShape(r.hyperDraftWidth(), uint64(rows)), Data: streams.Data}
	} else if err := step.graph.Inputs.Set(step.streams, device); err != nil {
		return nil, nil, err
	} else {
		execute = r.cuda.ExecuteRetainedCompiledOnce
	}
	retained, err := execute(ctx, step.graph.Graph, step.feeds, step.graph.Inputs, step.targets, step.attributes)
	if err != nil {
		history.cells = 0
		return nil, nil, err
	}
	history.cells = at + rows
	return retained, step, nil
}

// beginHyperDraftPromptLocked readies the draft's history for a request that reads its prompt into it: empty, with
// room for the window of the prompt's tokens after cached and the output, and the read keeping its rows' streams from
// the window's first position.
func (r *Runner) beginHyperDraftPromptLocked(ctx context.Context, prompt, cached, output int) error {
	from := max(cached, prompt-hyperDraftWindowCells)
	cells := prompt - from + min(output, int(r.spec.ContextLength)) + int(q8InputSpanColumnsWindow) + 1
	if err := r.ensureHyperDraftHistory(ctx, uint32(cells)); err != nil {
		return err
	}
	r.hyperDraft.prompt = &hyperDraftPrompt{from: uint32(from)}
	return nil
}

// recordHyperDraftPromptLocked writes into the history the cells a prompt read's rows make, the first row at position
// first, from the read's streams on the card: the last row read before them with this read's first token (a drafting
// round of one cell, its draft unused), then each row but the last with the token after it, those before the window
// left out; the last row comes down to wait for the token after it.
func (r *Runner) recordHyperDraftPromptLocked(ctx context.Context, streams executor.DeviceValue, kind dtype.Type, tokens []tokenizer.TokenID, first uint32) error {
	prompt, width, rows := r.hyperDraft.prompt, r.hyperDraftWidth(), uint64(len(tokens))
	if elements, err := streams.Shape.Elements(); err != nil || kind != dtype.F32 || rows == 0 || streams.Pointer == 0 || elements != rows*width {
		return errors.New("inference: a prompt read's draft streams are incompatible")
	}
	if prompt.pending && prompt.position+1 == first && prompt.position >= prompt.from {
		if _, err := r.draftHyperFromHistoryLocked(ctx, prompt.last, tokens[:1], prompt.position); err != nil {
			return err
		}
	}
	rowBytes := width * hyperDraftStreamBytes
	cells, position, row := tokens[1:], first, uint64(0)
	if skip := min(uint32(len(cells)), prompt.from-min(prompt.from, position)); skip != 0 {
		cells, position, row = cells[skip:], position+skip, uint64(skip)
	}
	for len(cells) != 0 {
		round := min(hyperDraftPromptRound, 1<<(bits.Len(uint(len(cells)))-1))
		retained, _, err := r.runHyperDraftRoundLocked(ctx, reference.Value{}, streams.Pointer+driver.DevicePtr(row*rowBytes), cells[:round], position, false)
		if err != nil {
			return err
		}
		if err := retained.Release(context.Background()); err != nil {
			return err
		}
		cells, position, row = cells[round:], position+uint32(round), row+uint64(round)
	}
	last, err := r.cuda.ReadDeviceValues(ctx, []executor.DeviceValue{{Pointer: streams.Pointer + driver.DevicePtr((rows-1)*rowBytes), Shape: tensor.MustShape(width, 1)}})
	if err != nil {
		return err
	}
	prompt.last = reference.Value{Shape: tensor.MustShape(width, 1), Data: last[0]}
	prompt.position, prompt.pending = first+uint32(rows-1), true
	return nil
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
