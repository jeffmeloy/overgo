package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Eagle3Session: shifted feature plus draft KV.
type Eagle3Session struct {
	Cache          *KVCache
	PendingFeature reference.Value
	Position       uint32
}

// Eagle3StepResult: logits, pre-norm feature, and cache.
type Eagle3StepResult struct {
	Logits      reference.Value
	NextFeature reference.Value
	Cache       *KVCache
}

// FuseEagle3Features: projects three target-layer inputs.
func (r *Runner) FuseEagle3Features(ctx context.Context, features reference.Value) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.spec.Architecture != "eagle3" || r.weights.FeatureProjection == nil {
		return reference.Value{}, errors.New("inference: Eagle3 feature encoder is unavailable")
	}
	builder := r.newGraphBuilder()
	input := builder.Input("eagle3.features", dtype.F32, features.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: features}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var projection *tensor.Tensor
	var err error
	if r.hasPreloadedWeights() {
		var pointer driver.DevicePtr
		projection, pointer, err = r.deviceInput(builder, *r.weights.FeatureProjection)
		if err == nil {
			deviceFeeds[projection] = pointer
		}
	} else {
		var value reference.Value
		value, err = model.LoadHostTensor(ctx, r.file, *r.weights.FeatureProjection)
		if err == nil {
			projection = builder.Input("fc.weight", dtype.F32, value.Shape)
			hostFeeds[projection] = value
		}
	}
	if err != nil {
		return reference.Value{}, err
	}
	output, err := model.BuildEagle3FeatureEncoder(builder, input, projection, r.spec)
	if err != nil {
		return reference.Value{}, err
	}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, []*tensor.Tensor{output}, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, []*tensor.Tensor{output}, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

// NewEagle3Session: full-prefix shifted-cache construction.
func (r *Runner) NewEagle3Session(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*Eagle3Session, error) {
	if r == nil || target == nil || r == target || r.path == target.path || len(tokenIDs) == 0 {
		return nil, errors.New("inference: Eagle3 and target inputs are invalid")
	}
	if r.spec.Architecture != "eagle3" || target.spec.EmbeddingLength != r.spec.TargetHiddenSize {
		return nil, errors.New("inference: Eagle3 target model is incompatible")
	}
	features, err := target.ExtractLayerInputs(ctx, tokenIDs, r.spec.TargetLayers)
	if err != nil {
		return nil, err
	}
	fused, err := r.FuseEagle3Features(ctx, features)
	if err != nil {
		return nil, err
	}
	width := int(fused.Shape.Dims[0])
	pending := reference.Value{
		Shape: tensor.MustShape(uint64(width), 1),
		Data:  append([]float32(nil), fused.Data[(len(tokenIDs)-1)*width:]...),
	}
	session := &Eagle3Session{PendingFeature: pending, Position: uint32(len(tokenIDs) - 1)}
	for index := 0; index+1 < len(tokenIDs); index++ {
		feature := reference.Value{
			Shape: tensor.MustShape(uint64(width), 1),
			Data:  append([]float32(nil), fused.Data[index*width:(index+1)*width]...),
		}
		step, stepErr := r.stepEagle3(ctx, target, tokenIDs[index+1], feature, uint32(index), session.Cache)
		if stepErr != nil {
			return nil, stepErr
		}
		session.Cache = step.Cache
	}
	return session, nil
}

// AdvanceEagle3: one autoregressive draft step.
func (r *Runner) AdvanceEagle3(
	ctx context.Context,
	target *Runner,
	tokenID tokenizer.TokenID,
	session *Eagle3Session,
) (reference.Value, *Eagle3Session, error) {
	if session == nil || session.PendingFeature.Shape.Rank != 2 {
		return reference.Value{}, nil, errors.New("inference: Eagle3 session is invalid")
	}
	step, err := r.stepEagle3(ctx, target, tokenID, session.PendingFeature, session.Position, session.Cache)
	if err != nil {
		return reference.Value{}, nil, err
	}
	next := &Eagle3Session{Cache: step.Cache, PendingFeature: step.NextFeature, Position: session.Position + 1}
	return step.Logits, next, nil
}

func (r *Runner) stepEagle3(
	ctx context.Context,
	target *Runner,
	tokenID tokenizer.TokenID,
	feature reference.Value,
	position uint32,
	cache *KVCache,
) (Eagle3StepResult, error) {
	if r == nil || target == nil || r == target || r.path == target.path {
		return Eagle3StepResult{}, errors.New("inference: Eagle3 and target runners are invalid")
	}
	first, second := r, target
	if first.path > second.path {
		first, second = second, first
	}
	first.mu.Lock()
	second.mu.Lock()
	defer second.mu.Unlock()
	defer first.mu.Unlock()
	if r.closed || target.closed || r.spec.Architecture != "eagle3" {
		return Eagle3StepResult{}, errors.New("inference: Eagle3 runner is unavailable")
	}
	if tokenID < 0 || int(tokenID) >= target.vocab.Len() {
		return Eagle3StepResult{}, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	if feature.Shape.Rank != 2 || feature.Shape.Dims[0] != uint64(r.spec.EmbeddingLength) || feature.Shape.Dims[1] != 1 {
		return Eagle3StepResult{}, errors.New("inference: Eagle3 feature shape is incompatible")
	}
	if cache != nil && (len(cache.Layers) != 1 || cache.Position != position) {
		return Eagle3StepResult{}, errors.New("inference: Eagle3 cache position is incompatible")
	}
	rows := []uint32{uint32(tokenID)}
	var tokenEmbedding reference.Value
	var err error
	if r.weights.TokenEmbedding.Name != "" {
		tokenEmbedding, err = r.loadRows(ctx, r.weights.TokenEmbedding, rows)
	} else {
		tokenEmbedding, err = target.loadEmbeddings(ctx, rows)
	}
	if err != nil {
		return Eagle3StepResult{}, err
	}
	builder := r.newGraphBuilder()
	tokenInput := builder.Input("eagle3.token", dtype.F32, tokenEmbedding.Shape)
	featureInput := builder.Input("eagle3.feature", dtype.F32, feature.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{tokenInput: tokenEmbedding, featureInput: feature}
	graphWeights, deviceFeeds, err := r.eagle3LayerInputs(ctx, builder, hostFeeds)
	if err != nil {
		return Eagle3StepResult{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if cache != nil {
		pastKey = builder.Input("eagle3.past_key", dtype.F32, cache.Layers[0].Key.Shape)
		pastValue = builder.Input("eagle3.past_value", dtype.F32, cache.Layers[0].Value.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = cache.Layers[0].Key, cache.Layers[0].Value
	}
	block, err := model.BuildEagle3BlockCached(
		builder, tokenInput, featureInput, r.spec, graphWeights, []uint32{position}, pastKey, pastValue,
	)
	if err != nil {
		return Eagle3StepResult{}, err
	}
	norm, normPointer, err := r.deviceOrHostTensor(ctx, builder, r.weights.OutputNorm, hostFeeds)
	if err != nil {
		return Eagle3StepResult{}, err
	}
	if normPointer != 0 {
		deviceFeeds[norm] = normPointer
	}
	normalized := builder.WeightedRMSNorm(block.Output, norm, r.spec.RMSNormEpsilon)
	outputInfo := target.weights.TokenEmbedding
	outputOwner := target
	if target.weights.Output != nil {
		outputInfo = *target.weights.Output
	}
	if r.weights.Output != nil {
		outputInfo, outputOwner = *r.weights.Output, r
	}
	output, outputPointer, err := outputOwner.deviceOrHostTensor(ctx, builder, outputInfo, hostFeeds)
	if err != nil {
		return Eagle3StepResult{}, err
	}
	if outputPointer != 0 && outputOwner == r {
		deviceFeeds[output] = outputPointer
	} else if outputPointer != 0 {
		value, loadErr := model.LoadHostTensor(ctx, target.file, outputInfo)
		if loadErr != nil {
			return Eagle3StepResult{}, loadErr
		}
		output = builder.Input(outputInfo.Name+".shared", dtype.F32, value.Shape)
		hostFeeds[output] = value
	}
	logits := builder.MulMat(output, normalized)
	outputs := []*tensor.Tensor{block.Output, block.Key, block.Value, logits}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, outputs, hostFeeds)
	}
	if err != nil {
		return Eagle3StepResult{}, err
	}
	logitValue := results[logits]
	if r.weights.DraftToTarget != nil {
		logitValue, err = r.remapEagle3Logits(ctx, logitValue)
		if err != nil {
			return Eagle3StepResult{}, err
		}
	}
	nextCache := &KVCache{
		Layers: []LayerCache{{Key: results[block.Key], Value: results[block.Value]}},
		Tokens: uint32(results[block.Key].Shape.Dims[2]), Position: position + 1,
	}
	return Eagle3StepResult{Logits: logitValue, NextFeature: results[block.Output], Cache: nextCache}, nil
}

func (r *Runner) eagle3LayerInputs(
	ctx context.Context,
	builder *tensor.Builder,
	hostFeeds map[*tensor.Tensor]reference.Value,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	if r.hasPreloadedWeights() {
		return r.layerDeviceInputs(builder, r.weights.Layers[0])
	}
	hostLayer, err := model.LoadHostLayer(ctx, r.file, r.weights.Layers[0])
	if err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	graph, feeds, err := hostLayer.GraphInputs(builder, "blk.0.")
	if err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	for node, value := range feeds {
		hostFeeds[node] = value
	}
	return graph, map[*tensor.Tensor]driver.DevicePtr{}, nil
}

func (r *Runner) deviceOrHostTensor(
	ctx context.Context,
	builder *tensor.Builder,
	info gguf.TensorInfo,
	hostFeeds map[*tensor.Tensor]reference.Value,
) (*tensor.Tensor, driver.DevicePtr, error) {
	if r.hasPreloadedWeights() {
		return r.deviceInput(builder, info)
	}
	value, err := model.LoadHostTensor(ctx, r.file, info)
	if err != nil {
		return nil, 0, err
	}
	item := builder.Input(info.Name, dtype.F32, value.Shape)
	hostFeeds[item] = value
	return item, 0, nil
}

func (r *Runner) remapEagle3Logits(ctx context.Context, logits reference.Value) (reference.Value, error) {
	mapping, err := model.LoadHostTensor(ctx, r.file, *r.weights.DraftToTarget)
	if err != nil {
		return reference.Value{}, err
	}
	result := reference.Value{Shape: tensor.MustShape(uint64(r.spec.VocabularySize), 1), Data: make([]float32, r.spec.VocabularySize)}
	for index := range result.Data {
		result.Data[index] = float32(math.Inf(-1))
	}
	for draft, rawTarget := range mapping.Data {
		target := int(rawTarget)
		if target < 0 || target >= len(result.Data) || draft >= len(logits.Data) {
			return reference.Value{}, errors.New("inference: Eagle3 vocabulary map is invalid")
		}
		result.Data[target] = logits.Data[draft]
	}
	return result, nil
}
