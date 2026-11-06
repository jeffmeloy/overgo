package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelcli"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/vitencoder"
)

// compositionTask is the task an image composite's recipe is active under, and bridgeWeightTensor and bridgeBiasTensor
// the tensors its bridge's adapter file holds: the map's rows, a row for each of the text model's channels, and the bias.
const (
	compositionTask    = recipe.TaskGeneration
	bridgeWeightTensor = "bridge.weight"
	bridgeBiasTensor   = "bridge.bias"
)

// openCompositionSession evaluates a text model that sees images through its active composition: the image encoder the
// request names encodes each picture, the composition's bridge carries the encoding to the model's width, and the rows
// take the placeholder token the request names, through the projector's placeholder path.
func openCompositionSession(ctx context.Context, store *overgodb.Store, value manifest, request modelRequest) (evaluationSession, error) {
	if request.Placeholder == "" {
		return nil, fmt.Errorf("evaluate: %s sees images through a composition, and the request names no placeholder token", request.Path)
	}
	source, err := mediacapability.Catalog[recipe.TaskImageEmbedding].Resolve(ctx, request.Encoder)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no image encoder: %w", request.Encoder, err)
	}
	encoder, err := vitencoder.Load(request.Encoder)
	if err != nil {
		return nil, err
	}
	runner, err := modelcli.OpenRunner(ctx, value.Repository, request.Path, inference.OpenOptions{DeviceOrdinal: value.Device})
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (evaluationSession, error) { return nil, errors.Join(cause, runner.Close()) }
	resources := compositionResources{
		source: encoderSource{id: source.Inventory.Manifest.ID, encoder: encoder},
		target: runnerTarget{runner: runner, placeholder: request.Placeholder},
		store:  store,
	}
	composite, err := inference.OpenProductionComposition(ctx, store, source.Inventory.Manifest.ID, runner.ModelID(), compositionTask, resources)
	if err != nil {
		return fail(err)
	}
	description, err := runner.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil {
		return fail(err)
	}
	identity := description.Identity
	environment, err := cudaEnvironment(value.Device)
	if err != nil {
		return fail(err)
	}
	runtime := projectionRuntime{runner: runner, vision: composite, mediaFiles: &mediaFiles{root: request.MediaRoot}, ffmpeg: os.Getenv("OVERGO_FFMPEG")}
	runtime.identity = evaluation.MediaRuntime{Model: runner.ModelID(), Recipe: identity.Recipe, Projection: composite.Recipe()}
	campaign, err := evaluation.NewIsolatedCampaign(store, runtime, identity, environment, value.CodeCommit)
	if err != nil {
		return fail(err)
	}
	return &projectionSession{
		nativeSession: nativeSession{store: store, runner: runner, campaign: campaign, model: identity.Model, commit: value.CodeCommit},
		vision:        composite,
	}, nil
}

// compositionResources are a composite's live parts: the encoder, the text model and the store its bridge's adapter
// is read from.
type compositionResources struct {
	source encoderSource
	target runnerTarget
	store  *overgodb.Store
}

// Source is the composite's image encoder.
func (r compositionResources) Source(context.Context, modelrecipe.ComponentSession) (inference.RepresentationSource, error) {
	return r.source, nil
}

// Target is the composite's text model.
func (r compositionResources) Target(context.Context, modelrecipe.ComponentSession) (inference.RepresentationTarget, error) {
	return r.target, nil
}

// BridgeWeights reads the bridge's adapter from the store: a safetensors file of the map, the text model's channels by
// the encoder's, and the bias.
func (r compositionResources) BridgeWeights(
	ctx context.Context, _ modelrecipe.ComponentSession, _ artifact.ID, authority composition.BridgeWeightAuthority,
) (inference.RepresentationBridgeWeights, error) {
	_, stream, found, err := r.store.OpenContent(ctx, authority.Weights.ID)
	if err != nil || !found {
		return inference.RepresentationBridgeWeights{}, errors.Join(err, errors.New("evaluate: the bridge's adapter is not in the store"))
	}
	file, err := os.CreateTemp("", "overgo-bridge-*.safetensors")
	if err != nil {
		return inference.RepresentationBridgeWeights{}, err
	}
	defer os.Remove(file.Name())
	if _, err := io.Copy(file, stream); err != nil {
		return inference.RepresentationBridgeWeights{}, errors.Join(err, file.Close())
	}
	if err := file.Close(); err != nil {
		return inference.RepresentationBridgeWeights{}, err
	}
	adapter, err := safetensors.OpenFile(file.Name())
	if err != nil {
		return inference.RepresentationBridgeWeights{}, err
	}
	defer adapter.Close()
	weight, weightFound := adapter.Tensors[bridgeWeightTensor]
	bias, biasFound := adapter.Tensors[bridgeBiasTensor]
	if !weightFound || !biasFound || len(weight.Shape) != tensor.PairedExtent || len(bias.Shape) != tensor.SingletonExtent || bias.Shape[0] != weight.Shape[0] {
		return inference.RepresentationBridgeWeights{}, errors.New("evaluate: the bridge's adapter holds no map of the text model's channels by the encoder's and its bias")
	}
	first, err := adapterValue(weight, weight.Shape[1], weight.Shape[0])
	if err != nil {
		return inference.RepresentationBridgeWeights{}, err
	}
	firstBias, err := adapterValue(bias, bias.Shape[0])
	if err != nil {
		return inference.RepresentationBridgeWeights{}, err
	}
	return inference.RepresentationBridgeWeights{First: &first, FirstBias: &firstBias}, nil
}

// adapterValue reads one adapter tensor as a host value of the extents the bridge reads it under.
func adapterValue(stored safetensors.Tensor, extents ...uint64) (reference.Value, error) {
	data, err := safetensors.ReadF32(stored)
	if err != nil {
		return reference.Value{}, err
	}
	shape, err := tensor.NewShape(extents...)
	if err != nil {
		return reference.Value{}, err
	}
	return reference.NewValue(shape, data)
}

// encoderSource is a composite's image encoder: every token of a picture, the encoder's width by its tokens.
type encoderSource struct {
	id      artifact.ID
	encoder *vitencoder.Encoder
}

// ModelID is the encoder's identity.
func (s encoderSource) ModelID() artifact.ID { return s.id }

// EncodeImage is every token the encoder gives the picture, its width by its tokens.
func (s encoderSource) EncodeImage(_ context.Context, picture image.Image) (reference.Value, error) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, picture); err != nil {
		return reference.Value{}, err
	}
	tokens, err := s.encoder.Tokens(encoded.Bytes())
	if err != nil {
		return reference.Value{}, err
	}
	_, _, hidden := s.encoder.Shape()
	shape, err := tensor.NewShape(uint64(hidden), uint64(len(tokens)/hidden))
	if err != nil {
		return reference.Value{}, err
	}
	return reference.NewValue(shape, tokens)
}

// runnerTarget is a composite's text model and the placeholder token its rows take.
type runnerTarget struct {
	runner      *inference.Runner
	placeholder string
}

// ModelID is the text model's identity.
func (t runnerTarget) ModelID() artifact.ID { return t.runner.ModelID() }

// EmbeddingWidth is the text model's embedding width, the rows the bridge must give.
func (t runnerTarget) EmbeddingWidth() int { return int(t.runner.Spec().EmbeddingLength) }

// ImagePlaceholder is the token each of the composite's rows takes.
func (t runnerTarget) ImagePlaceholder() string { return t.placeholder }
