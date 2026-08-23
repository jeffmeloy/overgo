//go:build windows

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/checked"
	"overgo/internal/composition"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/representation"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

const (
	compositeTargetAlias = "research/composite-generation/cuda/target"
	compositeSourceAlias = "research/composite-generation/cuda/source"
	compositeModelEnv    = "OVERGO_COMPOSITE_MODEL"
)

type laneConfig struct {
	TargetDirectory          string                                           `json:"target_directory"`
	SourceWeights            string                                           `json:"source_weights"`
	ConditionContext         string                                           `json:"condition_context"`
	UnconditionContext       string                                           `json:"uncondition_context"`
	Frames                   int                                              `json:"frames"`
	Width                    int                                              `json:"width"`
	Height                   int                                              `json:"height"`
	Steps                    int                                              `json:"steps"`
	Shift                    float64                                          `json:"shift"`
	GuideScale               float64                                          `json:"guide_scale"`
	Seed                     uint64                                           `json:"seed"`
	NoiseOffset              uint64                                           `json:"noise_offset"`
	ThreadsPerMultiprocessor int                                              `json:"threads_per_multiprocessor"`
	BridgeMaxAbsDelta        float64                                          `json:"bridge_max_abs_delta"`
	PromotionPolicy          composition.RepresentationBridgePromotionPolicy  `json:"promotion_policy"`
	PromotionTrials          []composition.RepresentationBridgePromotionTrial `json:"promotion_trials"`
}

type laneCatalog struct {
	Source artifact.ID
	Target artifact.ID
	Path   string
}

type laneAuthority struct {
	Plan           composition.CompositionExecutionPlan
	BaselineRecipe artifact.ID
	SourceContract representation.Contract
	TargetContract representation.Contract
	Bridge         composition.BridgeDefinition
}

type generationMeasurement struct {
	Content artifact.Content
	Result  latentvideo.GenerationResult
	Started time.Time
	Elapsed time.Duration
}

var laneTensor = tensor.MustShape
var laneCount = checked.Rows[float32]

var laneF32Bytes, _ = dtype.F32.ScalarBytes()

func run() error {
	ctx := context.Background()
	var config laneConfig
	fixtureDirectory := filepath.Join("fixtures", "wan")
	if err := jsonfile.Decode(filepath.Join(fixtureDirectory, "composite_generation_lane.json"), &config); err != nil {
		return err
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	store, err := repodb.Open(roots.Store)
	if err != nil {
		return err
	}
	defer store.Close()
	catalog, err := resolveLaneCatalog(ctx, store, roots, config)
	if err != nil {
		return runrecord.LaneError(runrecord.LaneUnavailable, err.Error())
	}
	profile, err := latentvideo.ResolveProfile(catalog.Path)
	if err != nil {
		return err
	}
	denoiser, err := latentvideo.LoadDenoiserConfig(catalog.Path, profile.Policy)
	if err != nil {
		return err
	}
	contextElements, ok := checked.MulInt(denoiser.TextLen, denoiser.Dim)
	if !ok {
		return errors.New("composite generation lane: context extent overflows")
	}
	conditionBytes, condition, err := loadLaneContext(filepath.Join(fixtureDirectory, config.ConditionContext), contextElements)
	if err != nil {
		return err
	}
	unconditionBytes, uncondition, err := loadLaneContext(filepath.Join(fixtureDirectory, config.UnconditionContext), contextElements)
	if err != nil {
		return err
	}
	authority, inputs, err := ensureLaneComposition(
		ctx, store, catalog, config, denoiser.TextLen, denoiser.Dim, conditionBytes, unconditionBytes,
	)
	if err != nil {
		return err
	}
	bridged, bridgeFact, err := executeLaneBridge(ctx, authority, condition, config.BridgeMaxAbsDelta)
	if err != nil {
		return err
	}
	deviceInfo, noise, err := laneDevicePlan(config, authority.Plan, profile, catalog.Path)
	if err != nil {
		return err
	}
	generator, err := latentvideo.NewGenerator(latentvideo.GeneratorConfig{
		ModelDirectory: catalog.Path, Policy: profile.Policy, LatentStats: profile.LatentStats,
		Frames: config.Frames, Width: config.Width, Height: config.Height,
		Precision:     latentvideo.DenoiserPrecision{MatmulWeights: dtype.BF16, RoundAttentionStorage: true},
		DeviceOrdinal: device.DefaultOrdinal(),
	})
	if err != nil {
		return err
	}
	defer generator.Close()
	baseline, err := executeWanGeneration(ctx, generator, profile, config, condition, uncondition, noise)
	if err != nil {
		return err
	}
	composed, err := executeWanGeneration(ctx, generator, profile, config, bridged, uncondition, noise)
	if err != nil {
		return err
	}
	if baseline.Content.Descriptor.ID != composed.Content.Descriptor.ID || !slices.Equal(baseline.Content.Data, composed.Content.Data) {
		return errors.New("composite generation lane: promoted identity bridge changed model-native output")
	}
	evidence, err := publishLaneEvidence(
		ctx, store, config, catalog, authority, inputs, deviceInfo, bridgeFact, baseline, composed,
	)
	if err != nil {
		return err
	}
	fmt.Printf(
		"COMPOSITE GENERATION CUDA GREEN target=%s source=%s recipe=%s output=%s evidence=%s baseline=%.3fs composed=%.3fs\n",
		catalog.Target, catalog.Source, authority.Plan.CompositionRecipe, baseline.Content.Descriptor.ID,
		evidence.ID, baseline.Elapsed.Seconds(), composed.Elapsed.Seconds(),
	)
	return nil
}

func resolveLaneCatalog(
	ctx context.Context,
	store *repodb.Store,
	roots dataroot.Roots,
	config laneConfig,
) (laneCatalog, error) {
	target, targetFound, err := store.ResolveAlias(ctx, compositeTargetAlias)
	if err != nil {
		return laneCatalog{}, err
	}
	source, sourceFound, err := store.ResolveAlias(ctx, compositeSourceAlias)
	if err != nil {
		return laneCatalog{}, err
	}
	if targetFound && sourceFound {
		path, err := artifact.AvailablePath(ctx, store, target, artifact.LocationDirectory)
		return laneCatalog{Source: source, Target: target, Path: path}, err
	}
	path := strings.TrimSpace(os.Getenv(compositeModelEnv))
	if path == "" {
		path = filepath.Join(roots.Models, config.TargetDirectory)
	}
	if !latentvideo.IsWan(path) {
		return laneCatalog{}, fmt.Errorf("set %s to the real Wan model directory; %s is unavailable", compositeModelEnv, path)
	}
	targetInventory, err := modelartifact.FromFiles(path, []modelartifact.FileSpec{
		{Path: filepath.Join(path, "config.json"), Name: "config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(path, "diffusion_pytorch_model.safetensors"), Name: "denoiser/weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(path, "Wan2.1_VAE.pth"), Name: "vae/weights", Role: artifact.ComponentWeights},
	})
	if err != nil {
		return laneCatalog{}, err
	}
	sourceInventory, err := modelartifact.FromFiles(path, []modelartifact.FileSpec{{
		Path: filepath.Join(path, config.SourceWeights), Name: "encoder/weights", Role: artifact.ComponentWeights,
	}})
	if err != nil {
		return laneCatalog{}, err
	}
	for alias, inventory := range map[string]modelartifact.Inventory{
		compositeTargetAlias: targetInventory,
		compositeSourceAlias: sourceInventory,
	} {
		batch, err := inventory.Batch("composite-generation/catalog/" + inventory.Manifest.ID.String())
		if err != nil {
			return laneCatalog{}, err
		}
		batch.Aliases = []artifact.AliasBinding{{Name: alias, Target: inventory.Manifest.ID}}
		if _, err := store.Commit(ctx, batch); err != nil {
			return laneCatalog{}, err
		}
	}
	return laneCatalog{Source: sourceInventory.Manifest.ID, Target: targetInventory.Manifest.ID, Path: path}, nil
}

func ensureLaneComposition(
	ctx context.Context,
	store *repodb.Store,
	catalog laneCatalog,
	config laneConfig,
	textLength, channels int,
	conditionBytes, unconditionBytes []byte,
) (laneAuthority, []artifact.ID, error) {
	_, found, err := composition.ActiveComposition(ctx, store, catalog.Source, catalog.Target, recipe.TaskGeneration)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	baselineContent, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindRecipe, "overgo/composite-generation-cuda-baseline/v1"),
		struct {
			Target artifact.ID `json:"target"`
			Task   recipe.Task `json:"task"`
		}{Target: catalog.Target, Task: recipe.TaskGeneration},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	inputContract := artifact.DocumentContract{
		Kind: artifact.KindOutput, MediaType: "application/octet-stream", Schema: "overgo/wan-context-f32le/v1",
	}
	conditionContent, err := inputContract.ContentBytes(conditionBytes)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	unconditionContent, err := inputContract.ContentBytes(unconditionBytes)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	inputs := []artifact.ID{conditionContent.Descriptor.ID, unconditionContent.Descriptor.ID}
	if found {
		plan, err := composition.CompileCompositionExecutionPlan(ctx, store, catalog.Source, catalog.Target, recipe.TaskGeneration)
		if err != nil {
			return laneAuthority{}, nil, err
		}
		sourceContract, err := representation.LoadContract(ctx, store, plan.SourceContract)
		if err != nil {
			return laneAuthority{}, nil, err
		}
		targetContract, err := representation.LoadContract(ctx, store, plan.TargetContract)
		if err != nil {
			return laneAuthority{}, nil, err
		}
		bridge, err := composition.LoadBridgeDefinition(ctx, store, plan.BridgeDefinitions[tensor.FirstOffset])
		return laneAuthority{
			Plan: plan, BaselineRecipe: baselineContent.Descriptor.ID,
			SourceContract: sourceContract, TargetContract: targetContract, Bridge: bridge,
		}, inputs, err
	}
	sourceDefinition, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindModelDefinition, "overgo/umt5-context-source/v1"),
		struct {
			Model artifact.ID `json:"model"`
		}{Model: catalog.Source},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	targetDefinition, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindModelDefinition, "overgo/wan-context-target/v1"),
		struct {
			Model artifact.ID `json:"model"`
		}{Model: catalog.Target},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	sequence := representation.SequenceContract{
		Axis: representation.AxisSequence, Mask: representation.MaskPrefix, Padding: representation.PaddingSuffix,
		Position: representation.PositionSequential, PositionAxes: []representation.AxisKind{representation.AxisSequence},
	}
	contractTensor := representation.TensorContract{DataType: dtype.F32, Axes: []representation.Axis{
		{Kind: representation.AxisChannel, Bounds: representation.AxisBounds{Extent: uint64(channels)}},
		{Kind: representation.AxisSequence, Bounds: representation.AxisBounds{Extent: uint64(textLength)}},
	}}
	normalization := representation.NormalizationContract{
		Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative,
	}
	sourceContract, err := representation.NewContract(representation.Contract{
		Producer: representation.Producer{
			Model: catalog.Source, Definition: sourceDefinition.Descriptor.ID, Tap: representation.TapEncoderOutput,
		},
		Modality: representation.ModalityText, Tensor: contractTensor, Sequence: sequence, Normalization: normalization,
	})
	if err != nil {
		return laneAuthority{}, nil, err
	}
	layer := uint32(tensor.FirstOffset)
	targetContract, err := representation.NewContract(representation.Contract{
		Producer: representation.Producer{
			Model: catalog.Target, Definition: targetDefinition.Descriptor.ID,
			Tap: representation.TapAttentionInput, Layer: &layer,
		},
		Modality: representation.ModalityText, Tensor: contractTensor, Sequence: sequence, Normalization: normalization,
	})
	if err != nil {
		return laneAuthority{}, nil, err
	}
	bridgeWeights, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindAdapter, "overgo/identity-linear-bridge/v1"),
		struct {
			Channels int `json:"channels"`
		}{Channels: channels},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	bridgeInventory, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindTensorInventory, "overgo/identity-linear-bridge-inventory/v1"),
		struct {
			Input  int `json:"input"`
			Output int `json:"output"`
		}{Input: channels, Output: channels},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	trainingPolicy, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindProfile, "overgo/frozen-identity-bridge-training/v1"),
		struct {
			Frozen bool `json:"frozen"`
		}{Frozen: true},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	dataset, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindDataset, "overgo/wan-heldout-context-dataset/v1"),
		struct {
			Inputs []artifact.ID `json:"inputs"`
		}{Inputs: inputs},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	heldOut, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindDatasetShard, "overgo/wan-heldout-context-split/v1"),
		struct {
			Dataset artifact.ID `json:"dataset"`
			Purpose string      `json:"purpose"`
		}{Dataset: dataset.Descriptor.ID, Purpose: "held-out"},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	regression, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindDatasetShard, "overgo/wan-regression-context-split/v1"),
		struct {
			Dataset artifact.ID `json:"dataset"`
			Purpose string      `json:"purpose"`
		}{Dataset: dataset.Descriptor.ID, Purpose: "regression"},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	evaluator, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindEvidence, "overgo/composite-generation-cuda-evaluator/v1"),
		struct {
			ExactParity bool `json:"exact_parity"`
		}{ExactParity: true},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	prerequisites := []artifact.Content{
		baselineContent, conditionContent, unconditionContent, sourceDefinition, targetDefinition,
		bridgeWeights, bridgeInventory, trainingPolicy, dataset, heldOut, regression, evaluator,
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "composite-generation/cuda/prerequisites", Contents: prerequisites,
	}); err != nil {
		return laneAuthority{}, nil, err
	}
	bridge, err := composition.NewBridgeDefinition(composition.BridgeDefinition{
		SourceModel: catalog.Source, TargetModel: catalog.Target,
		Graph: bridgegraph.Definition{
			Source: sourceContract.ID, Target: targetContract.ID, Operator: bridgegraph.OperatorLinear,
		},
		Weights: bridgeWeights.Descriptor.ID, WeightInventory: bridgeInventory.Descriptor.ID,
	}, sourceContract, targetContract)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	executionRecipe, err := (modelrecipe.RepresentationBridgeCompiler{
		SourcePlacement: recipe.PlacementDevice, TargetPlacement: recipe.PlacementDevice,
		SourceResidency: recipe.ResidencyDeviceF32, TargetResidency: recipe.ResidencyDeviceF32,
		SourceSession: recipe.SessionCapacity, TargetSession: recipe.SessionRequest,
	}).Definition(catalog.Source, catalog.Target, sourceContract.ID, targetContract.ID, bridgeWeights.Descriptor.ID)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	promotionPolicy, err := composition.NewRepresentationBridgePromotionPolicy(config.PromotionPolicy)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	promotion, err := (composition.RepresentationBridgePromoter{}).Evaluate(
		promotionPolicy,
		composition.RepresentationBridgePromotion{
			Bridge: bridgeWeights.Descriptor.ID, SourceModel: catalog.Source, TargetModel: catalog.Target,
			SourceContract: sourceContract.ID, TargetContract: targetContract.ID,
			HeldOutSplit: heldOut.Descriptor.ID, RegressionSet: regression.Descriptor.ID,
			Evaluator: evaluator.Descriptor.ID, Trials: slices.Clone(config.PromotionTrials),
		},
	)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	compositionRecipe, err := composition.NewCompositionRecipe(composition.CompositionRecipe{
		SourceModel: catalog.Source, TargetModel: catalog.Target, Task: recipe.TaskGeneration,
		SourceContract: sourceContract.ID, TargetContract: targetContract.ID,
		BridgeDefinition: bridge.ID, BridgeWeights: bridgeWeights.Descriptor.ID,
		ExecutionRecipe: executionRecipe.ID, TrainingPolicy: trainingPolicy.Descriptor.ID,
		PromotionPolicy: promotionPolicy.ID, Promotion: promotion.ID,
	})
	if err != nil {
		return laneAuthority{}, nil, err
	}
	authority := composition.CompositionAuthority{
		SourceContract: sourceContract, TargetContract: targetContract, Bridge: bridge,
		Execution: executionRecipe, PromotionPolicy: promotionPolicy, Promotion: promotion, Recipe: compositionRecipe,
	}
	batch, err := authority.Batch("composite-generation/cuda/authority")
	if err != nil {
		return laneAuthority{}, nil, err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return laneAuthority{}, nil, err
	}
	activation, err := authority.Recipe.ActivationBatch(ctx, store, "composite-generation/cuda/activate", nil)
	if err != nil {
		return laneAuthority{}, nil, err
	}
	if _, err := store.Commit(ctx, activation); err != nil {
		return laneAuthority{}, nil, err
	}
	plan, err := composition.CompileCompositionExecutionPlan(ctx, store, catalog.Source, catalog.Target, recipe.TaskGeneration)
	return laneAuthority{
		Plan: plan, BaselineRecipe: baselineContent.Descriptor.ID,
		SourceContract: sourceContract, TargetContract: targetContract, Bridge: bridge,
	}, inputs, err
}

func executeLaneBridge(
	ctx context.Context,
	authority laneAuthority,
	condition []float32,
	maximumDelta float64,
) ([]float32, artifact.Content, error) {
	sourceContent, err := authority.SourceContract.Content()
	if err != nil {
		return nil, artifact.Content{}, err
	}
	targetContent, err := authority.TargetContract.Content()
	if err != nil {
		return nil, artifact.Content{}, err
	}
	program, err := (bridgegraph.Compiler{}).Compile(authority.Bridge.Graph, sourceContent.Data, targetContent.Data)
	if err != nil {
		return nil, artifact.Content{}, err
	}
	channels := int(authority.Plan.Capture.Channels)
	rows, err := laneCount(condition, channels)
	if err != nil {
		return nil, artifact.Content{}, err
	}
	weightElements, ok := checked.MulInt(channels, channels)
	if !ok {
		return nil, artifact.Content{}, errors.New("composite generation lane: bridge weight extent overflows")
	}
	identity := make([]float32, weightElements)
	for index := range channels {
		identity[index*channels+index] = float32(tensor.SingletonExtent)
	}
	builder := tensor.NewBuilder()
	inputNode := builder.Input("source-context", dtype.F32, laneTensor(uint64(channels), uint64(rows)))
	weightNode := builder.Input("identity-bridge", dtype.F32, laneTensor(uint64(channels), uint64(channels)))
	outputNode, err := program.Build(builder, inputNode, bridgegraph.Weights{First: weightNode})
	if err != nil {
		return nil, artifact.Content{}, err
	}
	compiled, err := executor.Compile(outputNode)
	if err != nil {
		return nil, artifact.Content{}, err
	}
	cuda, err := executor.New(device.DefaultOrdinal())
	if err != nil {
		return nil, artifact.Content{}, err
	}
	defer cuda.Close()
	feeds := map[*tensor.Tensor]reference.Value{
		inputNode:  {Shape: inputNode.Shape, Data: condition},
		weightNode: {Shape: weightNode.Shape, Data: identity},
	}
	var output []float32
	var metrics executor.ExecutionMetrics
	for {
		retained, executeErr := cuda.ExecuteRetainedCompiled(ctx, compiled, feeds, nil, nil, nil)
		if executeErr != nil {
			return nil, artifact.Content{}, executeErr
		}
		value, copyErr := retained.CopyToHost(ctx, outputNode)
		releaseErr := retained.Release(ctx)
		if copyErr != nil || releaseErr != nil {
			return nil, artifact.Content{}, errors.Join(copyErr, releaseErr)
		}
		output = slices.Clone(value.Data)
		metrics, err = cuda.Metrics(ctx)
		if err != nil {
			return nil, artifact.Content{}, err
		}
		if checked.Nonzero(metrics.GraphCacheHits) {
			break
		}
		if metrics.GraphCacheMisses > metrics.GraphCacheCapacity {
			return nil, artifact.Content{}, errors.New("composite generation lane: CUDA graph replay cache did not stabilize")
		}
	}
	var maxDelta float64
	for index := range condition {
		maxDelta = max(maxDelta, math.Abs(float64(output[index]-condition[index])))
	}
	if maxDelta > maximumDelta || !checked.Nonzero(metrics.GraphCacheHits) {
		return nil, artifact.Content{}, fmt.Errorf(
			"composite generation lane: CUDA bridge parity or replay evidence failed: max abs delta %g (limit %g), graph cache hits %d",
			maxDelta, maximumDelta, metrics.GraphCacheHits,
		)
	}
	fact, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindEvidence, "overgo/composite-generation-bridge-cuda/v1"),
		struct {
			Plan        artifact.ID               `json:"plan"`
			Bridge      artifact.ID               `json:"bridge"`
			MaxAbsDelta float64                   `json:"max_abs_delta"`
			Metrics     executor.ExecutionMetrics `json:"metrics"`
		}{Plan: authority.Plan.ID, Bridge: authority.Bridge.Weights, MaxAbsDelta: maxDelta, Metrics: metrics},
	)
	return output, fact, err
}

func laneDevicePlan(
	config laneConfig,
	plan composition.CompositionExecutionPlan,
	profile latentvideo.Profile,
	modelPath string,
) (driver.DeviceInfo, sampling.CounterNoisePlan, error) {
	library, err := driver.Open()
	if err != nil {
		return driver.DeviceInfo{}, sampling.CounterNoisePlan{}, err
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		return driver.DeviceInfo{}, sampling.CounterNoisePlan{}, err
	}
	info, err := library.DeviceInfo(device.DefaultOrdinal())
	if err != nil {
		return driver.DeviceInfo{}, sampling.CounterNoisePlan{}, err
	}
	denoiser, err := latentvideo.LoadDenoiserConfig(modelPath, profile.Policy)
	if err != nil {
		return driver.DeviceInfo{}, sampling.CounterNoisePlan{}, err
	}
	geometry, err := denoiser.CompileLatentGeometry(config.Frames, config.Width, config.Height)
	if err != nil {
		return driver.DeviceInfo{}, sampling.CounterNoisePlan{}, err
	}
	noise, _, err := sampling.CompileCounterNoisePlan(
		config.Seed, config.NoiseOffset, geometry.Elements(), info.MultiprocessorCount, config.ThreadsPerMultiprocessor,
	)
	if err != nil || !plan.ID.Valid() {
		return driver.DeviceInfo{}, sampling.CounterNoisePlan{}, errors.Join(err, errors.New("composite generation lane: execution plan is absent"))
	}
	return info, noise, nil
}

func executeWanGeneration(
	ctx context.Context,
	generator *latentvideo.Generator,
	profile latentvideo.Profile,
	config laneConfig,
	condition, uncondition []float32,
	noise sampling.CounterNoisePlan,
) (generationMeasurement, error) {
	encoder, err := latentvideo.NewGIFEncoder(profile.SampleFPS, latentvideo.SignedUnitPixels)
	if err != nil {
		return generationMeasurement{}, err
	}
	started := time.Now()
	result, err := generator.Generate(ctx, latentvideo.GenerateRequest{
		Steps: config.Steps, Shift: config.Shift, GuideScale: config.GuideScale,
		CondContext: condition, UncondContext: uncondition, Noise: noise, Sink: encoder.Add,
	})
	elapsed := time.Since(started)
	if err != nil {
		return generationMeasurement{}, err
	}
	video, err := encoder.Finish()
	if err != nil {
		return generationMeasurement{}, err
	}
	content, err := latentvideo.GIFContent(video)
	return generationMeasurement{Content: content, Result: result, Started: started, Elapsed: elapsed}, err
}

func publishLaneEvidence(
	ctx context.Context,
	store *repodb.Store,
	config laneConfig,
	catalog laneCatalog,
	authority laneAuthority,
	inputs []artifact.ID,
	deviceInfo driver.DeviceInfo,
	bridgeFact artifact.Content,
	baseline, composed generationMeasurement,
) (composition.CompositeGenerationCUDAEvidence, error) {
	deviceFact, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindEvidence, "overgo/cuda-device-profile/v1"), deviceInfo,
	)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	codeCommit, err := currentCodeCommit()
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	baselineRun, err := laneRun(authority.BaselineRecipe, inputs, baseline.Content.Descriptor.ID, codeCommit, deviceFact.Descriptor.ID, baseline)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	composedRun, err := laneRun(authority.Plan.CompositionRecipe, inputs, composed.Content.Descriptor.ID, codeCommit, deviceFact.Descriptor.ID, composed)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	baselineObservation, err := laneObservation(
		catalog.Target, authority.BaselineRecipe, deviceFact.Descriptor.ID, baselineRun, config, baseline,
	)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	composedObservation, err := laneObservation(
		catalog.Target, authority.Plan.CompositionRecipe, deviceFact.Descriptor.ID, composedRun, config, composed,
	)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	evidence, err := (composition.CompositeGenerationCUDAAuthority{}).New(composition.CompositeGenerationCUDAEvidence{
		SourceModel: catalog.Source, TargetModel: catalog.Target,
		CompositionRecipe: authority.Plan.CompositionRecipe, TargetBaselineRecipe: authority.BaselineRecipe,
		ExecutionPlan: authority.Plan.ID, Promotion: authority.Plan.Promotion, Bridge: authority.Bridge.Weights,
		HeldOutInputs: inputs, BaselineOutput: baseline.Content.Descriptor.ID, ComposedOutput: composed.Content.Descriptor.ID,
		BaselineRun: baselineRun.ID, ComposedRun: composedRun.ID,
		BaselineObservation: baselineObservation.ID, ComposedObservation: composedObservation.ID,
		Device: deviceFact.Descriptor.ID, BridgeExecution: bridgeFact.Descriptor.ID, ExactOutputParity: true,
	})
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	planContent, err := authority.Plan.Content()
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	evidenceContent, err := evidence.Content()
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	surfacePromotion, err := (composition.CompositeGenerationPromotionAuthority{}).Promote(authority.Plan, evidence)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	promotionContent, err := surfacePromotion.Content()
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	contents := []artifact.Content{
		baseline.Content, planContent, deviceFact, bridgeFact, evidenceContent, promotionContent,
	}
	for _, document := range []interface {
		Content() (artifact.Content, error)
	}{
		baselineRun, composedRun, baselineObservation, composedObservation,
	} {
		content, contentErr := document.Content()
		if contentErr != nil {
			return composition.CompositeGenerationCUDAEvidence{}, contentErr
		}
		contents = append(contents, content)
	}
	lineage := slices.Clone(authority.Plan.Lineage())
	lineage = append(lineage, baselineRun.Lineage()...)
	lineage = append(lineage, composedRun.Lineage()...)
	lineage = append(lineage, baselineObservation.Lineage()...)
	lineage = append(lineage, composedObservation.Lineage()...)
	lineage = append(lineage, evidence.Lineage()...)
	lineage = append(lineage, surfacePromotion.Lineage()...)
	alias, err := composition.CompositeGenerationPromotionAlias(catalog.Source, catalog.Target, authority.Plan.Task)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	previous, found, err := artifact.ResolveAlias(ctx, store, alias)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	aliases := []artifact.AliasBinding{{Name: alias, Target: surfacePromotion.ID}}
	if found {
		aliases[tensor.FirstOffset].Previous = &previous
	}
	batch, err := artifact.NewDocumentBatch(
		"composite-generation/cuda/evidence/"+evidence.ID.String(), contents, lineage, aliases,
	)
	if err != nil {
		return composition.CompositeGenerationCUDAEvidence{}, err
	}
	_, err = artifact.CommitBatch(ctx, store, batch)
	return evidence, err
}

func laneRun(
	recipeID artifact.ID,
	inputs []artifact.ID,
	output artifact.ID,
	codeCommit string,
	environment artifact.ID,
	measurement generationMeasurement,
) (runrecord.Run, error) {
	return runrecord.NewBoundRun(
		recipeID, runrecord.OutcomeSucceeded, inputs, []artifact.ID{output}, "", codeCommit, environment,
		uint64(measurement.Elapsed.Nanoseconds()),
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: uint64(measurement.Elapsed.Nanoseconds())}},
	)
}

func laneObservation(
	modelID, recipeID, environment artifact.ID,
	run runrecord.Run,
	config laneConfig,
	measurement generationMeasurement,
) (runrecord.ServingObservation, error) {
	peak := max(measurement.Result.DenoiseMemory.PeakBytes, measurement.Result.DecodeMemory.PeakBytes)
	elapsed := uint64(measurement.Elapsed.Nanoseconds())
	inputElements := len(measurement.Result.Denoise.Latent)
	return runrecord.NewServingObservation(runrecord.ServingObservation{
		Model: modelID, Recipe: recipeID, Environment: environment, Run: run.ID,
		Task: recipe.TaskGeneration, Outcome: runrecord.OutcomeSucceeded,
		StartedUnixNS: measurement.Started.UnixNano(), MeasuredNS: elapsed,
		Usage: runrecord.ServingUsage{
			InputTokens: uint64(config.Frames), OutputTokens: uint64(measurement.Result.Decode.OutputFrames),
			InputBytes:  uint64(inputElements) * laneF32Bytes,
			OutputBytes: uint64(measurement.Content.Descriptor.Size),
		},
		Resources: runrecord.ServingResources{
			PeakDeviceBytes:   peak,
			HostToDeviceBytes: measurement.Result.Execution.HostToDeviceBytes,
			DeviceToHostBytes: measurement.Result.Execution.DeviceToHostBytes,
		},
		Phases: []runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: elapsed}},
		Hardware: []runrecord.ServingHardwareSample{
			{Stage: runrecord.ServingHardwareStart, DeviceCurrentBytes: peak, DevicePeakBytes: peak},
			{Stage: runrecord.ServingHardwareFinish, ElapsedNS: elapsed, DeviceCurrentBytes: peak, DevicePeakBytes: peak},
		},
	})
}

func loadLaneContext(path string, elements int) ([]byte, []float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	want, ok := checked.MulInt(elements, int(laneF32Bytes))
	if !ok || len(raw) != want {
		return nil, nil, fmt.Errorf("composite generation lane: context bytes=%d want=%d", len(raw), want)
	}
	values := make([]float32, elements)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*int(laneF32Bytes):]))
	}
	return raw, values, nil
}

func currentCodeCommit() (string, error) {
	output, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}
