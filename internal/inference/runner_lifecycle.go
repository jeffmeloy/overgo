package inference

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modeldevice"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tokenizer"
)

// OpenWithProgram binds hardware to the recipe-owned serving program.
func OpenWithProgram(ctx context.Context, loaded *modelrecipe.LoadedProgram, options OpenOptions) (*Runner, error) {
	if ctx == nil {
		return nil, errors.New("inference: model-open context is nil")
	}
	if loaded == nil {
		return nil, errors.New("inference: resolved model program is nil")
	}
	if options.PromptCacheEntries < 0 {
		return nil, errors.New("inference: prompt cache entry count is negative")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	runtimePolicy, err := loaded.RuntimePolicy()
	if err != nil {
		return nil, err
	}
	file, path, spec, weights, program, evidenceTier, claims, err := loaded.Take()
	if err != nil {
		return nil, fmt.Errorf("inference: take model program: %w", err)
	}
	promptCacheCapacity := options.PromptCacheEntries
	if promptCacheCapacity == 0 {
		promptCacheCapacity = runtimePolicy.Serving.PromptCacheEntries
	}
	// Residency is compiled into recipe identity; runtime flags may only confirm it.
	residency, err := bindResidency(program.Residency)
	if err != nil {
		return nil, errors.Join(err, file.Close())
	}
	var cuda *executor.Executor
	var worker *device.Worker
	var deviceWeights *modeldevice.DeviceConvertedWeights
	var rawWeights *modeldevice.DeviceWeights
	var decodeWeights *modeldevice.DeviceConvertedWeights
	fail := func(openErr error) (*Runner, error) {
		return nil, errors.Join(
			openErr,
			closeAcceleratorResources(decodeWeights, rawWeights, deviceWeights, cuda, worker),
			file.Close(),
		)
	}
	// Claimed shard identities are proved by the pass that uploads the weights.
	// A residency that uploads none proves them by hashing before it serves.
	if len(claims) != 0 && !residency.deviceF32 && !residency.deviceNative {
		if err := modelartifact.VerifyClaims(file, claims); err != nil {
			return fail(err)
		}
	}
	// The parts of the open name themselves to a request that keeps a ledger of
	// them (processmeasure): the tokenizer here, and below the device worker
	// with its context, the executor, and the one pass that reads, proves and
	// uploads the weights.
	tokenized := processmeasure.Stage(ctx, "tokenizer")
	vocab, err := tokenizer.Load(file)
	tokenized()
	if err != nil {
		return fail(err)
	}
	loraAdapters := make([]loadedLoRA, len(options.LoRAAdapters))
	for index, configured := range options.LoRAAdapters {
		if !checked.Finite32(configured.Scale) {
			return fail(fmt.Errorf("inference: LoRA adapter %d scale is invalid", index))
		}
		adapter, loadErr := model.LoadLoRA(ctx, configured.Path, file, spec)
		if loadErr != nil {
			return fail(fmt.Errorf("inference: load LoRA adapter %d: %w", index, loadErr))
		}
		loraAdapters[index] = loadedLoRA{
			adapter: adapter, scale: configured.Scale, signature: loRAStaticSignature(adapter),
		}
	}
	var outputBias []float32
	if weights.OutputBias != nil {
		value, loadErr := model.LoadHostTensor(
			ctx,
			file,
			*weights.OutputBias,
		)
		if loadErr != nil {
			return fail(loadErr)
		}
		outputBias = slices.Clone(value.Data)
	}
	var hostWeights *model.HostTensorStore
	if residency.hostCache {
		hostWeights = model.NewHostTensorStore()
	}
	recovered := false
	if residency.deviceF32 || residency.deviceNative {
		started := processmeasure.Stage(ctx, "device worker")
		worker, err = device.New(options.DeviceOrdinal)
		started()
		if err != nil {
			return fail(err)
		}
		built := processmeasure.Stage(ctx, "executor")
		cuda, err = executor.NewWithWorker(worker)
		built()
		if err != nil {
			return fail(err)
		}
		// Resolve tied or dedicated output projection before residency selection.
		outputProjection := weights.TokenEmbedding
		if program.Model.Terminal().OutputHead == model.OutputHeadDedicated && weights.Output != nil {
			outputProjection = *weights.Output
		}
		uploaded := processmeasure.Stage(ctx, "weights")
		recovered, err = loadWithHostRecovery(residency.hostRecovery, func() error {
			return loadResidentWeights(
				ctx, file, weights, outputProjection, loraAdapters, residency, worker, claims,
				&deviceWeights, &rawWeights, &decodeWeights,
			)
		}, func() error {
			return releasePartialStores(&decodeWeights, &rawWeights, &deviceWeights)
		})
		uploaded()
		if err != nil {
			return fail(err)
		}
		// The pass that fell back to streaming stopped before it hashed the whole file.
		if recovered && len(claims) != 0 {
			if err := modelartifact.VerifyClaims(file, claims); err != nil {
				return fail(err)
			}
		}
	} else if !residency.hostReference {
		cuda, err = executor.New(options.DeviceOrdinal)
		if err != nil {
			return fail(err)
		}
	}
	realized, err := realizedResidencyForOpen(program.Residency, deviceWeights != nil, recovered, rawWeights.HasExpertResidency())
	if err != nil {
		return fail(err)
	}
	if rawWeights.HasExpertResidency() {
		if err := cuda.SetExpertBanks(rawWeights); err != nil {
			return fail(err)
		}
	}
	return &Runner{preparedModel: preparedModel{
		file: file, path: path, spec: spec, program: program, runtimePolicy: runtimePolicy, evidenceTier: evidenceTier,
		realizedResidency: realized,
		weights:           weights, vocab: vocab,
		cuda: cuda, worker: worker, deviceWeights: deviceWeights, rawWeights: rawWeights, decodeWeights: decodeWeights,
		hostWeights:         hostWeights,
		outputBias:          outputBias,
		outputExclusions:    vocab.NonTextGenerationRanges(),
		promptCacheCapacity: promptCacheCapacity,
	}, runnerState: runnerState{loraAdapters: loraAdapters}}, nil
}

type residencyBinding struct {
	deviceF32, deviceNative, decodeBF16    bool
	hostCache, hostRecovery, hostReference bool
}

func bindResidency(policy recipe.ResidencyPolicy) (residencyBinding, error) {
	var binding residencyBinding
	switch policy {
	case recipe.ResidencyStream:
	case recipe.ResidencyHostCache:
		binding.hostCache = true
	case recipe.ResidencyDeviceF32:
		binding.deviceF32 = true
	case recipe.ResidencyDeviceNative:
		binding.deviceNative = true
	case recipe.ResidencyDeviceNativeBF16:
		binding.deviceNative, binding.decodeBF16 = true, true
	case recipe.ResidencyHybridNative:
		binding.deviceNative, binding.hostRecovery = true, true
	case recipe.ResidencyHostReference:
		binding.hostReference = true
	default:
		return residencyBinding{}, errors.New("inference: compiled residency policy is unavailable")
	}
	return binding, nil
}

// loadWithHostRecovery keeps an OOM fallback and partial-store release on one
// testable path. A release failure cannot become a successful streamed open.
func loadWithHostRecovery(recoverOOM bool, load, release func() error) (bool, error) {
	err := load()
	if err == nil {
		return false, nil
	}
	if !recoverOOM || !driver.IsOutOfMemory(err) {
		return false, err
	}
	if releaseErr := release(); releaseErr != nil {
		return false, errors.Join(err, releaseErr)
	}
	return true, nil
}

// releasePartialStores frees what a failed hybrid preload left, before the
// retained executor streams weights on later requests, and forgets what it
// freed. A model with no decode tensors never made a decode store.
func releasePartialStores(
	decode **modeldevice.DeviceConvertedWeights,
	raw **modeldevice.DeviceWeights,
	f32 **modeldevice.DeviceConvertedWeights,
) error {
	err := closeAcceleratorResources(*decode, *raw, *f32, nil, nil)
	if err == nil {
		*decode, *raw, *f32 = nil, nil, nil
	}
	return err
}

func realizedResidencyForOpen(policy recipe.ResidencyPolicy, resident, recovered, experts bool) (recipe.RealizedResidency, error) {
	if experts {
		if policy == recipe.ResidencyHybridNative && resident && !recovered {
			return recipe.RealizedExpertOffloaded, nil
		}
		return "", errors.New("inference: inconsistent expert residency")
	}
	if recovered {
		if policy == recipe.ResidencyHybridNative && !resident {
			return recipe.RealizedOOMStreamed, nil
		}
		return "", errors.New("inference: inconsistent recovered residency")
	}
	switch policy {
	case recipe.ResidencyDeviceF32:
		if resident {
			return recipe.RealizedDeviceF32, nil
		}
	case recipe.ResidencyDeviceNative:
		if resident {
			return recipe.RealizedDeviceNative, nil
		}
	case recipe.ResidencyDeviceNativeBF16:
		if resident {
			return recipe.RealizedDeviceNativeBF16, nil
		}
	case recipe.ResidencyHybridNative:
		if resident {
			return recipe.RealizedDeviceNative, nil
		}
	case recipe.ResidencyStream:
		if !resident {
			return recipe.RealizedStream, nil
		}
	case recipe.ResidencyHostCache:
		if !resident {
			return recipe.RealizedHostCache, nil
		}
	case recipe.ResidencyHostReference:
		if !resident {
			return recipe.RealizedHostReference, nil
		}
	}
	return "", errors.New("inference: inconsistent realized residency")
}

// loadResidentWeights: uploads the resident weight stores for a preload open in
// one pass over the file; the stores stay owned by the caller on failure.
// claims, when present, are the shard identities the pass must find by hashing
// what it uploads, and the stores hold nothing unless they match.
func loadResidentWeights(
	ctx context.Context,
	file *gguf.File,
	weights model.Weights,
	outputProjection gguf.TensorInfo,
	loraAdapters []loadedLoRA,
	residency residencyBinding,
	worker *device.Worker,
	claims []artifact.ID,
	deviceWeights **modeldevice.DeviceConvertedWeights,
	rawWeights **modeldevice.DeviceWeights,
	decodeWeights **modeldevice.DeviceConvertedWeights,
) error {
	var err error
	*deviceWeights, err = modeldevice.NewDeviceConvertedWeights(worker, dtype.F32)
	if err != nil {
		return err
	}
	loader := modeldevice.NewLoader(file)
	selected := selectedModelTensors(file, weights)
	f32Tensors := selected
	var fileBackedExperts []gguf.TensorInfo
	if residency.deviceNative {
		adaptedTensors := make(map[string]struct{})
		for _, loaded := range loraAdapters {
			for name := range loaded.adapter.Weights {
				adaptedTensors[name] = struct{}{}
			}
		}
		f32Required := f32RequiredModelTensors(weights)
		embeddingTensors := getRowsSourceTensors(weights)
		expertBanks := nativeExpertBankTensors(weights, adaptedTensors)
		f32Tensors = make([]gguf.TensorInfo, 0, len(selected))
		var quantized []gguf.TensorInfo
		for _, info := range selected {
			_, adapted := adaptedTensors[info.Name]
			_, requiresF32 := f32Required[info.Name]
			_, isEmbedding := embeddingTensors[info.Name]
			_, isExpertBank := expertBanks[info.Name]
			// native half-precision residency: 2D F16/BF16 matmul weights stream
			// into the raw store at 2 bytes (no F32 blowup). F8E4M3 joins the same
			// raw native route at ~1 byte/element + a per-row F32 scale (the
			// combined [rows*inner e4m3 | rows*4 scale] payload the GGUF carries and
			// the fp8 matmul kernel consumes). Decode reads the native dtype
			// directly; prefill upconverts to F32 for exact SGEMM. BF16 embeddings
			// use native get_rows; other embeddings and adapted/f32-required
			// tensors stay F32. BF16 routed-expert banks take the raw route too:
			// their kernels read them as stored, in decode and prefill alike.
			if isExpertBank || !adapted && !requiresF32 && (!isEmbedding || info.Type == dtype.BF16) && info.Dimensions == 2 &&
				(info.Type == dtype.F16 || info.Type == dtype.BF16 || info.Type == dtype.F8E4M3) {
				quantized = append(quantized, info)
				continue
			}
			if !adapted && !requiresF32 && info.Type.IsQuantized() {
				quantized = append(quantized, info)
			} else {
				f32Tensors = append(f32Tensors, info)
			}
		}
		*rawWeights, err = modeldevice.NewDeviceWeights(worker)
		if err != nil {
			return err
		}
		if residency.hostRecovery {
			var available uint64
			if err := worker.Do(ctx, func(state *device.State) error {
				free, _, err := state.Driver.MemInfo()
				available = free
				return err
			}); err != nil {
				return err
			}
			quantized, fileBackedExperts, err = splitExpertResidency(quantized, f32Tensors, expertBanks, available)
			if err != nil {
				return err
			}
		}
		loader.Raw(*rawWeights, quantized)
		if len(fileBackedExperts) != 0 {
			// The pass that reads, and with claims hashes, the file fills the experts.
			loader.Experts(*rawWeights, fileBackedExperts, 0)
		}
		// native 2D BF16 matmul weights now live in the raw store (above),
		// serving both decode (native kernel) and prefill (upconvert + SGEMM)
		// from a single 2-byte copy. The decode catalog covers only tensors that
		// stayed F32 but drive decode MulMat.
		var decodeTensors []gguf.TensorInfo
		outputIsRaw := slices.ContainsFunc(quantized, func(info gguf.TensorInfo) bool {
			return info.Name == outputProjection.Name
		})
		if _, isEmbedding := embeddingTensors[outputProjection.Name]; isEmbedding && !outputIsRaw &&
			outputProjection.Type == dtype.BF16 && outputProjection.Dimensions == 2 {
			decodeTensors = append(decodeTensors, outputProjection)
		}
		if residency.decodeBF16 {
			for _, info := range quantized {
				_, adapted := adaptedTensors[info.Name]
				if !adapted && info.Type == dtype.Q8_0 && info.Dimensions >= 2 {
					decodeTensors = append(decodeTensors, info)
				}
			}
		}
		if len(decodeTensors) > 0 {
			*decodeWeights, err = modeldevice.NewDeviceConvertedWeights(worker, dtype.BF16)
			if err != nil {
				return err
			}
			loader.Converted(*decodeWeights, decodeTensors)
		}
	}
	loader.Converted(*deviceWeights, f32Tensors)
	return loader.Load(ctx, claims)
}

func (r *Runner) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	caches := r.detachPromptCaches()
	resources := r.preparedModel.detachResources()
	r.mu.Unlock()
	_, cacheErr := releasePromptCaches(context.Background(), caches)
	return errors.Join(
		cacheErr,
		r.releaseSpanSlots(context.Background()),
		resources.close(),
	)
}

func (s *runnerState) detachPromptCaches() []*cachedPrompt {
	caches := s.promptCaches
	s.promptCaches = nil
	s.decodeSession, s.decodeSessions = nil, nil
	return caches
}

func releasePromptCaches(ctx context.Context, caches []*cachedPrompt) ([]*cachedPrompt, error) {
	var errs []error
	failed := caches[:0]
	for _, promptCache := range caches {
		if promptCache.Device != nil {
			if err := promptCache.Device.Release(ctx); err != nil {
				errs = append(errs, err)
				failed = append(failed, promptCache)
				continue
			}
		}
		promptCache.Device = nil
	}
	return failed, errors.Join(errs...)
}

func closeAcceleratorResources(
	decodeWeights *modeldevice.DeviceConvertedWeights,
	rawWeights *modeldevice.DeviceWeights,
	deviceWeights *modeldevice.DeviceConvertedWeights,
	cuda *executor.Executor,
	worker *device.Worker,
) error {
	var errs []error
	if decodeWeights != nil {
		errs = append(errs, decodeWeights.Close())
	}
	if rawWeights != nil {
		errs = append(errs, rawWeights.Close())
	}
	if deviceWeights != nil {
		errs = append(errs, deviceWeights.Close())
	}
	if cuda != nil {
		errs = append(errs, cuda.Close())
	}
	if worker != nil {
		errs = append(errs, worker.Close())
	}
	return errors.Join(errs...)
}

type preparedResources struct {
	file          *gguf.File
	hostWeights   *model.HostTensorStore
	decodeWeights *modeldevice.DeviceConvertedWeights
	rawWeights    *modeldevice.DeviceWeights
	deviceWeights *modeldevice.DeviceConvertedWeights
	cuda          *executor.Executor
	worker        *device.Worker
}

func (m *preparedModel) detachResources() preparedResources {
	resources := preparedResources{
		file: m.file, hostWeights: m.hostWeights,
		decodeWeights: m.decodeWeights, rawWeights: m.rawWeights,
		deviceWeights: m.deviceWeights, cuda: m.cuda, worker: m.worker,
	}
	m.file, m.hostWeights = nil, nil
	m.decodeWeights, m.rawWeights, m.deviceWeights = nil, nil, nil
	m.cuda, m.worker = nil, nil
	return resources
}

func (m *preparedResources) close() error {
	var errs []error
	if m.hostWeights != nil {
		m.hostWeights.Release()
		m.hostWeights = nil
	}
	errs = append(errs, closeAcceleratorResources(
		m.decodeWeights, m.rawWeights, m.deviceWeights, m.cuda, m.worker,
	))
	if m.file != nil {
		errs = append(errs, m.file.Close())
	}
	return errors.Join(errs...)
}
