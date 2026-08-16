package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
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
	file, path, spec, weights, program, evidenceTier, err := loaded.Take()
	if err != nil {
		return nil, fmt.Errorf("inference: take model program: %w", err)
	}
	promptCacheCapacity := options.PromptCacheEntries
	if promptCacheCapacity == 0 {
		promptCacheCapacity = 1
	}
	cachePageTokens := resolveCachePageTokens(options.CachePageTokens)
	// Residency is compiled into recipe identity; runtime flags may only confirm it.
	residency, err := bindResidency(program.Residency)
	if err != nil {
		return nil, errors.Join(err, file.Close())
	}
	var cuda *executor.Executor
	var worker *device.Worker
	var deviceWeights *model.DeviceF32Weights
	var rawWeights *model.DeviceWeights
	var decodeWeights *model.DeviceBF16Weights
	fail := func(openErr error) (*Runner, error) {
		return nil, errors.Join(
			openErr,
			closeAcceleratorResources(decodeWeights, rawWeights, deviceWeights, cuda, worker),
			file.Close(),
		)
	}
	vocab, err := tokenizer.Load(file)
	if err != nil {
		return fail(err)
	}
	loraAdapters := make([]loadedLoRA, len(options.LoRAAdapters))
	for index, configured := range options.LoRAAdapters {
		if math.IsNaN(float64(configured.Scale)) || math.IsInf(float64(configured.Scale), 0) {
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
	if residency.deviceF32 || residency.deviceNative {
		worker, err = device.New(options.DeviceOrdinal)
		if err != nil {
			return fail(err)
		}
		cuda, err = executor.NewWithWorker(worker)
		if err != nil {
			return fail(err)
		}
		// Resolve tied or dedicated output projection before residency selection.
		outputProjection := weights.TokenEmbedding
		if program.Model.Terminal().OutputHead == model.OutputHeadDedicated && weights.Output != nil {
			outputProjection = *weights.Output
		}
		if err = loadResidentWeights(
			ctx, file, weights, outputProjection, loraAdapters, residency, worker,
			&deviceWeights, &rawWeights, &decodeWeights,
		); err != nil {
			if !residency.hostRecovery || !driver.IsOutOfMemory(err) {
				return fail(err)
			}
			// Measured fit answered no: release the partial stores and keep
			// the executor; decode streams weights as before.
			err = errors.Join(decodeWeights.Close(), rawWeights.Close(), deviceWeights.Close())
			if err != nil {
				return fail(err)
			}
			decodeWeights, rawWeights, deviceWeights = nil, nil, nil
		}
	} else if !residency.hostReference {
		cuda, err = executor.New(options.DeviceOrdinal)
		if err != nil {
			return fail(err)
		}
	}
	return &Runner{preparedModel: preparedModel{
		file: file, path: path, spec: spec, program: program, evidenceTier: evidenceTier,
		weights: weights, vocab: vocab,
		cuda: cuda, worker: worker, deviceWeights: deviceWeights, rawWeights: rawWeights, decodeWeights: decodeWeights,
		hostWeights:         hostWeights,
		outputBias:          outputBias,
		promptCacheCapacity: promptCacheCapacity, cachePageTokens: cachePageTokens,
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

// loadResidentWeights: uploads the resident weight stores for a preload open;
// partially loaded stores stay owned by the caller on failure.
func loadResidentWeights(
	ctx context.Context,
	file *gguf.File,
	weights model.Weights,
	outputProjection gguf.TensorInfo,
	loraAdapters []loadedLoRA,
	residency residencyBinding,
	worker *device.Worker,
	deviceWeights **model.DeviceF32Weights,
	rawWeights **model.DeviceWeights,
	decodeWeights **model.DeviceBF16Weights,
) error {
	var err error
	*deviceWeights, err = model.NewDeviceF32Weights(worker)
	if err != nil {
		return err
	}
	selected := selectedModelTensors(file, weights)
	f32Tensors := selected
	if residency.deviceNative {
		adaptedTensors := make(map[string]struct{})
		for _, loaded := range loraAdapters {
			for name := range loaded.adapter.Weights {
				adaptedTensors[name] = struct{}{}
			}
		}
		f32Required := f32RequiredModelTensors(weights)
		embeddingTensors := getRowsSourceTensors(weights)
		f32Tensors = make([]gguf.TensorInfo, 0, len(selected))
		var quantized []gguf.TensorInfo
		for _, info := range selected {
			_, adapted := adaptedTensors[info.Name]
			_, requiresF32 := f32Required[info.Name]
			_, isEmbedding := embeddingTensors[info.Name]
			// native half-precision residency: 2D F16/BF16 matmul weights stream
			// into the raw store at 2 bytes (no F32 blowup). F8E4M3 joins the same
			// raw native route at ~1 byte/element + a per-row F32 scale (the
			// combined [rows*inner e4m3 | rows*4 scale] payload the GGUF carries and
			// the fp8 matmul kernel consumes). Decode reads the native dtype
			// directly; prefill upconverts to F32 for exact SGEMM. BF16 embeddings
			// use native get_rows; other embeddings and adapted/f32-required
			// tensors stay F32.
			if !adapted && !requiresF32 && (!isEmbedding || info.Type == dtype.BF16) && info.Dimensions == 2 &&
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
		*rawWeights, err = model.NewDeviceWeights(worker)
		if err != nil {
			return err
		}
		if err = (*rawWeights).Load(ctx, file, quantized); err != nil {
			return err
		}
		// native 2D BF16 matmul weights now live in the raw store (above),
		// serving both decode (native kernel) and prefill (upconvert + SGEMM)
		// from a single 2-byte copy. The decode catalog covers only tensors that
		// stayed F32 but drive decode MulMat.
		var decodeTensors []gguf.TensorInfo
		_, outputIsRaw := (*rawWeights).Lookup(outputProjection.Name)
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
			*decodeWeights, err = model.NewDeviceBF16Weights(worker)
			if err == nil {
				err = (*decodeWeights).Load(ctx, file, decodeTensors)
			}
			if err != nil {
				return err
			}
		}
	}
	return (*deviceWeights).Load(ctx, file, f32Tensors)
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
		resources.close(),
	)
}

func (s *runnerState) detachPromptCaches() []*cachedPrompt {
	caches := s.promptCaches
	s.promptCaches = nil
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
	decodeWeights *model.DeviceBF16Weights,
	rawWeights *model.DeviceWeights,
	deviceWeights *model.DeviceF32Weights,
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
	decodeWeights *model.DeviceBF16Weights
	rawWeights    *model.DeviceWeights
	deviceWeights *model.DeviceF32Weights
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
