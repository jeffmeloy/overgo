package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tokenizer"
)

func Open(path string, deviceOrdinal int) (*Runner, error) {
	return OpenWithOptions(path, OpenOptions{DeviceOrdinal: deviceOrdinal})
}

func OpenWithOptions(path string, options OpenOptions) (*Runner, error) {
	if options.PreloadDeviceWeights && options.PreloadQuantizedWeights {
		return nil, errors.New("inference: F32 and native-quantized preload modes are mutually exclusive")
	}
	if options.PreloadBF16DecodeWeights && !options.PreloadQuantizedWeights {
		return nil, errors.New("inference: BF16 decode catalog requires native-quantized preload")
	}
	if options.CacheHostWeights && (options.PreloadDeviceWeights || options.PreloadQuantizedWeights) {
		return nil, errors.New("inference: host and device weight retention are mutually exclusive")
	}
	if options.PromptCacheEntries < 0 {
		return nil, errors.New("inference: prompt cache entry count is negative")
	}
	promptCacheCapacity := options.PromptCacheEntries
	if promptCacheCapacity == 0 {
		promptCacheCapacity = 1
	}
	cachePageTokens := resolveCachePageTokens(options.CachePageTokens)
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(openErr error) (*Runner, error) {
		_ = file.Close()
		return nil, openErr
	}
	var spec model.Spec
	if options.Profile == nil {
		spec, err = model.ReadSpec(file)
	} else {
		spec, err = model.ReadSpecWithProfile(file, *options.Profile)
	}
	if err != nil {
		return fail(err)
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return fail(err)
	}
	program, err := modelrecipe.CompileRuntime(spec, weights)
	if err != nil {
		return fail(err)
	}
	plan := program.Model
	cacheSchemas, err := model.CompileCacheSchemas(spec, plan, weights.Layers)
	if err != nil {
		return fail(err)
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
		adapter, loadErr := model.LoadLoRA(context.Background(), configured.Path, file, spec)
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
			context.Background(),
			file,
			*weights.OutputBias,
		)
		if loadErr != nil {
			return fail(loadErr)
		}
		outputBias = slices.Clone(value.Data)
	}
	var cuda *executor.Executor
	var worker *device.Worker
	var deviceWeights *model.DeviceF32Weights
	var rawWeights *model.DeviceWeights
	var decodeWeights *model.DeviceBF16Weights
	var hostWeights *model.HostTensorStore
	if options.CacheHostWeights {
		hostWeights = model.NewHostTensorStore()
	}
	if options.PreloadDeviceWeights || options.PreloadQuantizedWeights {
		worker, err = device.New(options.DeviceOrdinal)
		if err != nil {
			return fail(err)
		}
		cuda, err = executor.NewWithWorker(worker)
		if err != nil {
			_ = worker.Close()
			return fail(err)
		}
		deviceWeights, err = model.NewDeviceF32Weights(worker)
		if err != nil {
			_ = cuda.Close()
			_ = worker.Close()
			return fail(err)
		}
		selected := selectedModelTensors(file, weights)
		f32Tensors := selected
		if options.PreloadQuantizedWeights {
			adaptedTensors := make(map[string]struct{})
			for _, loaded := range loraAdapters {
				for name := range loaded.adapter.Weights {
					adaptedTensors[name] = struct{}{}
				}
			}
			f32Required := f32RequiredModelTensors(weights)
			f32Tensors = make([]gguf.TensorInfo, 0, len(selected))
			var quantized []gguf.TensorInfo
			for _, info := range selected {
				_, adapted := adaptedTensors[info.Name]
				_, requiresF32 := f32Required[info.Name]
				if !adapted && !requiresF32 && (info.Type == dtype.Q4_0 ||
					info.Type == dtype.Q4_1 ||
					info.Type == dtype.Q5_0 ||
					info.Type == dtype.Q5_1 ||
					info.Type == dtype.Q1_0 ||
					info.Type == dtype.Q2_0 ||
					info.Type == dtype.TQ1_0 ||
					info.Type == dtype.TQ2_0 ||
					info.Type == dtype.Q8_0 ||
					info.Type == dtype.Q8_1 ||
					info.Type == dtype.Q2K ||
					info.Type == dtype.Q3K ||
					info.Type == dtype.Q4K ||
					info.Type == dtype.Q5K ||
					info.Type == dtype.Q6K ||
					info.Type == dtype.Q8K ||
					info.Type == dtype.IQ2XXS ||
					info.Type == dtype.IQ2XS ||
					info.Type == dtype.IQ2S ||
					info.Type == dtype.IQ3XXS ||
					info.Type == dtype.IQ3S ||
					info.Type == dtype.IQ1S ||
					info.Type == dtype.IQ1M ||
					info.Type == dtype.IQ4NL ||
					info.Type == dtype.IQ4XS ||
					info.Type == dtype.MXFP4 ||
					info.Type == dtype.NVFP4) {
					quantized = append(quantized, info)
				} else {
					f32Tensors = append(f32Tensors, info)
				}
			}
			rawWeights, err = model.NewDeviceWeights(worker)
			if err != nil {
				_ = deviceWeights.Close()
				_ = cuda.Close()
				_ = worker.Close()
				return fail(err)
			}
			if err = rawWeights.Load(context.Background(), file, quantized); err != nil {
				_ = rawWeights.Close()
				_ = deviceWeights.Close()
				_ = cuda.Close()
				_ = worker.Close()
				return fail(err)
			}
			if options.PreloadBF16DecodeWeights {
				decodeTensors := make([]gguf.TensorInfo, 0, len(quantized))
				for _, info := range quantized {
					_, adapted := adaptedTensors[info.Name]
					if !adapted && info.Type == dtype.Q8_0 && info.Dimensions >= 2 {
						decodeTensors = append(decodeTensors, info)
					}
				}
				decodeWeights, err = model.NewDeviceBF16Weights(worker)
				if err == nil {
					err = decodeWeights.Load(context.Background(), file, decodeTensors)
				}
				if err != nil {
					_ = decodeWeights.Close()
					_ = rawWeights.Close()
					_ = deviceWeights.Close()
					_ = cuda.Close()
					_ = worker.Close()
					return fail(err)
				}
			}
		}
		if err = deviceWeights.Load(context.Background(), file, f32Tensors); err != nil {
			_ = rawWeights.Close()
			_ = deviceWeights.Close()
			_ = cuda.Close()
			_ = worker.Close()
			return fail(err)
		}
	} else {
		cuda, err = executor.New(options.DeviceOrdinal)
		if err != nil {
			return fail(err)
		}
	}
	return &Runner{preparedModel: preparedModel{
		file: file, path: path, spec: spec, program: program, plan: plan, cacheSchemas: cacheSchemas,
		weights: weights, vocab: vocab,
		cuda: cuda, worker: worker, deviceWeights: deviceWeights, rawWeights: rawWeights, decodeWeights: decodeWeights,
		hostWeights:         hostWeights,
		outputBias:          outputBias,
		promptCacheCapacity: promptCacheCapacity, cachePageTokens: cachePageTokens,
	}, runnerState: runnerState{loraAdapters: loraAdapters}}, nil
}

func (r *Runner) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	return errors.Join(
		r.runnerState.release(context.Background()),
		r.preparedModel.close(),
	)
}

func (s *runnerState) release(ctx context.Context) error {
	var errs []error
	for _, promptCache := range s.promptCaches {
		if promptCache.Device != nil {
			errs = append(errs, promptCache.Device.Release(ctx))
			promptCache.Device = nil
		}
	}
	s.promptCaches = nil
	return errors.Join(errs...)
}

func (m *preparedModel) close() error {
	var errs []error
	if m.hostWeights != nil {
		m.hostWeights.Release()
		m.hostWeights = nil
	}
	if m.rawWeights != nil {
		errs = append(errs, m.rawWeights.Close())
	}
	if m.deviceWeights != nil {
		errs = append(errs, m.deviceWeights.Close())
	}
	if m.decodeWeights != nil {
		errs = append(errs, m.decodeWeights.Close())
	}
	if m.cuda != nil {
		errs = append(errs, m.cuda.Close())
	}
	if m.worker != nil {
		errs = append(errs, m.worker.Close())
	}
	if m.file != nil {
		errs = append(errs, m.file.Close())
	}
	return errors.Join(errs...)
}
