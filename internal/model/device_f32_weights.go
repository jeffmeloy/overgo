package model

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

type DeviceF32Tensor struct {
	Info    gguf.TensorInfo
	Shape   tensor.Shape
	Pointer driver.DevicePtr
	Size    uint64
}

// DeviceF32Weights: owns host-dequantized F32 weights in one CUDA context
// correctness bridge used before native quantized CUDA matmul
type DeviceF32Weights struct {
	worker *device.Worker

	mu      sync.RWMutex
	tensors map[string]DeviceF32Tensor
	closed  bool
}

func NewDeviceF32Weights(worker *device.Worker) (*DeviceF32Weights, error) {
	if worker == nil {
		return nil, errors.New("F32 device weights require a CUDA worker")
	}
	return &DeviceF32Weights{
		worker:  worker,
		tensors: make(map[string]DeviceF32Tensor),
	}, nil
}

// Load dequantizes and uploads tensors one at time; operation is
// transactional for supplied batch
func (w *DeviceF32Weights) Load(
	ctx context.Context,
	file *gguf.File,
	infos []gguf.TensorInfo,
) (returnErr error) {
	if file == nil {
		return errors.New("F32 device weights: GGUF file is nil")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return errors.New("F32 device weights are closed")
	}
	var added []string
	defer func() {
		if returnErr == nil {
			return
		}
		var pointers []driver.DevicePtr
		for _, name := range added {
			pointers = append(pointers, w.tensors[name].Pointer)
			delete(w.tensors, name)
		}
		_ = w.worker.Do(context.Background(), func(state *device.State) error {
			for _, pointer := range pointers {
				_ = state.Driver.MemFree(pointer)
			}
			return nil
		})
	}()
	for _, info := range infos {
		if _, exists := w.tensors[info.Name]; exists {
			return fmt.Errorf("F32 device tensor %q is already loaded", info.Name)
		}
		value, err := LoadHostTensor(ctx, file, info)
		if err != nil {
			return err
		}
		if uint64(len(value.Data)) > ^uint64(0)/4 {
			return fmt.Errorf("F32 device tensor %q byte size overflows", info.Name)
		}
		size := uint64(len(value.Data)) * 4
		var pointer driver.DevicePtr
		if err := w.worker.Do(ctx, func(state *device.State) error {
			var allocateErr error
			pointer, allocateErr = state.Driver.MemAlloc(size)
			if allocateErr != nil {
				return allocateErr
			}
			if copyErr := state.Driver.MemcpyHtoD(pointer, f32Bytes(value.Data)); copyErr != nil {
				_ = state.Driver.MemFree(pointer)
				pointer = 0
				return copyErr
			}
			return nil
		}); err != nil {
			return fmt.Errorf("upload F32 device tensor %q: %w", info.Name, err)
		}
		w.tensors[info.Name] = DeviceF32Tensor{
			Info:    info,
			Shape:   value.Shape,
			Pointer: pointer,
			Size:    size,
		}
		added = append(added, info.Name)
	}
	return nil
}

func (w *DeviceF32Weights) Lookup(name string) (DeviceF32Tensor, bool) {
	if w == nil {
		return DeviceF32Tensor{}, false
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.closed {
		return DeviceF32Tensor{}, false
	}
	value, ok := w.tensors[name]
	return value, ok
}

func (w *DeviceF32Weights) Input(
	builder *tensor.Builder,
	name string,
) (*tensor.Tensor, driver.DevicePtr, error) {
	value, ok := w.Lookup(name)
	if !ok {
		return nil, 0, fmt.Errorf("F32 device tensor %q is not loaded", name)
	}
	node := builder.Input(name, dtype.F32, value.Shape)
	if err := builder.Err(); err != nil {
		return nil, 0, err
	}
	return node, value.Pointer, nil
}

func (w *DeviceF32Weights) LayerGraphInputs(
	builder *tensor.Builder,
	info LayerWeights,
) (LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	if builder == nil {
		return LayerGraphWeights{}, nil, errors.New("F32 device layer graph builder is nil")
	}
	feeds := make(map[*tensor.Tensor]driver.DevicePtr, 11)
	input := func(tensorInfo gguf.TensorInfo) *tensor.Tensor {
		node, pointer, err := w.Input(builder, tensorInfo.Name)
		if err != nil {
			return nil
		}
		feeds[node] = pointer
		return node
	}
	// Preflight lookups: stable missing-tensor errors.
	required := []gguf.TensorInfo{}
	if info.FeedForwardRouter != nil {
		if (info.FeedForwardUpExperts == nil && info.FeedForwardGateUpExperts == nil) ||
			info.FeedForwardDownExperts == nil {
			return LayerGraphWeights{}, nil, errors.New("F32 device expert layer catalog is incomplete")
		}
		required = append(required,
			*info.FeedForwardRouter,
			*info.FeedForwardDownExperts,
		)
		if info.FeedForwardGateUpExperts != nil {
			required = append(required, *info.FeedForwardGateUpExperts)
		} else {
			if info.FeedForwardGateExperts != nil {
				required = append(required, *info.FeedForwardGateExperts)
			}
			required = append(required, *info.FeedForwardUpExperts)
		}
	}
	if info.FeedForwardUp.Name != "" && info.FeedForwardDown.Name != "" {
		required = append(required, info.FeedForwardUp, info.FeedForwardDown)
	}
	if info.Recurrent {
		var recurrent []*gguf.TensorInfo
		if info.TimeMixW1 != nil {
			recurrent = []*gguf.TensorInfo{
				info.TimeMixW1, info.TimeMixW2, info.TimeMixLerpX, info.TimeMixLerpFused,
				info.TimeMixW0, info.TimeMixA0, info.TimeMixA1, info.TimeMixA2,
				info.TimeMixV0, info.TimeMixV1, info.TimeMixV2, info.TimeMixG1, info.TimeMixG2,
				info.TimeMixKK, info.TimeMixKA, info.TimeMixRK,
				info.TimeMixLerpW, info.TimeMixLerpK, info.TimeMixLerpV, info.TimeMixLerpR, info.TimeMixLerpG,
				info.TimeMixFirst,
				info.TimeMixDecay, info.TimeMixDecayW1, info.TimeMixDecayW2,
				info.TimeMixKey, info.TimeMixValue, info.TimeMixReceptance,
				info.TimeMixGate, info.TimeMixLN, info.TimeMixLNBias, info.TimeMixOutput,
				info.ChannelMixLerpK, info.ChannelMixLerpR, info.ChannelMixKey,
				info.ChannelMixValue, info.ChannelMixReceptance,
			}
		} else if info.ShortConvKernel != nil {
			recurrent = []*gguf.TensorInfo{info.ShortConvKernel, info.ShortConvInput, info.ShortConvOutput}
		} else if info.SSMInput != nil {
			recurrent = []*gguf.TensorInfo{
				info.SSMInput, info.SSMConv1D,
				info.SSMTimeStep, info.SSMA, info.SSMD, info.SSMOutput,
			}
			if info.SSMConv1DBias != nil {
				recurrent = append(recurrent, info.SSMConv1DBias)
			}
			if info.SSMX != nil {
				recurrent = append(recurrent, info.SSMX, info.SSMTimeStepWeight)
				if info.SSMTimeStepNorm != nil {
					recurrent = append(recurrent, info.SSMTimeStepNorm, info.SSMBNorm, info.SSMCNorm)
				}
			} else {
				recurrent = append(recurrent, info.SSMNorm)
			}
		} else {
			recurrent = []*gguf.TensorInfo{
				info.AttentionQKV, info.AttentionGate, info.SSMConv1D, info.SSMTimeStep,
				info.SSMA, info.SSMBeta, info.SSMAlpha, info.SSMBetaAlpha, info.SSMNorm, info.SSMOutput,
			}
		}
		for _, item := range recurrent {
			if item == nil {
				continue
			}
			required = append(required, *item)
		}
	} else {
		if info.AttentionOutput.Name != "" {
			required = append(required, info.AttentionOutput)
		}
		if info.AttentionQKV != nil {
			required = append(required, *info.AttentionQKV)
		} else if info.AttentionKVAMQA != nil {
			required = append(required, info.AttentionQ, *info.AttentionKVAMQA, *info.AttentionKVANorm)
			if info.AttentionKVB != nil {
				required = append(required, *info.AttentionKVB)
			} else {
				required = append(required, *info.AttentionKB, *info.AttentionVB)
			}
		} else if info.AttentionQ.Name != "" {
			required = append(required, info.AttentionQ)
			if info.AttentionK.Name != "" {
				required = append(required, info.AttentionK)
			}
			if info.AttentionV.Name != "" {
				required = append(required, info.AttentionV)
			}
		}
	}
	if !info.Recurrent && info.SSMInput != nil {
		for _, item := range []*gguf.TensorInfo{
			info.SSMInput, info.SSMConv1D, info.SSMConv1DBias, info.SSMTimeStep,
			info.SSMA, info.SSMD, info.SSMNorm, info.SSMOutput,
		} {
			if item != nil {
				required = append(required, *item)
			}
		}
	}
	if info.FeedForwardGate.Name != "" {
		required = append(required, info.FeedForwardGate)
	}
	if info.AttentionNorm.Name != "" {
		required = append(required, info.AttentionNorm)
	}
	if info.AttentionNormBias != nil {
		required = append(required, *info.AttentionNormBias)
	}
	if info.AttentionNorm2 != nil {
		required = append(required, *info.AttentionNorm2)
	}
	if info.AttentionNorm2Bias != nil {
		required = append(required, *info.AttentionNorm2Bias)
	}
	if info.FeedForwardNorm.Name != "" {
		required = append(required, info.FeedForwardNorm)
	}
	if info.FeedForwardNormBias != nil {
		required = append(required, *info.FeedForwardNormBias)
	}
	if info.FeedForwardExpertNorm != nil {
		required = append(required, *info.FeedForwardExpertNorm)
	}
	for _, tensorInfo := range required {
		if _, ok := w.Lookup(tensorInfo.Name); !ok {
			return LayerGraphWeights{}, nil, fmt.Errorf("F32 device tensor %q is not loaded", tensorInfo.Name)
		}
	}
	result := LayerGraphWeights{}
	if info.FeedForwardUp.Name != "" && info.FeedForwardDown.Name != "" {
		result.FeedForwardUp = input(info.FeedForwardUp)
		result.FeedForwardDown = input(info.FeedForwardDown)
	}
	if !info.Recurrent {
		if info.AttentionOutput.Name != "" {
			result.AttentionOutput = input(info.AttentionOutput)
		}
		if info.AttentionQKV == nil && info.AttentionKVAMQA != nil {
			result.AttentionQ = input(info.AttentionQ)
		} else if info.AttentionQKV == nil && info.AttentionQ.Name != "" {
			result.AttentionQ = input(info.AttentionQ)
			if info.AttentionK.Name != "" {
				result.AttentionK = input(info.AttentionK)
			}
			if info.AttentionV.Name != "" {
				result.AttentionV = input(info.AttentionV)
			}
		}
	}
	if info.FeedForwardGate.Name != "" {
		result.FeedForwardGate = input(info.FeedForwardGate)
	}
	if info.AttentionNorm.Name != "" {
		result.AttentionNorm = input(info.AttentionNorm)
	}
	if info.FeedForwardNorm.Name != "" {
		result.FeedForwardNorm = input(info.FeedForwardNorm)
	}
	if err := bindDeviceLayerGraphFields(w, builder, &info, &result, feeds); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	if err := builder.Err(); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	return result, feeds, nil
}

func (w *DeviceF32Weights) Count() int {
	if w == nil {
		return 0
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return len(w.tensors)
}

func (w *DeviceF32Weights) Close() error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	pointers := make([]driver.DevicePtr, 0, len(w.tensors))
	for _, value := range w.tensors {
		pointers = append(pointers, value.Pointer)
	}
	clear(w.tensors)
	return w.worker.Do(context.Background(), func(state *device.State) error {
		var errs []error
		for _, pointer := range pointers {
			if err := state.Driver.MemFree(pointer); err != nil {
				errs = append(errs, err)
			}
		}
		return errors.Join(errs...)
	})
}

func f32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
