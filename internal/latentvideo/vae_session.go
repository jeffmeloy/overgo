// Full-CUDA VAE decode: one session owns the device-resident decoder weights
// (uploaded once, 293MB) and streams the compiled 21-op plan one latent frame
// (chunk) at a time, exactly mirroring the host DecodeLatentVideo graph —
// per-op temporal caches, the temporal upsample 'Rep' first-chunk convention,
// clamp, per-frame emission. Engine: direct tiled fp32-FMA causal conv3d,
// f64 channel RMS norm, fp32 spatial attention — the same accumulation class
// as the CUDA-captured goldens the host oracle was verified against.
package latentvideo

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"
	"unsafe"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
)

type vaeDeviceWeights = media.CodecBindings[driver.DevicePtr]

// vaeDeviceCache: device-resident temporal cache (reference feat_cache).
type vaeDeviceCache struct {
	ptr              driver.DevicePtr
	frames, slot     int
	initialized, rep bool
}

type vaeDeviceOpState struct {
	cache0, cache1 vaeDeviceCache
}

type vaeDeviceBuffer struct {
	ptr      driver.DevicePtr
	elements int
}

// VAEDecoderCUDASession: device residency plus streaming state for one
// compiled decoder plan.
type VAEDecoderCUDASession struct {
	Plan        VAEDecoderPlan
	WeightBytes uint64

	worker  *device.Worker
	module  driver.Module
	kernels [media.CodecKernelABICount]driver.Function
	weights []vaeDeviceWeights
	buffers map[media.CodecWorkspaceKey]vaeDeviceBuffer

	// profile: per-op accumulated synchronized wall, enabled by
	// OVERGO_VAE_PROFILE=1 (measurement apparatus; nil in production).
	profile map[string]time.Duration

	ctx context.Context
}

// NewVAEDecoderCUDASession: loads the kernel module and uploads every decoder
// weight tensor once (f32, the checkpoint dtype the host oracle serves).
func NewVAEDecoderCUDASession(checkpoint string, plan VAEDecoderPlan, ordinal int) (session *VAEDecoderCUDASession, err error) {
	session, err = newVAECUDASession(checkpoint, plan.vaePlanCore, ordinal)
	if session != nil {
		session.Plan = plan
	}
	return session, err
}

func newVAECUDASession(checkpoint string, plan vaePlanCore, ordinal int) (session *VAEDecoderCUDASession, err error) {
	if len(plan.Operations) == 0 {
		return nil, errors.New("vae cuda session: empty plan")
	}
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	session = &VAEDecoderCUDASession{
		Plan:    VAEDecoderPlan{vaePlanCore: plan},
		worker:  worker,
		buffers: make(map[media.CodecWorkspaceKey]vaeDeviceBuffer),
		ctx:     context.Background(),
	}
	if os.Getenv("OVERGO_VAE_PROFILE") == "1" {
		session.profile = make(map[string]time.Duration)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Close())
			session = nil
		}
	}()
	if err = worker.Do(session.ctx, func(state *device.State) error {
		if assetErr := kernel.ValidateAssets(); assetErr != nil {
			return assetErr
		}
		module, moduleErr := state.Driver.ModuleLoadData(kernel.OpsF32PTX)
		if moduleErr != nil {
			return moduleErr
		}
		session.module = module
		kernels := plan.CodecProgram.KernelABIs()
		for abi := media.CodecKernelFirstABI; abi < media.CodecKernelABICount; abi++ {
			if !kernels.Has(abi) {
				continue
			}
			function, functionErr := state.Driver.ModuleFunction(module, abi.EntryPoint())
			if functionErr != nil {
				return fmt.Errorf("vae cuda session kernel %s: %w", abi.EntryPoint(), functionErr)
			}
			session.kernels[abi] = function
		}
		return nil
	}); err != nil {
		return session, err
	}
	if err = session.uploadWeights(checkpoint); err != nil {
		return session, err
	}
	return session, nil
}

func (s *VAEDecoderCUDASession) uploadWeights(checkpoint string) error {
	reader, err := pytorchzip.Open(checkpoint)
	if err != nil {
		return err
	}
	defer reader.Close()
	s.weights = make([]vaeDeviceWeights, len(s.Plan.Operations))
	for index, op := range s.Plan.Operations {
		values, readErr := reader.ReadBindingValues(op.BindingValues())
		if readErr != nil {
			return fmt.Errorf("vae cuda session %s: %w", op.Name, readErr)
		}
		pointers := make([]driver.DevicePtr, len(values))
		if err := s.worker.Do(s.ctx, func(state *device.State) error {
			for valueIndex, value := range values {
				payload := driver.Bytes(value)
				pointer, allocErr := state.Driver.MemAlloc(uint64(len(payload)))
				if allocErr != nil {
					return allocErr
				}
				pointers[valueIndex] = pointer
				if copyErr := state.Driver.MemcpyHtoD(pointer, payload); copyErr != nil {
					return copyErr
				}
				s.WeightBytes += uint64(len(payload))
			}
			return nil
		}); err != nil {
			return fmt.Errorf("vae cuda session upload %s: %w", op.Name, err)
		}
		s.weights[index], err = media.BindCodecWeights(op.Operator, op.RequiresProjection(), pointers)
		if err != nil {
			return fmt.Errorf("vae cuda session upload %s: %w", op.Name, err)
		}
	}
	return nil
}

// buffer: typed grow-only device workspace; growth synchronizes the stream
// before freeing the prior allocation (queued kernels may still read it).
func (s *VAEDecoderCUDASession) buffer(state *device.State, key media.CodecWorkspaceKey, elements int) (driver.DevicePtr, error) {
	if !checked.PositiveInts(elements) {
		return 0, fmt.Errorf("vae cuda buffer %+v: elements=%d", key, elements)
	}
	existing := s.buffers[key]
	if checked.Nonzero(existing.ptr) && checked.AtLeastInt(existing.elements, elements) {
		return existing.ptr, nil
	}
	if checked.Nonzero(existing.ptr) {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return 0, err
		}
		if err := state.Driver.MemFree(existing.ptr); err != nil {
			return 0, err
		}
		delete(s.buffers, key)
	}
	pointer, err := state.Driver.MemAlloc(uint64(elements) * binaryschema.Uint32Bytes)
	if err != nil {
		return 0, fmt.Errorf("vae cuda buffer %+v (%d elements): %w", key, elements, err)
	}
	s.buffers[key] = vaeDeviceBuffer{ptr: pointer, elements: elements}
	return pointer, nil
}

func launchVAEKernel(state *device.State, function driver.Function, grid, block driver.Dim3, shared uint32, args ...any) error {
	pointers := make([]unsafe.Pointer, len(args))
	for index, argument := range args {
		switch value := argument.(type) {
		case *driver.DevicePtr:
			pointers[index] = unsafe.Pointer(value)
		case *uint32:
			pointers[index] = unsafe.Pointer(value)
		case *float32:
			pointers[index] = unsafe.Pointer(value)
		default:
			return fmt.Errorf("vae cuda launch: unsupported argument %T", argument)
		}
	}
	err := state.Driver.LaunchKernel(function, grid, block, shared, state.Stream, pointers)
	runtime.KeepAlive(args)
	return err
}

func (s *VAEDecoderCUDASession) conv3d(state *device.State, out, x, cache driver.DevicePtr, cacheT int, weight, bias driver.DevicePtr, cIn, cOut, frames, h, w, kt, kh, kw int) error {
	positions := frames * h * w
	tile := kernel.VAEConvTile()
	channelTiles := (cOut + tile - 1) / tile
	positionTiles := (positions + tile - 1) / tile
	cInArg, cOutArg := uint32(cIn), uint32(cOut)
	inT, inH, inW := uint32(frames), uint32(h), uint32(w)
	ktArg, khArg, kwArg := uint32(kt), uint32(kh), uint32(kw)
	padT, padH, padW := uint32(kt/tensor.PairedExtent), uint32(kh/tensor.PairedExtent), uint32(kw/tensor.PairedExtent)
	cacheTArg := uint32(cacheT)
	return launchVAEKernel(state, s.kernels[media.CodecKernelConvolution],
		kernel.Grid1D(channelTiles*positionTiles), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&out, &x, &cache, &weight, &bias,
		&cInArg, &cOutArg, &inT, &inH, &inW,
		&ktArg, &khArg, &kwArg, &padT, &padH, &padW, &cacheTArg)
}

func (s *VAEDecoderCUDASession) rmsNorm(state *device.State, out, x, gamma driver.DevicePtr, c, plane int, silu bool) error {
	channels, planeArg, applySiLU := uint32(c), uint32(plane), uint32(tensor.FirstOffset)
	if silu {
		applySiLU = uint32(tensor.SingletonExtent)
	}
	return launchVAEKernel(state, s.kernels[media.CodecKernelRMSNorm],
		kernel.Grid1D(plane), kernel.DefaultBlock1D(), uint32(kernel.DefaultThreads()*binaryschema.Uint64Bytes),
		&out, &x, &gamma, &channels, &planeArg, &applySiLU)
}

func (s *VAEDecoderCUDASession) addInPlace(state *device.State, out, a, b driver.DevicePtr, elements int) error {
	count := uint32(elements)
	return launchVAEKernel(state, s.kernels[media.CodecKernelAdd], kernel.ElementwiseGrid(elements), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&a, &b, &out, &count)
}

// updateCache: reference temporal_cache_update semantics on device; the new
// cache lands in the opposite slot so in-flight consumers of the prior cache
// stay valid on the stream.
func (s *VAEDecoderCUDASession) updateCache(state *device.State, owner media.CodecWorkspaceKey, prior vaeDeviceCache, x driver.DevicePtr, c, frames, spatial int, replicatePrefix bool) (vaeDeviceCache, error) {
	nextFrames := min(tensor.PairedExtent, frames)
	mode := uint32(tensor.FirstOffset)
	if replicatePrefix {
		nextFrames, _ = checked.AddInt(frames, tensor.SingletonExtent)
		mode = uint32(tensor.SingletonExtent)
	} else if !checked.AtLeastInt(frames, tensor.PairedExtent) && prior.initialized && checked.PositiveInts(prior.frames) {
		nextFrames, _ = checked.AddInt(frames, tensor.SingletonExtent)
	}
	slot := tensor.SingletonExtent - prior.slot
	owner.Slot = slot
	out, err := s.buffer(state, owner, c*nextFrames*spatial)
	if err != nil {
		return vaeDeviceCache{}, err
	}
	elements := c * nextFrames * spatial
	channels, currentFrames, priorFrames := uint32(c), uint32(frames), uint32(prior.frames)
	outFrames, spatialArg, count := uint32(nextFrames), uint32(spatial), uint32(elements)
	if err := launchVAEKernel(state, s.kernels[media.CodecKernelCacheUpdate], kernel.ElementwiseGrid(elements), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&out, &x, &prior.ptr,
		&channels, &currentFrames, &priorFrames, &outFrames, &spatialArg, &mode, &count); err != nil {
		return vaeDeviceCache{}, err
	}
	return vaeDeviceCache{ptr: out, frames: nextFrames, slot: slot, initialized: true}, nil
}

// runOp executes one op on one device chunk [cIn][frames][h][w], mutating the
// op's temporal state exactly as the host runVAEOp does. actName is the
// ping-pong activation target for this op's output.
func (s *VAEDecoderCUDASession) runOp(state *device.State, opIndex, chunkIndex int, op media.CodecOperation[pytorchzip.TensorBinding], values vaeDeviceWeights, opState *vaeDeviceOpState, x driver.DevicePtr, activation media.CodecWorkspaceKey, frames, h, w int) (driver.DevicePtr, int, int, int, error) {
	spatial := h * w
	c := op.InputChannels
	cachedConv := func(out, input driver.DevicePtr, cache *vaeDeviceCache, weight, bias driver.DevicePtr, cOut, kt, kh, kw int) error {
		if err := s.conv3d(state, out, input, cache.ptr, cache.frames, weight, bias, c, cOut, frames, h, w, kt, kh, kw); err != nil {
			return err
		}
		if checked.Nonzero(kt / tensor.PairedExtent) {
			next, err := s.updateCache(state, media.OperationCodecWorkspace(media.CodecWorkspaceCacheInput, opIndex, tensor.FirstOffset), *cache, input, c, frames, spatial, false)
			if err != nil {
				return err
			}
			*cache = next
		}
		return nil
	}
	switch op.Operator {
	case media.CodecPointwise, media.CodecConvolution:
		weight, bias := values.WeightInput, values.BiasInput
		kernel := op.Convolution.Kernel
		out, err := s.buffer(state, activation, op.OutputChannels*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := cachedConv(out, x, &opState.cache0, weight, bias, op.OutputChannels, kernel[0], kernel[1], kernel[2]); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	case media.CodecResidual:
		gamma0, w0, b0 := values.NormInput, values.WeightInput, values.BiasInput
		gamma1, w1, b1 := values.NormOutput, values.WeightOutput, values.BiasOutput
		n0, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.FirstOffset), c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.rmsNorm(state, n0, x, gamma0, c, frames*spatial, true); err != nil {
			return 0, 0, 0, 0, err
		}
		h0, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.SingletonExtent), op.OutputChannels*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		kernel := op.Convolution.Kernel
		if err := cachedConv(h0, n0, &opState.cache0, w0, b0, op.OutputChannels, kernel[0], kernel[1], kernel[2]); err != nil {
			return 0, 0, 0, 0, err
		}
		n1, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.FirstOffset), op.OutputChannels*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.rmsNorm(state, n1, h0, gamma1, op.OutputChannels, frames*spatial, true); err != nil {
			return 0, 0, 0, 0, err
		}
		out, err := s.buffer(state, activation, op.OutputChannels*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.conv3d(state, out, n1, opState.cache1.ptr, opState.cache1.frames, w1, b1, op.OutputChannels, op.OutputChannels, frames, h, w, kernel[0], kernel[1], kernel[2]); err != nil {
			return 0, 0, 0, 0, err
		}
		nextCache1, err := s.updateCache(state, media.OperationCodecWorkspace(media.CodecWorkspaceCacheOutput, opIndex, tensor.FirstOffset), opState.cache1, n1, op.OutputChannels, frames, spatial, false)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		opState.cache1 = nextCache1
		if !op.RequiresProjection() {
			if err := s.addInPlace(state, out, out, x, op.OutputChannels*frames*spatial); err != nil {
				return 0, 0, 0, 0, err
			}
			return out, frames, h, w, nil
		}
		shortcut, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.SingletonExtent), op.OutputChannels*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		var noCache driver.DevicePtr
		if err := s.conv3d(state, shortcut, x, noCache, tensor.FirstOffset, values.WeightProjection, values.BiasProjection, c, op.OutputChannels, frames, h, w, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent); err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.addInPlace(state, out, out, shortcut, op.OutputChannels*frames*spatial); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	case media.CodecAttention:
		gamma, qkvW, qkvB := values.NormInput, values.WeightInput, values.BiasInput
		projW, projB := values.WeightOutput, values.BiasOutput
		norm, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.FirstOffset), c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.rmsNorm(state, norm, x, gamma, c, frames*spatial, false); err != nil {
			return 0, 0, 0, 0, err
		}
		qkvChannels, ok := checked.MulInt(tensor.TripleExtent, c)
		qkvElements, elementsOK := checked.ProductInt(qkvChannels, frames, spatial)
		if !ok || !elementsOK {
			return 0, 0, 0, 0, fmt.Errorf("vae cuda attention geometry overflows")
		}
		qkv, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceAttention, tensor.FirstOffset), qkvElements)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		var noCache driver.DevicePtr
		if err := s.conv3d(state, qkv, norm, noCache, tensor.FirstOffset, qkvW, qkvB, c, qkvChannels, frames, h, w, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent); err != nil {
			return 0, 0, 0, 0, err
		}
		// scores (8-aligned) + one double per thread (kernel shared layout).
		shared, ok := kernel.AttentionSharedMemoryBytes(spatial, kernel.DefaultThreads())
		if !ok {
			return 0, 0, 0, 0, fmt.Errorf("vae cuda attention shared-memory geometry overflows")
		}
		channels, groups, frame := uint32(c), uint32(frames), uint32(spatial)
		scale := float32(tensor.SingletonExtent) / float32(math.Sqrt(float64(c)))
		if err := launchVAEKernel(state, s.kernels[media.CodecKernelAttention],
			kernel.Grid1D(frames*spatial), kernel.DefaultBlock1D(), shared,
			&qkv, &norm, &channels, &groups, &frame, &scale); err != nil {
			return 0, 0, 0, 0, err
		}
		out, err := s.buffer(state, activation, c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.conv3d(state, out, norm, noCache, tensor.FirstOffset, projW, projB, c, c, frames, h, w, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent); err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.addInPlace(state, out, out, x, c*frames*spatial); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	case media.CodecDownsampleSpatial:
		scale := op.Operator.SpatialScale()
		outH, outW := h/scale, w/scale
		out, err := s.buffer(state, activation, op.OutputChannels*frames*outH*outW)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.downsample2D(state, out, x, values.WeightSpatial, values.BiasSpatial, c, frames, h, w); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, outH, outW, nil
	case media.CodecDownsampleSpatiotemporal:
		scale := op.Operator.SpatialScale()
		outH, outW := h/scale, w/scale
		spatial := outH * outW
		spatialOut, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.FirstOffset), op.OutputChannels*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.downsample2D(state, spatialOut, x, values.WeightSpatial, values.BiasSpatial, c, frames, h, w); err != nil {
			return 0, 0, 0, 0, err
		}
		prior := opState.cache0
		next, err := s.updateCache(state, media.OperationCodecWorkspace(media.CodecWorkspaceCacheInput, opIndex, tensor.FirstOffset), opState.cache0, spatialOut, op.OutputChannels, frames, spatial, false)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		opState.cache0 = next
		if checked.Equal(chunkIndex, tensor.FirstOffset) {
			out, err := s.buffer(state, activation, op.OutputChannels*frames*spatial)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			if err := state.Driver.MemcpyDtoD(out, spatialOut, uint64(op.OutputChannels*frames*spatial*binaryschema.Uint32Bytes)); err != nil {
				return 0, 0, 0, 0, err
			}
			return out, frames, outH, outW, nil
		}
		outFrames := (frames-tensor.SingletonExtent)/op.Operator.TemporalScale() + tensor.SingletonExtent
		out, err := s.buffer(state, activation, op.OutputChannels*outFrames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.temporalDownsample(state, out, spatialOut, prior.ptr, values.WeightTemporal, values.BiasTemporal, op.OutputChannels, frames, prior.frames, spatial, outFrames); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, outFrames, outH, outW, nil
	case media.CodecUpsampleSpatial:
		scale := op.Operator.SpatialScale()
		out, err := s.buffer(state, activation, op.OutputChannels*frames*scale*h*scale*w)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.upsample2D(state, out, x, values.WeightSpatial, values.BiasSpatial, c, op.OutputChannels, frames, h, w); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, scale * h, scale * w, nil
	case media.CodecUpsampleSpatiotemporal:
		timeW, timeB := values.WeightTemporal, values.BiasTemporal
		resampleW, resampleB := values.WeightSpatial, values.BiasSpatial
		spatialInput, spatialFrames := x, frames
		if checked.Equal(chunkIndex, tensor.FirstOffset) {
			opState.cache0 = vaeDeviceCache{initialized: true, rep: true}
		} else {
			replicatePrefix := opState.cache0.rep && !checked.AtLeastInt(frames, tensor.PairedExtent)
			nextCache, err := s.updateCache(state, media.OperationCodecWorkspace(media.CodecWorkspaceCacheInput, opIndex, tensor.FirstOffset), opState.cache0, x, c, frames, spatial, replicatePrefix)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			cachePtr, cacheFrames := opState.cache0.ptr, opState.cache0.frames
			if opState.cache0.rep {
				var noCache driver.DevicePtr
				cachePtr, cacheFrames = noCache, tensor.FirstOffset
			}
			temporalChannels, ok := checked.MulInt(tensor.PairedExtent, c)
			elements, elementsOK := checked.ProductInt(temporalChannels, frames, spatial)
			if !ok || !elementsOK {
				return 0, 0, 0, 0, fmt.Errorf("vae cuda temporal upsample geometry overflows")
			}
			timeOut, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.FirstOffset), elements)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			if err := s.conv3d(state, timeOut, x, cachePtr, cacheFrames, timeW, timeB, c, temporalChannels, frames, h, w, tensor.TripleExtent, tensor.SingletonExtent, tensor.SingletonExtent); err != nil {
				return 0, 0, 0, 0, err
			}
			spatialFrames, _ = checked.MulInt(tensor.PairedExtent, frames)
			interleaved, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.SingletonExtent), c*spatialFrames*spatial)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			channels, framesArg, spatialArg, count := uint32(c), uint32(frames), uint32(spatial), uint32(elements)
			if err := launchVAEKernel(state, s.kernels[media.CodecKernelInterleave], kernel.ElementwiseGrid(elements), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
				&interleaved, &timeOut, &channels, &framesArg, &spatialArg, &count); err != nil {
				return 0, 0, 0, 0, err
			}
			spatialInput = interleaved
			opState.cache0 = nextCache
		}
		scale := op.Operator.SpatialScale()
		out, err := s.buffer(state, activation, op.OutputChannels*spatialFrames*scale*h*scale*w)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.upsample2D(state, out, spatialInput, resampleW, resampleB, c, op.OutputChannels, spatialFrames, h, w); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, spatialFrames, scale * h, scale * w, nil
	case media.CodecHead:
		gamma, weight, bias := values.NormInput, values.WeightInput, values.BiasInput
		norm, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, tensor.FirstOffset), c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.rmsNorm(state, norm, x, gamma, c, frames*spatial, true); err != nil {
			return 0, 0, 0, 0, err
		}
		out, err := s.buffer(state, activation, op.OutputChannels*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		kernel := op.Convolution.Kernel
		if err := cachedConv(out, norm, &opState.cache0, weight, bias, op.OutputChannels, kernel[0], kernel[1], kernel[2]); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	}
	return 0, 0, 0, 0, fmt.Errorf("vae cuda decode: unsupported op %d", op.Operator)
}

func (s *VAEDecoderCUDASession) upsample2D(state *device.State, out, x, weight, bias driver.DevicePtr, cIn, cOut, frames, h, w int) error {
	scale := media.CodecUpsampleSpatial.SpatialScale()
	outH, outW := scale*h, scale*w
	elements := cOut * frames * outH * outW
	cInArg, cOutArg, framesArg := uint32(cIn), uint32(cOut), uint32(frames)
	heightArg, widthArg := uint32(h), uint32(w)
	outHArg, outWArg, count := uint32(outH), uint32(outW), uint32(elements)
	return launchVAEKernel(state, s.kernels[media.CodecKernelUpsample], kernel.ElementwiseGrid(elements), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&out, &x, &weight, &bias,
		&cInArg, &cOutArg, &framesArg, &heightArg, &widthArg, &outHArg, &outWArg, &count)
}

func (s *VAEDecoderCUDASession) downsample2D(state *device.State, out, x, weight, bias driver.DevicePtr, channels, frames, h, w int) error {
	scale := media.CodecDownsampleSpatial.SpatialScale()
	outH, outW := h/scale, w/scale
	elements := channels * frames * outH * outW
	channelsArg, framesArg := uint32(channels), uint32(frames)
	heightArg, widthArg := uint32(h), uint32(w)
	outHArg, outWArg, count := uint32(outH), uint32(outW), uint32(elements)
	return launchVAEKernel(state, s.kernels[media.CodecKernelDownsample], kernel.ElementwiseGrid(elements), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&out, &x, &weight, &bias, &channelsArg, &framesArg, &heightArg, &widthArg, &outHArg, &outWArg, &count)
}

func (s *VAEDecoderCUDASession) temporalDownsample(state *device.State, out, x, prior, weight, bias driver.DevicePtr, channels, frames, priorFrames, spatial, outFrames int) error {
	elements := channels * outFrames * spatial
	channelsArg, framesArg, priorFramesArg := uint32(channels), uint32(frames), uint32(priorFrames)
	spatialArg, outFramesArg, count := uint32(spatial), uint32(outFrames), uint32(elements)
	return launchVAEKernel(state, s.kernels[media.CodecKernelTemporalDownsample], kernel.ElementwiseGrid(elements), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&out, &x, &prior, &weight, &bias, &channelsArg, &framesArg, &priorFramesArg, &spatialArg, &outFramesArg, &count)
}

// Decode streams the plan over the latent volume on CUDA: host denorm
// (z*std+mean) per chunk, upload, the op chain with device temporal caches,
// clamp to [-1,1], then per-frame device-to-host emission through the sink.
func (s *VAEDecoderCUDASession) Decode(stats VAELatentStats, z []float32, latentFrames, latentH, latentW int, sink VideoFrameSink) (VAEDecodeStats, error) {
	plan := s.Plan
	decodeStats, geometry, err := prepareVAEDecode("vae cuda decode", "cuda_streamed_chunks", plan, len(s.weights), stats, z, latentFrames, latentH, latentW, sink)
	if err != nil {
		return decodeStats, err
	}
	spatial := geometry.spatial
	started := time.Now()
	states := make([]vaeDeviceOpState, len(plan.Operations))
	staging := make([]float32, plan.ZDim*spatial)
	var frameScratch []float32
	frameIndex := tensor.FirstOffset
	for chunkIndex := range latentFrames {
		if err := s.worker.Do(s.ctx, func(state *device.State) error {
			denormalizeLatentChunk(staging, z, stats, plan.ZDim, latentFrames, spatial, chunkIndex)
			x, err := s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceActivation, tensor.FirstOffset), plan.ZDim*spatial)
			if err != nil {
				return err
			}
			if err := state.Driver.MemcpyHtoD(x, driver.Bytes(staging)); err != nil {
				return err
			}
			actIndex := tensor.FirstOffset
			volume, runErr := media.ExecuteCodecProgram("vae cuda decode", plan.CodecProgram, states, media.CodecVolume[driver.DevicePtr]{
				Storage: x, Channels: plan.ZDim, Frames: tensor.SingletonExtent, Height: latentH, Width: latentW,
			}, chunkIndex > tensor.FirstOffset, func(index int, operation media.CodecOperation[pytorchzip.TensorBinding], opState *vaeDeviceOpState, current media.CodecVolume[driver.DevicePtr]) (media.CodecVolume[driver.DevicePtr], error) {
				if err := s.ctx.Err(); err != nil {
					return media.CodecVolume[driver.DevicePtr]{}, err
				}
				actIndex = tensor.SingletonExtent - actIndex
				opStarted := time.Now()
				next, frames, height, width, stepErr := s.runOp(state, index, chunkIndex, operation, s.weights[index], opState, current.Storage, media.ProgramCodecWorkspace(media.CodecWorkspaceActivation, actIndex), current.Frames, current.Height, current.Width)
				if s.profile != nil {
					if syncErr := state.Driver.StreamSynchronize(state.Stream); syncErr != nil {
						return media.CodecVolume[driver.DevicePtr]{}, syncErr
					}
					s.profile[operation.Name] += time.Since(opStarted)
				}
				return media.CodecVolume[driver.DevicePtr]{Storage: next, Channels: operation.OutputChannels, Frames: frames, Height: height, Width: width}, stepErr
			})
			if runErr != nil {
				return fmt.Errorf("vae cuda decode chunk %d: %w", chunkIndex, runErr)
			}
			x, frames, h, w := volume.Storage, volume.Frames, volume.Height, volume.Width
			if !checked.Equal(h, geometry.height) || !checked.Equal(w, geometry.width) {
				return fmt.Errorf("vae cuda decode chunk %d output %dx%d, want %dx%d", chunkIndex, w, h, geometry.width, geometry.height)
			}
			elements, ok := checked.ProductInt(geometry.channels, frames, h, w)
			if !ok {
				return fmt.Errorf("vae cuda decode output geometry overflows")
			}
			minimum, maximum := media.SignedUnitBounds32()
			count := uint32(elements)
			if err := launchVAEKernel(state, s.kernels[media.CodecKernelClamp], kernel.ElementwiseGrid(elements), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
				&x, &x, &minimum, &maximum, &count); err != nil {
				return err
			}
			if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
				return err
			}
			chunkSpatial, ok := checked.MulInt(h, w)
			if !ok {
				return fmt.Errorf("vae cuda decode spatial geometry overflows")
			}
			frameElements, ok := checked.MulInt(geometry.channels, chunkSpatial)
			if !ok {
				return fmt.Errorf("vae cuda decode frame geometry overflows")
			}
			if cap(frameScratch) < frameElements {
				frameScratch = make([]float32, frameElements)
			}
			frame := frameScratch[:frameElements]
			for chunkFrame := range frames {
				if err := s.ctx.Err(); err != nil {
					return err
				}
				for ch := range geometry.channels {
					source := x + driver.DevicePtr((ch*frames+chunkFrame)*chunkSpatial*binaryschema.Uint32Bytes)
					if err := state.Driver.MemcpyDtoH(driver.Bytes(frame[ch*chunkSpatial:(ch+tensor.SingletonExtent)*chunkSpatial]), source); err != nil {
						return err
					}
				}
				if err := sink(frameIndex, frame, h, w); err != nil {
					return fmt.Errorf("vae cuda decode frame sink %d: %w", frameIndex, err)
				}
				frameIndex++
			}
			return nil
		}); err != nil {
			return decodeStats, err
		}
	}
	if !checked.Equal(frameIndex, geometry.frames) {
		return decodeStats, fmt.Errorf("vae cuda decode produced %d frames, want %d", frameIndex, geometry.frames)
	}
	decodeStats.OutputFrames = frameIndex
	decodeStats.DecodeWallSec = time.Since(started).Seconds()
	if memory, memErr := s.worker.MemoryStats(s.ctx); memErr == nil {
		decodeStats.PeakDeviceBytes = memory.PeakBytes
	}
	return decodeStats, nil
}

// MemoryStats: runtime-owned device allocation accounting.
func (s *VAEDecoderCUDASession) MemoryStats() (driver.MemoryStats, error) {
	return s.worker.MemoryStats(s.ctx)
}

// Profile: accumulated per-op synchronized wall (nil unless enabled).
func (s *VAEDecoderCUDASession) Profile() map[string]time.Duration {
	return s.profile
}

// Close: frees weights and workspaces, unloads the module, stops the worker.
func (s *VAEDecoderCUDASession) Close() error {
	if s == nil {
		return nil
	}
	var errs []error
	if s.worker != nil {
		errs = append(errs, s.worker.Do(context.WithoutCancel(s.ctx), func(state *device.State) error {
			var innerErrs []error
			innerErrs = append(innerErrs, state.Driver.StreamSynchronize(state.Stream))
			for index, weights := range s.weights {
				operation := s.Plan.Operations[index]
				for _, pointer := range media.CodecBindingValues(operation.Operator, operation.RequiresProjection(), weights) {
					if checked.Nonzero(pointer) {
						innerErrs = append(innerErrs, state.Driver.MemFree(pointer))
					}
				}
			}
			s.weights = nil
			for name, buffer := range s.buffers {
				if checked.Nonzero(buffer.ptr) {
					innerErrs = append(innerErrs, state.Driver.MemFree(buffer.ptr))
				}
				delete(s.buffers, name)
			}
			if checked.Nonzero(s.module) {
				innerErrs = append(innerErrs, state.Driver.ModuleUnload(s.module))
				var unloaded driver.Module
				s.module = unloaded
			}
			return errors.Join(innerErrs...)
		}))
		errs = append(errs, s.worker.Close())
		s.worker = nil
	}
	return errors.Join(errs...)
}
