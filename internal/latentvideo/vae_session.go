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

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/pytorchzip"
)

// vaeKernelBlock: manifest defaultThreads (shared launch ABI).
const vaeKernelBlock = 256

// vaeConvTile: conv kernel position-by-channel tile side (VAE_CONV_TILE).
const vaeConvTile = 64

// vaeDeviceOp: one plan op with weights resident on device.
type vaeDeviceOp struct {
	vaeDecoderOp
	values []driver.DevicePtr
}

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

type vaeKernelSet struct {
	conv3d, rmsNorm, attention, upsample2d driver.Function
	downsample2d, temporalDownsample       driver.Function
	interleave, cacheUpdate, add, clamp    driver.Function
}

// VAEDecoderCUDASession: device residency plus streaming state for one
// compiled decoder plan.
type VAEDecoderCUDASession struct {
	Plan        VAEDecoderPlan
	WeightBytes uint64

	worker  *device.Worker
	module  driver.Module
	kernels vaeKernelSet
	ops     []vaeDeviceOp
	buffers map[string]vaeDeviceBuffer

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
	if len(plan.ops) == 0 {
		return nil, errors.New("vae cuda session: empty plan")
	}
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	session = &VAEDecoderCUDASession{
		Plan:    VAEDecoderPlan{vaePlanCore: plan},
		worker:  worker,
		buffers: make(map[string]vaeDeviceBuffer),
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
		for _, item := range []struct {
			name   string
			target *driver.Function
		}{
			{"vae_causal_conv3d_f32", &session.kernels.conv3d},
			{"vae_channel_rms_norm_f32", &session.kernels.rmsNorm},
			{"vae_spatial_attention_f32", &session.kernels.attention},
			{"vae_upsample2d_f32", &session.kernels.upsample2d},
			{"vae_downsample2d_f32", &session.kernels.downsample2d},
			{"vae_temporal_downsample_f32", &session.kernels.temporalDownsample},
			{"vae_time_interleave_f32", &session.kernels.interleave},
			{"vae_temporal_cache_update_f32", &session.kernels.cacheUpdate},
			{"add_f32", &session.kernels.add},
			{"clamp_f32", &session.kernels.clamp},
		} {
			function, functionErr := state.Driver.ModuleFunction(module, item.name)
			if functionErr != nil {
				return fmt.Errorf("vae cuda session kernel %s: %w", item.name, functionErr)
			}
			*item.target = function
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
	s.ops = make([]vaeDeviceOp, len(s.Plan.ops))
	for index, op := range s.Plan.ops {
		values, readErr := reader.ReadBindingValues(op.bindings)
		if readErr != nil {
			return fmt.Errorf("vae cuda session %s: %w", op.prefix, readErr)
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
			return fmt.Errorf("vae cuda session upload %s: %w", op.prefix, err)
		}
		s.ops[index] = vaeDeviceOp{vaeDecoderOp: op, values: pointers}
	}
	return nil
}

// buffer: named grow-only device workspace; growth synchronizes the stream
// before freeing the prior allocation (queued kernels may still read it).
func (s *VAEDecoderCUDASession) buffer(state *device.State, name string, elements int) (driver.DevicePtr, error) {
	if elements <= 0 {
		return 0, fmt.Errorf("vae cuda buffer %s: elements=%d", name, elements)
	}
	existing := s.buffers[name]
	if existing.ptr != 0 && existing.elements >= elements {
		return existing.ptr, nil
	}
	if existing.ptr != 0 {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return 0, err
		}
		if err := state.Driver.MemFree(existing.ptr); err != nil {
			return 0, err
		}
		delete(s.buffers, name)
	}
	pointer, err := state.Driver.MemAlloc(uint64(elements) * 4)
	if err != nil {
		return 0, fmt.Errorf("vae cuda buffer %s (%d elements): %w", name, elements, err)
	}
	s.buffers[name] = vaeDeviceBuffer{ptr: pointer, elements: elements}
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

func vaeElementwiseGrid(elements int) driver.Dim3 {
	return driver.Dim3{X: uint32((elements + vaeKernelBlock - 1) / vaeKernelBlock), Y: 1, Z: 1}
}

var vaeBlock1D = driver.Dim3{X: vaeKernelBlock, Y: 1, Z: 1}

func (s *VAEDecoderCUDASession) conv3d(state *device.State, out, x, cache driver.DevicePtr, cacheT int, weight, bias driver.DevicePtr, cIn, cOut, frames, h, w, kt, kh, kw int) error {
	positions := frames * h * w
	channelTiles := (cOut + vaeConvTile - 1) / vaeConvTile
	positionTiles := (positions + vaeConvTile - 1) / vaeConvTile
	cInArg, cOutArg := uint32(cIn), uint32(cOut)
	inT, inH, inW := uint32(frames), uint32(h), uint32(w)
	ktArg, khArg, kwArg := uint32(kt), uint32(kh), uint32(kw)
	padT, padH, padW := uint32(kt/2), uint32(kh/2), uint32(kw/2)
	cacheTArg := uint32(cacheT)
	return launchVAEKernel(state, s.kernels.conv3d,
		driver.Dim3{X: uint32(channelTiles * positionTiles), Y: 1, Z: 1}, vaeBlock1D, 0,
		&out, &x, &cache, &weight, &bias,
		&cInArg, &cOutArg, &inT, &inH, &inW,
		&ktArg, &khArg, &kwArg, &padT, &padH, &padW, &cacheTArg)
}

func (s *VAEDecoderCUDASession) rmsNorm(state *device.State, out, x, gamma driver.DevicePtr, c, plane int, silu bool) error {
	channels, planeArg, applySiLU := uint32(c), uint32(plane), uint32(0)
	if silu {
		applySiLU = 1
	}
	return launchVAEKernel(state, s.kernels.rmsNorm,
		driver.Dim3{X: uint32(plane), Y: 1, Z: 1}, vaeBlock1D, vaeKernelBlock*8,
		&out, &x, &gamma, &channels, &planeArg, &applySiLU)
}

func (s *VAEDecoderCUDASession) addInPlace(state *device.State, out, a, b driver.DevicePtr, elements int) error {
	count := uint32(elements)
	return launchVAEKernel(state, s.kernels.add, vaeElementwiseGrid(elements), vaeBlock1D, 0,
		&a, &b, &out, &count)
}

// updateCache: reference temporal_cache_update semantics on device; the new
// cache lands in the opposite slot so in-flight consumers of the prior cache
// stay valid on the stream.
func (s *VAEDecoderCUDASession) updateCache(state *device.State, owner string, prior vaeDeviceCache, x driver.DevicePtr, c, frames, spatial int, replicatePrefix bool) (vaeDeviceCache, error) {
	nextFrames := min(2, frames)
	mode := uint32(0)
	if replicatePrefix {
		nextFrames = frames + 1
		mode = 1
	} else if frames < 2 && prior.initialized && prior.frames > 0 {
		nextFrames = frames + 1
	}
	slot := 1 - prior.slot
	out, err := s.buffer(state, fmt.Sprintf("cache_%s_%d", owner, slot), c*nextFrames*spatial)
	if err != nil {
		return vaeDeviceCache{}, err
	}
	elements := c * nextFrames * spatial
	channels, currentFrames, priorFrames := uint32(c), uint32(frames), uint32(prior.frames)
	outFrames, spatialArg, count := uint32(nextFrames), uint32(spatial), uint32(elements)
	if err := launchVAEKernel(state, s.kernels.cacheUpdate, vaeElementwiseGrid(elements), vaeBlock1D, 0,
		&out, &x, &prior.ptr,
		&channels, &currentFrames, &priorFrames, &outFrames, &spatialArg, &mode, &count); err != nil {
		return vaeDeviceCache{}, err
	}
	return vaeDeviceCache{ptr: out, frames: nextFrames, slot: slot, initialized: true}, nil
}

// runOp executes one op on one device chunk [cIn][frames][h][w], mutating the
// op's temporal state exactly as the host runVAEOp does. actName is the
// ping-pong activation target for this op's output.
func (s *VAEDecoderCUDASession) runOp(state *device.State, opIndex, chunkIndex int, opState *vaeDeviceOpState, x driver.DevicePtr, actName string, frames, h, w int) (driver.DevicePtr, int, int, int, error) {
	op := s.ops[opIndex]
	spatial := h * w
	c := op.cIn
	cachedConv := func(out, input driver.DevicePtr, cache *vaeDeviceCache, weight, bias driver.DevicePtr, cOut, kt, kh, kw int) error {
		if err := s.conv3d(state, out, input, cache.ptr, cache.frames, weight, bias, c, cOut, frames, h, w, kt, kh, kw); err != nil {
			return err
		}
		if kt/2 != 0 {
			next, err := s.updateCache(state, fmt.Sprintf("op%d_c0", opIndex), *cache, input, c, frames, spatial, false)
			if err != nil {
				return err
			}
			*cache = next
		}
		return nil
	}
	switch op.kind {
	case vaeOpPointwise, vaeOpConv:
		weight, bias := op.values[0], op.values[1]
		kt := 1
		if op.kind == vaeOpConv {
			kt = 3
		}
		out, err := s.buffer(state, actName, op.cOut*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := cachedConv(out, x, &opState.cache0, weight, bias, op.cOut, kt, kt, kt); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	case vaeOpResidual:
		gamma0, w0, b0 := op.values[0], op.values[1], op.values[2]
		gamma1, w1, b1 := op.values[3], op.values[4], op.values[5]
		n0, err := s.buffer(state, "work_a", c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.rmsNorm(state, n0, x, gamma0, c, frames*spatial, true); err != nil {
			return 0, 0, 0, 0, err
		}
		h0, err := s.buffer(state, "work_b", op.cOut*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := cachedConv(h0, n0, &opState.cache0, w0, b0, op.cOut, 3, 3, 3); err != nil {
			return 0, 0, 0, 0, err
		}
		n1, err := s.buffer(state, "work_a", op.cOut*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.rmsNorm(state, n1, h0, gamma1, op.cOut, frames*spatial, true); err != nil {
			return 0, 0, 0, 0, err
		}
		out, err := s.buffer(state, actName, op.cOut*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.conv3d(state, out, n1, opState.cache1.ptr, opState.cache1.frames, w1, b1, op.cOut, op.cOut, frames, h, w, 3, 3, 3); err != nil {
			return 0, 0, 0, 0, err
		}
		nextCache1, err := s.updateCache(state, fmt.Sprintf("op%d_c1", opIndex), opState.cache1, n1, op.cOut, frames, spatial, false)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		opState.cache1 = nextCache1
		if op.cIn == op.cOut {
			if err := s.addInPlace(state, out, out, x, op.cOut*frames*spatial); err != nil {
				return 0, 0, 0, 0, err
			}
			return out, frames, h, w, nil
		}
		shortcut, err := s.buffer(state, "work_b", op.cOut*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.conv3d(state, shortcut, x, 0, 0, op.values[6], op.values[7], c, op.cOut, frames, h, w, 1, 1, 1); err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.addInPlace(state, out, out, shortcut, op.cOut*frames*spatial); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	case vaeOpAttention:
		gamma, qkvW, qkvB := op.values[0], op.values[1], op.values[2]
		projW, projB := op.values[3], op.values[4]
		norm, err := s.buffer(state, "work_a", c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.rmsNorm(state, norm, x, gamma, c, frames*spatial, false); err != nil {
			return 0, 0, 0, 0, err
		}
		qkv, err := s.buffer(state, "work_qkv", 3*c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.conv3d(state, qkv, norm, 0, 0, qkvW, qkvB, c, 3*c, frames, h, w, 1, 1, 1); err != nil {
			return 0, 0, 0, 0, err
		}
		// scores (8-aligned) + one double per thread (kernel shared layout).
		shared := uint32((spatial*4+7)/8*8 + vaeKernelBlock*8)
		channels, groups, frame := uint32(c), uint32(frames), uint32(spatial)
		scale := float32(1 / math.Sqrt(float64(c)))
		if err := launchVAEKernel(state, s.kernels.attention,
			driver.Dim3{X: uint32(frames * spatial), Y: 1, Z: 1}, vaeBlock1D, shared,
			&qkv, &norm, &channels, &groups, &frame, &scale); err != nil {
			return 0, 0, 0, 0, err
		}
		out, err := s.buffer(state, actName, c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.conv3d(state, out, norm, 0, 0, projW, projB, c, c, frames, h, w, 1, 1, 1); err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.addInPlace(state, out, out, x, c*frames*spatial); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	case vaeOpDownsample2D:
		out, err := s.buffer(state, actName, op.cOut*frames*(h/2)*(w/2))
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.downsample2D(state, out, x, op.values[0], op.values[1], c, frames, h, w); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h / 2, w / 2, nil
	case vaeOpDownsample3D:
		outH, outW := h/2, w/2
		spatial := outH * outW
		spatialOut, err := s.buffer(state, "work_a", op.cOut*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.downsample2D(state, spatialOut, x, op.values[0], op.values[1], c, frames, h, w); err != nil {
			return 0, 0, 0, 0, err
		}
		prior := opState.cache0
		next, err := s.updateCache(state, fmt.Sprintf("op%d_down", opIndex), opState.cache0, spatialOut, op.cOut, frames, spatial, false)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		opState.cache0 = next
		if chunkIndex == 0 {
			out, err := s.buffer(state, actName, op.cOut*frames*spatial)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			if err := state.Driver.MemcpyDtoD(out, spatialOut, uint64(op.cOut*frames*spatial*4)); err != nil {
				return 0, 0, 0, 0, err
			}
			return out, frames, outH, outW, nil
		}
		outFrames := (frames-1)/2 + 1
		out, err := s.buffer(state, actName, op.cOut*outFrames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.temporalDownsample(state, out, spatialOut, prior.ptr, op.values[2], op.values[3], op.cOut, frames, prior.frames, spatial, outFrames); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, outFrames, outH, outW, nil
	case vaeOpUpsample2D:
		out, err := s.buffer(state, actName, op.cOut*frames*vaeSpatialScale*h*vaeSpatialScale*w)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.upsample2D(state, out, x, op.values[0], op.values[1], c, op.cOut, frames, h, w); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, vaeSpatialScale * h, vaeSpatialScale * w, nil
	case vaeOpUpsample3D:
		timeW, timeB := op.values[0], op.values[1]
		resampleW, resampleB := op.values[2], op.values[3]
		spatialInput, spatialFrames := x, frames
		if chunkIndex == 0 {
			opState.cache0 = vaeDeviceCache{initialized: true, rep: true}
		} else {
			replicatePrefix := opState.cache0.rep && frames < 2
			nextCache, err := s.updateCache(state, fmt.Sprintf("op%d_c0", opIndex), opState.cache0, x, c, frames, spatial, replicatePrefix)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			cachePtr, cacheFrames := opState.cache0.ptr, opState.cache0.frames
			if opState.cache0.rep {
				cachePtr, cacheFrames = 0, 0
			}
			timeOut, err := s.buffer(state, "work_a", 2*c*frames*spatial)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			if err := s.conv3d(state, timeOut, x, cachePtr, cacheFrames, timeW, timeB, c, 2*c, frames, h, w, 3, 1, 1); err != nil {
				return 0, 0, 0, 0, err
			}
			spatialFrames = 2 * frames
			interleaved, err := s.buffer(state, "work_b", c*spatialFrames*spatial)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			elements := 2 * c * frames * spatial
			channels, framesArg, spatialArg, count := uint32(c), uint32(frames), uint32(spatial), uint32(elements)
			if err := launchVAEKernel(state, s.kernels.interleave, vaeElementwiseGrid(elements), vaeBlock1D, 0,
				&interleaved, &timeOut, &channels, &framesArg, &spatialArg, &count); err != nil {
				return 0, 0, 0, 0, err
			}
			spatialInput = interleaved
			opState.cache0 = nextCache
		}
		out, err := s.buffer(state, actName, op.cOut*spatialFrames*vaeSpatialScale*h*vaeSpatialScale*w)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.upsample2D(state, out, spatialInput, resampleW, resampleB, c, op.cOut, spatialFrames, h, w); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, spatialFrames, vaeSpatialScale * h, vaeSpatialScale * w, nil
	case vaeOpHead:
		gamma, weight, bias := op.values[0], op.values[1], op.values[2]
		norm, err := s.buffer(state, "work_a", c*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := s.rmsNorm(state, norm, x, gamma, c, frames*spatial, true); err != nil {
			return 0, 0, 0, 0, err
		}
		out, err := s.buffer(state, actName, op.cOut*frames*spatial)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if err := cachedConv(out, norm, &opState.cache0, weight, bias, op.cOut, 3, 3, 3); err != nil {
			return 0, 0, 0, 0, err
		}
		return out, frames, h, w, nil
	}
	return 0, 0, 0, 0, fmt.Errorf("vae cuda decode: unsupported op %s", op.kind)
}

func (s *VAEDecoderCUDASession) upsample2D(state *device.State, out, x, weight, bias driver.DevicePtr, cIn, cOut, frames, h, w int) error {
	outH, outW := vaeSpatialScale*h, vaeSpatialScale*w
	elements := cOut * frames * outH * outW
	cInArg, cOutArg, framesArg := uint32(cIn), uint32(cOut), uint32(frames)
	heightArg, widthArg := uint32(h), uint32(w)
	outHArg, outWArg, count := uint32(outH), uint32(outW), uint32(elements)
	return launchVAEKernel(state, s.kernels.upsample2d, vaeElementwiseGrid(elements), vaeBlock1D, 0,
		&out, &x, &weight, &bias,
		&cInArg, &cOutArg, &framesArg, &heightArg, &widthArg, &outHArg, &outWArg, &count)
}

func (s *VAEDecoderCUDASession) downsample2D(state *device.State, out, x, weight, bias driver.DevicePtr, channels, frames, h, w int) error {
	outH, outW := h/2, w/2
	elements := channels * frames * outH * outW
	channelsArg, framesArg := uint32(channels), uint32(frames)
	heightArg, widthArg := uint32(h), uint32(w)
	outHArg, outWArg, count := uint32(outH), uint32(outW), uint32(elements)
	return launchVAEKernel(state, s.kernels.downsample2d, vaeElementwiseGrid(elements), vaeBlock1D, 0,
		&out, &x, &weight, &bias, &channelsArg, &framesArg, &heightArg, &widthArg, &outHArg, &outWArg, &count)
}

func (s *VAEDecoderCUDASession) temporalDownsample(state *device.State, out, x, prior, weight, bias driver.DevicePtr, channels, frames, priorFrames, spatial, outFrames int) error {
	elements := channels * outFrames * spatial
	channelsArg, framesArg, priorFramesArg := uint32(channels), uint32(frames), uint32(priorFrames)
	spatialArg, outFramesArg, count := uint32(spatial), uint32(outFrames), uint32(elements)
	return launchVAEKernel(state, s.kernels.temporalDownsample, vaeElementwiseGrid(elements), vaeBlock1D, 0,
		&out, &x, &prior, &weight, &bias, &channelsArg, &framesArg, &priorFramesArg, &spatialArg, &outFramesArg, &count)
}

// Decode streams the plan over the latent volume on CUDA: host denorm
// (z*std+mean) per chunk, upload, the op chain with device temporal caches,
// clamp to [-1,1], then per-frame device-to-host emission through the sink.
func (s *VAEDecoderCUDASession) Decode(stats VAELatentStats, z []float32, latentFrames, latentH, latentW int, sink VideoFrameSink) (VAEDecodeStats, error) {
	var decodeStats VAEDecodeStats
	plan := s.Plan
	if sink == nil {
		return decodeStats, fmt.Errorf("vae cuda decode: nil frame sink")
	}
	if len(stats.Mean) != plan.ZDim || len(stats.Std) != plan.ZDim {
		return decodeStats, fmt.Errorf("vae cuda decode: latent stats mean=%d std=%d want %d", len(stats.Mean), len(stats.Std), plan.ZDim)
	}
	for channel, std := range stats.Std {
		if std <= 0 {
			return decodeStats, fmt.Errorf("vae cuda decode: nonpositive std for channel %d", channel)
		}
	}
	spatial := latentH * latentW
	if latentFrames <= 0 || spatial <= 0 || len(z) != plan.ZDim*latentFrames*spatial {
		return decodeStats, fmt.Errorf("vae cuda decode: latent len=%d want %d", len(z), plan.ZDim*latentFrames*spatial)
	}
	outputChannels, totalFrames, outputH, outputW, err := VideoDecodeShape(plan, latentFrames, latentH, latentW)
	if err != nil {
		return decodeStats, err
	}
	started := time.Now()
	decodeStats.Ops = len(s.ops)
	decodeStats.LatentFrames = latentFrames
	decodeStats.WeightBytesRead = plan.UsedWeightBytes
	decodeStats.MaxOpWeightBytes = plan.LargestOpWeightBytes
	decodeStats.OutputChannels, decodeStats.OutputHeight, decodeStats.OutputWidth = outputChannels, outputH, outputW
	decodeStats.Engine = "cuda_streamed_chunks"
	states := make([]vaeDeviceOpState, len(s.ops))
	staging := make([]float32, plan.ZDim*spatial)
	var frameScratch []float32
	frameIndex := 0
	for chunkIndex := 0; chunkIndex < latentFrames; chunkIndex++ {
		if err := s.worker.Do(s.ctx, func(state *device.State) error {
			// Denormalized chunk [z][1][h][w] (host affine, reference upload).
			for ch := 0; ch < plan.ZDim; ch++ {
				mean, std := stats.Mean[ch], stats.Std[ch]
				src := z[(ch*latentFrames+chunkIndex)*spatial:]
				dst := staging[ch*spatial:]
				for pos := 0; pos < spatial; pos++ {
					dst[pos] = src[pos]*std + mean
				}
			}
			x, err := s.buffer(state, "act_0", plan.ZDim*spatial)
			if err != nil {
				return err
			}
			if err := state.Driver.MemcpyHtoD(x, driver.Bytes(staging)); err != nil {
				return err
			}
			frames, h, w := 1, latentH, latentW
			actIndex := 0
			for opIndex := range s.ops {
				actIndex = 1 - actIndex
				opStarted := time.Now()
				x, frames, h, w, err = s.runOp(state, opIndex, chunkIndex, &states[opIndex], x, fmt.Sprintf("act_%d", actIndex), frames, h, w)
				if err != nil {
					return fmt.Errorf("vae cuda decode %s chunk %d: %w", s.ops[opIndex].prefix, chunkIndex, err)
				}
				if s.profile != nil {
					if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
						return err
					}
					s.profile[s.ops[opIndex].prefix] += time.Since(opStarted)
				}
			}
			if h != outputH || w != outputW {
				return fmt.Errorf("vae cuda decode chunk %d output %dx%d, want %dx%d", chunkIndex, w, h, outputW, outputH)
			}
			elements := outputChannels * frames * h * w
			minimum, maximum, count := float32(-1), float32(1), uint32(elements)
			if err := launchVAEKernel(state, s.kernels.clamp, vaeElementwiseGrid(elements), vaeBlock1D, 0,
				&x, &x, &minimum, &maximum, &count); err != nil {
				return err
			}
			if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
				return err
			}
			chunkSpatial := h * w
			frameElements := outputChannels * chunkSpatial
			if cap(frameScratch) < frameElements {
				frameScratch = make([]float32, frameElements)
			}
			frame := frameScratch[:frameElements]
			for chunkFrame := 0; chunkFrame < frames; chunkFrame++ {
				for ch := 0; ch < outputChannels; ch++ {
					source := x + driver.DevicePtr((ch*frames+chunkFrame)*chunkSpatial*4)
					if err := state.Driver.MemcpyDtoH(driver.Bytes(frame[ch*chunkSpatial:(ch+1)*chunkSpatial]), source); err != nil {
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
	if frameIndex != totalFrames {
		return decodeStats, fmt.Errorf("vae cuda decode produced %d frames, want %d", frameIndex, totalFrames)
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
		errs = append(errs, s.worker.Do(s.ctx, func(state *device.State) error {
			var innerErrs []error
			innerErrs = append(innerErrs, state.Driver.StreamSynchronize(state.Stream))
			for _, op := range s.ops {
				for _, pointer := range op.values {
					if pointer != 0 {
						innerErrs = append(innerErrs, state.Driver.MemFree(pointer))
					}
				}
			}
			s.ops = nil
			for name, buffer := range s.buffers {
				if buffer.ptr != 0 {
					innerErrs = append(innerErrs, state.Driver.MemFree(buffer.ptr))
				}
				delete(s.buffers, name)
			}
			if s.module != 0 {
				innerErrs = append(innerErrs, state.Driver.ModuleUnload(s.module))
				s.module = 0
			}
			return errors.Join(innerErrs...)
		}))
		errs = append(errs, s.worker.Close())
		s.worker = nil
	}
	return errors.Join(errs...)
}
