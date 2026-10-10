package latentvideo

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/binaryschema"
	"overgo/internal/cuda/cudnn"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/media"
	"overgo/internal/tensor"
)

// The kernels a session's library convolutions run beside the library: one
// stages a chunk behind the frames before it in half precision, one keeps the
// last of the staged frames for the next chunk, one writes what the library
// convolved in single precision with the bias added, and one packs a filter
// to half precision once.
const (
	libraryStageKernel = "vae_causal_stage_f16"
	libraryTailKernel  = "vae_tail_frames_f16"
	libraryWidenKernel = "vae_widen_bias_f32"
	libraryPackKernel  = "f32_to_f16"
)

// The slots of a session's program workspace the library's convolutions use:
// the staged input, what the library wrote, and its algorithm's workspace.
const (
	libraryStageSlot = iota + tensor.TripleExtent
	libraryConvolvedSlot
	libraryWorkspaceSlot
)

// halfBytes is the size of a half-precision value.
const halfBytes = binaryschema.Uint32Bytes / tensor.PairedExtent

// vaeLibrary is the cuDNN context a codec session convolves through for the
// requests that take it: the library and a context on the session's stream, the kernels
// that hand it its input and take its output, the plan of each shape it has
// convolved, the shapes it refused, which the session's own kernel convolves,
// and the half-precision copy of each filter it has convolved by.
//
// The library computes on the device's tensor units, which take half
// precision operands, so a convolution's input and filter are handed to it in
// half precision and its sums are kept in single precision.
type vaeLibrary struct {
	library                  *cudnn.Library
	stage, tail, widen, pack driver.Function
	plans                    map[cudnn.Shape]cudnn.Convolution
	refused                  map[cudnn.Shape]bool
	filters                  map[driver.DevicePtr]driver.DevicePtr
}

// libraryClipValues is the fewest values of one channel, frames by rows by
// columns, a request's clip holds for it to convolve through the library.
// Setting the library up costs about a second, once in a process; it then
// saves about two tenths of a second on each million values of a round trip.
// A clip of this many repays that in two requests and is ahead over three,
// the rule the library was adopted by; a served clip of 21 frames is twice
// this and repays it in its own. A picture or a smaller clip is sooner done
// by the kernels, and loads nothing (docs/verification/video-codec-convolution-paths.txt).
const libraryClipValues = 1 << 22

// libraryRoute is what a session was told of the library's path and what it
// found of it: whether it was given the path, the fewest values of a clip
// that takes it, whether this machine turned out to have no library, and
// whether the request in hand takes it.
//
// The two paths do not give the same values. The kernels sum single precision
// operands and are held to the host reference value for value; the library
// sums half precision ones, as the autoencoder's own code does on a device,
// and is held to that code's frames within the levels a served frame is.
type libraryRoute struct {
	adopted, absent, taken bool
	from                   int
}

// adoptLibrary gives the session the library's path for a clip of at least
// values values of one channel.
func (s *vaeDecoderCUDASession) adoptLibrary(values int) {
	s.route = libraryRoute{adopted: true, from: values}
}

// routeClip decides what convolves the request for a clip of values values of
// one channel, and loads the library at the first clip that takes it. A
// machine without the library keeps the kernels, which is no failure: the
// library is not a thing a user must install.
func (s *vaeDecoderCUDASession) routeClip(ctx context.Context, values int) error {
	s.route.taken = false
	if !s.route.adopted || s.route.absent || values < s.route.from {
		return nil
	}
	if s.library == nil {
		if err := s.useLibrary(ctx); err != nil {
			return err
		}
	}
	s.route.taken = s.library != nil
	return nil
}

// useLibrary loads cuDNN and makes the session's context on its stream. A
// machine without the library leaves the session none, and the route says so;
// a library that loads and then gives no context is a failure.
func (s *vaeDecoderCUDASession) useLibrary(ctx context.Context) error {
	return s.worker.Do(ctx, func(state *device.State) error {
		library, err := cudnn.Open(state.Stream)
		if s.route.absent = errors.Is(err, cudnn.ErrAbsent); err != nil {
			if s.route.absent {
				return nil
			}
			return err
		}
		held := &vaeLibrary{
			library: library, plans: map[cudnn.Shape]cudnn.Convolution{}, refused: map[cudnn.Shape]bool{},
			filters: map[driver.DevicePtr]driver.DevicePtr{},
		}
		for _, entry := range []struct {
			name     string
			function *driver.Function
		}{{libraryStageKernel, &held.stage}, {libraryTailKernel, &held.tail}, {libraryWidenKernel, &held.widen}, {libraryPackKernel, &held.pack}} {
			if *entry.function, err = state.Driver.ModuleFunction(s.module, entry.name); err != nil {
				return errors.Join(fmt.Errorf("vae cuda session kernel %s: %w", entry.name, err), library.Close())
			}
		}
		s.library = held
		return nil
	})
}

// closeLibrary frees the filters' half-precision copies and releases the
// plans, the context and the library.
func (s *vaeDecoderCUDASession) closeLibrary(state *device.State) error {
	held := s.library
	if held == nil {
		return nil
	}
	s.library = nil
	var failed error
	for _, filter := range held.filters {
		failed = errors.Join(failed, state.Driver.MemFree(filter))
	}
	for _, plan := range held.plans {
		failed = errors.Join(failed, held.library.Release(plan))
	}
	return errors.Join(failed, held.library.Close())
}

// halves is a buffer of the session's that holds values half-precision
// values: a buffer is counted in single-precision values, two halves in each.
func (s *vaeDecoderCUDASession) halves(state *device.State, key media.CodecWorkspaceKey, values int) (driver.DevicePtr, error) {
	return s.buffer(state, key, (values+tensor.SingletonExtent)/tensor.PairedExtent)
}

// libraryConv3d is conv3d through the library, and reports false, having
// done nothing, for a shape the library does not convolve, which the caller's
// kernel then does. The chunk is staged behind the frames before it, zero
// where the cache holds fewer than the kernel reaches back, as the kernel
// reads them; the library convolves the staged frames with no padding in
// time, so it writes one frame a frame of the chunk; and its output is
// widened with the bias added in single precision, where the kernel begins
// each sum from the bias. It returns the staged frames, which stand until the
// session's next convolution, and how many they are.
func (s *vaeDecoderCUDASession) libraryConv3d(state *device.State, out, x driver.DevicePtr, cache vaeDeviceCache, weight, bias driver.DevicePtr, cIn, cOut, frames, h, w, kt, kh, kw int) (bool, driver.DevicePtr, int, error) {
	held, spatial, staged := s.library, h*w, frames+kt-tensor.SingletonExtent
	shape := cudnn.Shape{
		In: cIn, Out: cOut, Frames: staged, Height: h, Width: w,
		Kernel: [tensor.TripleExtent]int{kt, kh, kw},
		Stride: [tensor.TripleExtent]int{tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent},
		Pad:    [tensor.TripleExtent]int{tensor.FirstOffset, kh / tensor.PairedExtent, kw / tensor.PairedExtent},
	}
	if held.refused[shape] {
		return false, 0, 0, nil
	}
	plan, planned := held.plans[shape]
	if !planned {
		var err error
		if plan, err = held.library.Plan(shape, cudnn.DataF16, cudnn.DataF32); err != nil {
			held.refused[shape] = true
			return false, 0, 0, nil
		}
		held.plans[shape] = plan
	}
	filter, packed := held.filters[weight]
	if !packed {
		values := cOut * cIn * kt * kh * kw
		var err error
		if filter, err = state.Driver.MemAlloc(uint64(values) * halfBytes); err != nil {
			return true, 0, 0, err
		}
		held.filters[weight] = filter
		count := uint32(values)
		if err := launchVAEKernel(state, held.pack, kernel.ElementwiseGrid(values), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(), &weight, &filter, &count); err != nil {
			return true, 0, 0, err
		}
	}
	stagedValues, convolvedValues := cIn*staged*spatial, cOut*frames*spatial
	input, err := s.halves(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, libraryStageSlot), stagedValues)
	if err != nil {
		return true, 0, 0, err
	}
	convolved, err := s.halves(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, libraryConvolvedSlot), convolvedValues)
	if err != nil {
		return true, 0, 0, err
	}
	var workspace driver.DevicePtr
	if plan.Workspace != 0 {
		if workspace, err = s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, libraryWorkspaceSlot), int(plan.Workspace/binaryschema.Uint32Bytes)+tensor.SingletonExtent); err != nil {
			return true, 0, 0, err
		}
	}
	// A chunk with no frames before it has no cache to read, and the kernel reads none.
	before, kept, half := cache.ptr, uint32(cache.frames), uint32(tensor.FirstOffset)
	if cache.frames == 0 {
		before = x
	}
	if cache.half {
		half = uint32(tensor.SingletonExtent)
	}
	chunk, total, plane, stagedCount := uint32(frames), uint32(staged), uint32(spatial), uint32(stagedValues)
	if err := launchVAEKernel(state, held.stage, kernel.ElementwiseGrid(stagedValues), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&input, &x, &before, &chunk, &kept, &half, &total, &plane, &stagedCount); err != nil {
		return true, 0, 0, err
	}
	if err := held.library.Forward(plan, input, filter, workspace, convolved); err != nil {
		return true, 0, 0, err
	}
	channelPlane, count := uint32(frames*spatial), uint32(convolvedValues)
	return true, input, staged, launchVAEKernel(state, held.widen, kernel.ElementwiseGrid(convolvedValues), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&out, &convolved, &bias, &channelPlane, &count)
}

// libraryCachedConv convolves input through the library and keeps the frames
// the next chunk's convolution reads before its own. They are the last of
// what was staged, as many as the kernel reaches back or as there are yet, so
// they are kept as staged: in half precision, and once, where the kernels'
// cache is a single-precision copy written beside the one it replaces. It
// reports false, having done nothing, when the request does not take the
// library or the library does not convolve the shape.
func (s *vaeDecoderCUDASession) libraryCachedConv(state *device.State, out, input driver.DevicePtr, cache *vaeDeviceCache, owner media.CodecWorkspaceKey, weight, bias driver.DevicePtr, cIn, cOut, frames, h, w, kt, kh, kw int) (bool, error) {
	if !s.route.taken {
		return false, nil
	}
	convolved, staged, stagedFrames, err := s.libraryConv3d(state, out, input, *cache, weight, bias, cIn, cOut, frames, h, w, kt, kh, kw)
	if !convolved || err != nil || kt == tensor.SingletonExtent {
		return convolved, err
	}
	spatial, keep := h*w, min(kt-tensor.SingletonExtent, cache.frames+frames)
	kept, err := s.halves(state, owner, cIn*keep*spatial)
	if err != nil {
		return true, err
	}
	from, frameCount, plane, count := uint32(stagedFrames), uint32(keep), uint32(spatial), uint32(cIn*keep*spatial)
	if err := launchVAEKernel(state, s.library.tail, kernel.ElementwiseGrid(cIn*keep*spatial), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&kept, &staged, &from, &frameCount, &plane, &count); err != nil {
		return true, err
	}
	*cache = vaeDeviceCache{ptr: kept, frames: keep, initialized: true, half: true}
	return true, nil
}
