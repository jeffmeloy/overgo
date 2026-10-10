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
// stages a chunk behind the frames before it in half precision, one stages
// each frame enlarged or with zeros past its edge, one keeps the last of the
// staged frames for the next chunk, one writes what the library convolved in
// single precision with the bias added, and one packs a filter to half
// precision once.
const (
	libraryStageKernel   = "vae_causal_stage_f16"
	librarySpatialKernel = "vae_spatial_stage_f16"
	libraryTailKernel    = "vae_tail_frames_f16"
	libraryWidenKernel   = "vae_widen_bias_f32"
	libraryPackKernel    = "f32_to_f16"
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
	library                           *cudnn.Library
	stage, spatial, tail, widen, pack driver.Function
	plans                             map[cudnn.Shape]cudnn.Convolution
	refused                           map[cudnn.Shape]bool
	filters                           map[driver.DevicePtr]driver.DevicePtr
}

// libraryClipValues is the fewest values of one channel, frames by rows by
// columns, a request's clip holds for it to convolve through the library.
// It was set when the library saved two tenths of a second on each million
// values of a round trip and cost a second to set up, so that a clip of this
// many repaid it in two requests. It now saves nearly twice that, and a clip
// of this many repays it in its own request. Whether a smaller clip or a
// picture should take it too is not decided here: they keep the kernels and
// load nothing (docs/verification/video-codec-convolution-paths.txt).
const libraryClipValues = 1 << 22

// libraryRoute is what a session was told of the library's path and what it
// found of it: whether it was given the path, the fewest values of a clip
// that takes it, whether this machine turned out to have no library, whether
// the request in hand takes it, and how many of that request's convolutions
// each path has done.
//
// The two paths do not give the same values. The kernels sum single precision
// operands and are held to the host reference value for value; the library
// sums half precision ones, as the autoencoder's own code does on a device,
// and is held to that code's frames within the levels a served frame is.
type libraryRoute struct {
	adopted, absent, taken bool
	from                   int
	byKernels, byLibrary   int
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
	s.route.taken, s.route.byKernels, s.route.byLibrary = false, 0, 0
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
		}{{libraryStageKernel, &held.stage}, {librarySpatialKernel, &held.spatial}, {libraryTailKernel, &held.tail}, {libraryWidenKernel, &held.widen}, {libraryPackKernel, &held.pack}} {
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

// libraryConvolve convolves through the library what stage writes as the
// input of shape, and writes the result to out in single precision with the
// bias added, where a kernel begins each sum from the bias: plane is how many
// values of one channel the convolution writes. It reports false, having done
// nothing, for a shape the library does not convolve, which the caller's
// kernel then does. It returns what was staged, which stands until the
// session's next convolution.
func (s *vaeDecoderCUDASession) libraryConvolve(state *device.State, out driver.DevicePtr, shape cudnn.Shape, plane int, weight, bias driver.DevicePtr, stage func(input driver.DevicePtr, values int) error) (bool, driver.DevicePtr, error) {
	held := s.library
	if held.refused[shape] {
		return false, 0, nil
	}
	plan, planned := held.plans[shape]
	if !planned {
		var err error
		if plan, err = held.library.Plan(shape, cudnn.DataF16, cudnn.DataF32); err != nil {
			held.refused[shape] = true
			return false, 0, nil
		}
		held.plans[shape] = plan
	}
	filter, packed := held.filters[weight]
	if !packed {
		values := shape.Out * shape.In
		for _, extent := range shape.Kernel {
			values *= extent
		}
		var err error
		if filter, err = state.Driver.MemAlloc(uint64(values) * halfBytes); err != nil {
			return true, 0, err
		}
		held.filters[weight] = filter
		count := uint32(values)
		if err := launchVAEKernel(state, held.pack, kernel.ElementwiseGrid(values), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(), &weight, &filter, &count); err != nil {
			return true, 0, err
		}
	}
	stagedValues, convolvedValues := shape.In*shape.Frames*shape.Height*shape.Width, shape.Out*plane
	input, err := s.halves(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, libraryStageSlot), stagedValues)
	if err != nil {
		return true, 0, err
	}
	convolved, err := s.halves(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, libraryConvolvedSlot), convolvedValues)
	if err != nil {
		return true, 0, err
	}
	var workspace driver.DevicePtr
	if plan.Workspace != 0 {
		if workspace, err = s.buffer(state, media.ProgramCodecWorkspace(media.CodecWorkspaceScratch, libraryWorkspaceSlot), int(plan.Workspace/binaryschema.Uint32Bytes)+tensor.SingletonExtent); err != nil {
			return true, 0, err
		}
	}
	if err := stage(input, stagedValues); err != nil {
		return true, 0, err
	}
	if err := held.library.Forward(plan, input, filter, workspace, convolved); err != nil {
		return true, 0, err
	}
	s.route.byLibrary++
	channelPlane, count := uint32(plane), uint32(convolvedValues)
	return true, input, launchVAEKernel(state, held.widen, kernel.ElementwiseGrid(convolvedValues), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&out, &convolved, &bias, &channelPlane, &count)
}

// stageCausal writes a chunk behind the frames before it as the library's
// input, staged frames in all: the cache's, and zero where it holds fewer than
// the chunk is staged behind, as a kernel reads them.
func (s *vaeDecoderCUDASession) stageCausal(state *device.State, input, x driver.DevicePtr, cache vaeDeviceCache, frames, staged, spatial, values int) error {
	// A chunk with no frames before it has no cache to read, and the kernel reads none.
	before, kept, half := cache.ptr, uint32(cache.frames), uint32(tensor.FirstOffset)
	if cache.frames == 0 {
		before = x
	}
	if cache.half {
		half = uint32(tensor.SingletonExtent)
	}
	chunk, total, plane, count := uint32(frames), uint32(staged), uint32(spatial), uint32(values)
	return launchVAEKernel(state, s.library.stage, kernel.ElementwiseGrid(values), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
		&input, &x, &before, &chunk, &kept, &half, &total, &plane, &count)
}

// libraryConv3d is conv3d through the library. The chunk is staged behind the
// frames before it and the library convolves the staged frames with no
// padding in time, so it writes one frame a frame of the chunk. It returns the
// staged frames and how many they are.
func (s *vaeDecoderCUDASession) libraryConv3d(state *device.State, out, x driver.DevicePtr, cache vaeDeviceCache, weight, bias driver.DevicePtr, cIn, cOut, frames, h, w, kt, kh, kw int) (bool, driver.DevicePtr, int, error) {
	spatial, staged := h*w, frames+kt-tensor.SingletonExtent
	convolved, input, err := s.libraryConvolve(state, out, cudnn.Shape{
		In: cIn, Out: cOut, Frames: staged, Height: h, Width: w,
		Kernel: [tensor.TripleExtent]int{kt, kh, kw},
		Stride: [tensor.TripleExtent]int{tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent},
		Pad:    [tensor.TripleExtent]int{tensor.FirstOffset, kh / tensor.PairedExtent, kw / tensor.PairedExtent},
	}, frames*spatial, weight, bias, func(input driver.DevicePtr, values int) error {
		return s.stageCausal(state, input, x, cache, frames, staged, spatial, values)
	})
	return convolved, input, staged, err
}

// libraryResample is a convolution of three by three over each frame's rows
// and columns through the library, of the two kinds a kernel resamples as it
// convolves. Enlarging, the frame is staged at twice its extents, each value
// at four positions, and convolved at stride one with a position of zeros
// around it (upsample2D). Otherwise it is staged with a row of zeros below it
// and a column to its right and convolved at stride two with none around it,
// which writes a frame of half its extents (downsample2D); a frame of an odd
// extent has no such half, and is left to the kernel.
func (s *vaeDecoderCUDASession) libraryResample(state *device.State, out, x, weight, bias driver.DevicePtr, cIn, cOut, frames, h, w int, enlarge bool) (bool, error) {
	scale := media.CodecUpsampleSpatial.SpatialScale()
	if !enlarge && (h%scale != 0 || w%scale != 0) {
		return false, nil
	}
	shape := cudnn.Shape{
		In: cIn, Out: cOut, Frames: frames, Height: h + tensor.SingletonExtent, Width: w + tensor.SingletonExtent,
		Kernel: [tensor.TripleExtent]int{tensor.SingletonExtent, tensor.TripleExtent, tensor.TripleExtent},
		Stride: [tensor.TripleExtent]int{tensor.SingletonExtent, scale, scale},
	}
	plane, each := frames*(h/scale)*(w/scale), tensor.SingletonExtent
	if enlarge {
		shape.Height, shape.Width, each = scale*h, scale*w, scale
		shape.Stride = [tensor.TripleExtent]int{tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent}
		shape.Pad = [tensor.TripleExtent]int{tensor.FirstOffset, tensor.SingletonExtent, tensor.SingletonExtent}
		plane = frames * shape.Height * shape.Width
	}
	convolved, _, err := s.libraryConvolve(state, out, shape, plane, weight, bias, func(input driver.DevicePtr, values int) error {
		inH, inW, outH, outW, every, count := uint32(h), uint32(w), uint32(shape.Height), uint32(shape.Width), uint32(each), uint32(values)
		return launchVAEKernel(state, s.library.spatial, kernel.ElementwiseGrid(values), kernel.DefaultBlock1D(), kernel.NoSharedMemoryBytes(),
			&input, &x, &inH, &inW, &outH, &outW, &every, &count)
	})
	return convolved, err
}

// libraryTemporalDownsample is temporalDownsample through the library: the
// chunk staged behind the frame before it and convolved over three frames at
// stride two, which writes a frame for every two of an even chunk. A chunk of
// an odd count ends on a window the kernel reads short, and is left to it.
func (s *vaeDecoderCUDASession) libraryTemporalDownsample(state *device.State, out, x driver.DevicePtr, prior vaeDeviceCache, weight, bias driver.DevicePtr, channels, frames, h, w int) (bool, error) {
	scale := media.CodecDownsampleSpatiotemporal.TemporalScale()
	if frames%scale != 0 {
		return false, nil
	}
	spatial, staged := h*w, frames+tensor.SingletonExtent
	convolved, _, err := s.libraryConvolve(state, out, cudnn.Shape{
		In: channels, Out: channels, Frames: staged, Height: h, Width: w,
		Kernel: [tensor.TripleExtent]int{tensor.TripleExtent, tensor.SingletonExtent, tensor.SingletonExtent},
		Stride: [tensor.TripleExtent]int{scale, tensor.SingletonExtent, tensor.SingletonExtent},
	}, frames/scale*spatial, weight, bias, func(input driver.DevicePtr, values int) error {
		return s.stageCausal(state, input, x, prior, frames, staged, spatial, values)
	})
	return convolved, err
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
