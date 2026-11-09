//go:build windows

// Package nvvfx runs NVIDIA's Video Effects runtime, its RTX Video Super Resolution effect among them, from the
// directory OVERGO_NVVFX names: its libraries loaded by syscall from that directory, as internal/cuda/cudnn loads
// cuDNN, with no Python and no cgo. The runtime is NVIDIA's, under NVIDIA's licence: overgo loads it where it is
// installed and never copies it. Its interface is the one NVIDIA publishes (github.com/NVIDIA-Maxine/Maxine-VFX-SDK,
// nvvfx/include); the effect's selector and level parameter are those its nvidia-vfx binding names.
package nvvfx

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"

	"overgo/internal/cuda/driver"
)

// directoryEnvironment names the directory NVIDIA's Video Effects runtime is loaded from, as its nvidia-vfx wheel
// installs it (nvvfx/libs).
const directoryEnvironment = "OVERGO_NVVFX"

// runtimeFiles are the runtime's libraries in the order NVIDIA's binding loads them (nvidia-vfx 0.1.0.1, _lib_loader.py):
// each part before the libraries that ask for it by name, so a name asked for is a library already loaded.
var runtimeFiles = []string{
	"nppc64_12.dll", "nppial64_12.dll", "nppicc64_12.dll", "nppidei64_12.dll", "nppif64_12.dll", "nppig64_12.dll",
	"nppim64_12.dll", "nppist64_12.dll", "nppitc64_12.dll", "nvinfer_10.dll", "nvinfer_plugin_10.dll",
	"nvonnxparser_10.dll", imageLibrary, "nvngxruntime.dll", effectsLibrary, "nvVFXVideoSuperRes.dll", "nvngx_vsr.dll",
}

// The two libraries whose entry points overgo calls.
const (
	imageLibrary   = "NVCVImage.dll"
	effectsLibrary = "NVVideoEffects.dll"
)

// The names the effect is created and set by (nvVideoEffects.h, and nvidia-vfx's video_super_res.py for the RTX
// effect, which the public header does not list).
const (
	videoSuperResSelector = "VideoSuperRes"
	qualityParameter      = "QualityLevel"
	sourceImageParameter  = "SrcImage0"
	targetImageParameter  = "DstImage0"
	streamParameter       = "CudaStream"
)

// An NvCVImage's descriptions (nvCVImage.h): planar float32 RGB, overgo's frames, and interleaved 8-bit RGBA, the
// effect's only source (probed 2026-11-09: RGB, BGR, float32 and float16 sources are refused), on the card, in the 64
// bytes a descriptor takes on x64.
const (
	pixelFormatRGB    = 4
	pixelFormatRGBA   = 6
	componentU8       = 1
	componentF32      = 7
	layoutInterleaved = 0
	layoutPlanar      = 1
	memoryGPU         = 1
	imageDescriptor   = 64
	float32Bytes      = 4
	rgbChannels       = 3
	rgbaBytes         = 4
	byteScale         = 255
	statusSuccess     = 0
	runSynchronously  = 0
	statusTextBytes   = 256
)

// Runtime is NVIDIA's Video Effects runtime loaded from its directory, its entry points resolved.
type Runtime struct {
	directory                                          string
	create, destroy, setImage, setU32, setStream, load *syscall.Proc
	run, imageInit, transferImage, errorString         *syscall.Proc
}

// Open loads the runtime from the directory OVERGO_NVVFX names, every library by its path in runtimeFiles order, and
// resolves the entry points overgo calls. An unset setting or a library missing from the directory is refused by
// name before anything is loaded; nothing falls back to another upscaler. The libraries stay loaded for the life of the
// process, as the runtime keeps them.
func Open() (*Runtime, error) {
	// Read by its spelled name, which the identity of a capability that finishes its clips names among its result
	// settings (internal/inferencesurface resultSettings).
	directory := os.Getenv("OVERGO_NVVFX")
	if directory == "" {
		return nil, fmt.Errorf("nvvfx: %s names no directory of NVIDIA's Video Effects runtime", directoryEnvironment)
	}
	for _, file := range runtimeFiles {
		if _, err := os.Stat(filepath.Join(directory, file)); err != nil {
			return nil, fmt.Errorf("nvvfx: the runtime %s names has no %s: %w", directoryEnvironment, file, err)
		}
	}
	loaded := map[string]*syscall.DLL{}
	for _, file := range runtimeFiles {
		library, err := syscall.LoadDLL(filepath.Join(directory, file))
		if err != nil {
			return nil, fmt.Errorf("nvvfx: load %s: %w", file, err)
		}
		loaded[file] = library
	}
	r := &Runtime{directory: directory}
	for _, binding := range []struct {
		library, name string
		proc          **syscall.Proc
	}{
		{effectsLibrary, "NvVFX_CreateEffect", &r.create},
		{effectsLibrary, "NvVFX_DestroyEffect", &r.destroy},
		{effectsLibrary, "NvVFX_SetImage", &r.setImage},
		{effectsLibrary, "NvVFX_SetU32", &r.setU32},
		{effectsLibrary, "NvVFX_SetCudaStream", &r.setStream},
		{effectsLibrary, "NvVFX_Load", &r.load},
		{effectsLibrary, "NvVFX_Run", &r.run},
		{imageLibrary, "NvCVImage_Init", &r.imageInit},
		{imageLibrary, "NvCVImage_Transfer", &r.transferImage},
		{imageLibrary, "NvCV_GetErrorStringFromCode", &r.errorString},
	} {
		proc, err := loaded[binding.library].FindProc(binding.name)
		if err != nil {
			return nil, fmt.Errorf("nvvfx: resolve %s in %s: %w", binding.name, binding.library, err)
		}
		*binding.proc = proc
	}
	return r, nil
}

// Libraries are the runtime's libraries as loaded, each by its path in the order Open loads them: the files whose
// digests identify what the effect writes.
func (r *Runtime) Libraries() []string {
	paths := make([]string, len(runtimeFiles))
	for index, file := range runtimeFiles {
		paths[index] = filepath.Join(r.directory, file)
	}
	return paths
}

// status is an NvCV_Status as an error, its text the runtime's own.
func (r *Runtime) status(call string, code uintptr) error {
	if int32(code) == statusSuccess {
		return nil
	}
	text, _, _ := r.errorString.Call(code)
	return fmt.Errorf("nvvfx: %s: %s (%d)", call, driver.ReadCString(text, statusTextBytes), int32(code))
}

// Image is a planar float32 RGB image on the card in [0, 1], its three planes one after another, the layout overgo's
// frames have on the card and NVIDIA's binding hands the effect.
type Image struct {
	Pixels        driver.DevicePtr
	Width, Height int
}

// Bytes is the image's size on the card.
func (image Image) Bytes() uint64 {
	return uint64(image.Width) * uint64(image.Height) * rgbChannels * float32Bytes
}

// vfxEffect is one created effect of the runtime.
type vfxEffect struct {
	runtime *Runtime
	handle  uintptr
}

// VideoSuperRes creates RTX Video Super Resolution, the effect that upscales, denoises or deblurs a frame by the
// level it is set to.
func (r *Runtime) VideoSuperRes() (*vfxEffect, error) {
	selector, err := syscall.BytePtrFromString(videoSuperResSelector)
	if err != nil {
		return nil, err
	}
	var handle uintptr
	code, _, _ := r.create.Call(uintptr(unsafe.Pointer(selector)), uintptr(unsafe.Pointer(&handle)))
	runtime.KeepAlive(selector)
	if err := r.status("create "+videoSuperResSelector, code); err != nil {
		return nil, err
	}
	return &vfxEffect{runtime: r, handle: handle}, nil
}

// SetQuality sets the effect's level: 0 bicubic, 1 to 4 upscaling trained on compressed video, 8 to 11 denoising,
// 12 to 15 deblurring, 16 to 19 upscaling for clean sources (nvidia-vfx's QualityLevel).
func (e *vfxEffect) SetQuality(level uint32) error {
	return e.set(e.runtime.setU32, qualityParameter, uintptr(level))
}

// SetStream runs the effect on stream, overgo's own.
func (e *vfxEffect) SetStream(stream driver.Stream) error {
	return e.set(e.runtime.setStream, streamParameter, uintptr(stream))
}

// set calls one of the runtime's setters on the effect with the parameter named and its value.
func (e *vfxEffect) set(setter *syscall.Proc, parameter string, value uintptr) error {
	name, err := syscall.BytePtrFromString(parameter)
	if err != nil {
		return err
	}
	code, _, _ := setter.Call(e.handle, uintptr(unsafe.Pointer(name)), value)
	runtime.KeepAlive(name)
	return e.runtime.status("set "+parameter, code)
}

// descriptor is an NvCVImage describing pixels already on the card, filled by the runtime's own NvCVImage_Init.
type descriptor [imageDescriptor]byte

// describe is the descriptor of a width by height image at pixels: planar float32 RGB, overgo's frames, or interleaved
// 8-bit RGBA, the only source the effect takes.
func (r *Runtime) describe(pixels driver.DevicePtr, width, height int, planarFloat bool) (*descriptor, error) {
	if width <= 0 || height <= 0 || pixels == 0 {
		return nil, fmt.Errorf("nvvfx: an image of %dx%d at %#x", width, height, pixels)
	}
	format, component, layout, pitch := uintptr(pixelFormatRGBA), uintptr(componentU8), uintptr(layoutInterleaved), width*rgbaBytes
	if planarFloat {
		format, component, layout, pitch = pixelFormatRGB, componentF32, layoutPlanar, width*float32Bytes
	}
	image := new(descriptor)
	code, _, _ := r.imageInit.Call(uintptr(unsafe.Pointer(image)), uintptr(width), uintptr(height), uintptr(pitch), uintptr(pixels),
		format, component, layout, memoryGPU)
	return image, r.status("describe an image", code)
}

// transfer converts source into target on stream by the runtime's own NvCVImage_Transfer, each value multiplied by
// scale: 255 from float to 8 bits, 1/255 back, as NVIDIA's binding converts.
func (r *Runtime) transfer(source, target *descriptor, scale float32, stream driver.Stream) error {
	// The scale is the call's third argument, a float, which the Windows x64 convention passes in XMM2; the system
	// call copies each of the first four arguments into its XMM register as well as its integer one.
	code, _, _ := r.transferImage.Call(uintptr(unsafe.Pointer(source)), uintptr(unsafe.Pointer(target)), uintptr(math.Float32bits(scale)),
		uintptr(stream), 0)
	runtime.KeepAlive(source)
	runtime.KeepAlive(target)
	return r.status("transfer an image", code)
}

// setImage gives the effect the image named parameter, which the runtime copies, so the descriptor is not kept.
func (e *vfxEffect) setImage(parameter string, image *descriptor) error {
	defer runtime.KeepAlive(image)
	return e.set(e.runtime.setImage, parameter, uintptr(unsafe.Pointer(image)))
}

// Load builds the effect for the images and level it was given; a change of either asks for it again.
func (e *vfxEffect) Load() error {
	code, _, _ := e.runtime.load.Call(e.handle)
	return e.runtime.status("load", code)
}

// Run runs the effect once on its stream, returning when it has run.
func (e *vfxEffect) Run() error {
	code, _, _ := e.runtime.run.Call(e.handle, runSynchronously)
	return e.runtime.status("run", code)
}

// Close destroys the effect.
func (e *vfxEffect) Close() error {
	if e == nil || e.handle == 0 {
		return errors.New("nvvfx: the effect is closed")
	}
	_, _, _ = e.runtime.destroy.Call(e.handle)
	e.handle = 0
	return nil
}

// Upscaler runs VideoSuperRes at one level on frames of one size, planar float32 RGB on the card: each frame is
// converted to 8-bit RGBA by the runtime's own transfer, run, and converted back, as NVIDIA's binding runs it, on
// overgo's stream and in overgo's device memory.
type Upscaler struct {
	runtime                    *Runtime
	library                    *driver.Library
	stream                     driver.Stream
	effect                     *vfxEffect
	staged                     [stagedImages]driver.DevicePtr
	source, target             Image
	stagedSource, stagedTarget *descriptor
}

// The staged images an upscaler converts through, the effect's source and its target, and how many they are.
const (
	stagedSourceImage = iota
	stagedTargetImage
	stagedImages
)

// NewUpscaler builds the effect at level for frames of source's size written at target's, on stream, with its 8-bit
// staging in library's device memory.
func (r *Runtime) NewUpscaler(library *driver.Library, stream driver.Stream, level uint32, source, target Image) (*Upscaler, error) {
	effect, err := r.VideoSuperRes()
	if err != nil {
		return nil, err
	}
	Upscaler := &Upscaler{runtime: r, library: library, stream: stream, effect: effect, source: source, target: target}
	if err := Upscaler.build(level); err != nil {
		return nil, errors.Join(err, Upscaler.Close())
	}
	return Upscaler, nil
}

// build stages the effect's images, sets its level and stream, and loads it.
func (u *Upscaler) build(level uint32) error {
	for index, image := range []Image{u.source, u.target} {
		staged, err := u.library.MemAlloc(uint64(image.Width) * uint64(image.Height) * rgbaBytes)
		if err != nil {
			return err
		}
		u.staged[index] = staged
	}
	var err error
	if u.stagedSource, err = u.runtime.describe(u.staged[stagedSourceImage], u.source.Width, u.source.Height, false); err != nil {
		return err
	}
	if u.stagedTarget, err = u.runtime.describe(u.staged[stagedTargetImage], u.target.Width, u.target.Height, false); err != nil {
		return err
	}
	for _, step := range []func() error{
		func() error { return u.effect.SetQuality(level) },
		func() error { return u.effect.SetStream(u.stream) },
		func() error { return u.effect.setImage(sourceImageParameter, u.stagedSource) },
		func() error { return u.effect.setImage(targetImageParameter, u.stagedTarget) },
		u.effect.Load,
	} {
		if err := step(); err != nil {
			return err
		}
	}
	return nil
}

// Frame runs the effect on the frame at source, writing it at target, both planar float32 RGB of the Upscaler's sizes.
func (u *Upscaler) Frame(source, target driver.DevicePtr) error {
	from, err := u.runtime.describe(source, u.source.Width, u.source.Height, true)
	if err != nil {
		return err
	}
	to, err := u.runtime.describe(target, u.target.Width, u.target.Height, true)
	if err != nil {
		return err
	}
	if err := u.runtime.transfer(from, u.stagedSource, byteScale, u.stream); err != nil {
		return err
	}
	if err := u.effect.Run(); err != nil {
		return err
	}
	// The inverse is taken in float32: 1/byteScale of two untyped integer constants is 0.
	return u.runtime.transfer(u.stagedTarget, to, 1/float32(byteScale), u.stream)
}

// Close destroys the effect and frees its staging.
func (u *Upscaler) Close() error {
	err := u.effect.Close()
	for index, staged := range u.staged {
		if staged != 0 {
			err = errors.Join(err, u.library.MemFree(staged))
			u.staged[index] = 0
		}
	}
	return err
}
