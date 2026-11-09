//go:build windows && modeltest

package nvvfx

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// The frame the binding's reference was drawn on (scripts/python_golden_references/nvvfx_video_super_res.py).
const (
	referenceWidth  = 320
	referenceHeight = 180
	referencePeriod = 255
	referenceScale  = 2
)

// bindingDigests are the SHA-256 of nvidia-vfx 0.1.0.1's VideoSuperRes output on the reference frame, planar float32,
// at each level the effect offers: its two runs agreed to the bit at every level on 2026-11-09 (RTX 4090 D, driver
// 616.64), so overgo's output is held to them exactly.
var bindingDigests = map[uint32]string{
	0:  "16cc78a53f7dd2ac72570d913be544bf00e0b592824a573cc40c0c42d0b0f7d1",
	1:  "60b6e4fd7749092f9155ca780a8f2bf439881b4e507ae44016014f64fa60a322",
	2:  "5c75c839362338e014388707cb458a6d1dad6fd7547aae858ea5764ce7f864fc",
	3:  "d9c0c6777af2263b8940414ff75aaac9ffa565ee66bd578ea119ae2128f585dd",
	4:  "b0eb50ee71503c37e6d7b2177a6d35fe61734ca2e7836d7d9997aeee34b92a6e",
	8:  "b90a40d4b80b44ff13b83e0b63a33950b5ab2ef4d24e8e25e9195818b523edb6",
	9:  "209bdec78f776780bc7fd1f6c92ef40889ed91ec8cd49db70d7e0212ff4d2e66",
	10: "c80c8078822b2fcbc63ff3d996b922507ad66496755f804dd984bfed73ed1c72",
	11: "ff442265978d306a8011b8cbb72f69f6fd3fbb8c7e9f88dbd5a81420a53c1d34",
	12: "18b0ac186d21bb3212f25a48554aa863d4ca868f5f059f76cfbd3555764dbaf3",
	13: "d09ba4dff43dc9fc335346d4362c1fef0a12c3881ad9b169a704dff7f6a62c9f",
	14: "40b6ea29b67a11f9c4faaeb3d6134e8d159fe5108c80eaed7c750ff091f7b0dd",
	15: "9b46615fc2417bbeca16d9dd142b25253ed975b29e6a1f183e9b7ad3c73867cb",
	16: "e9df1c154477ac682da9a5cd22bc668b53dc7fb56ba0b93d001552a70220685f",
	17: "37edd8cf0a558904e2136882d4ab5e892479fb7cdadc54831373e50263da1475",
	18: "bffc011bd4f67b5267004e277ffb8fde0d0551a23e8aecc4f4c9fdf72cf83b06",
	19: "476cf31c1fc2243d0128def650b32bd4d848353081ae2d595e33e144263f5a10",
}

// sameSize reports a level that keeps the frame's size: denoising (8 to 11) and deblurring (12 to 15).
func sameSize(level uint32) bool { return level >= 8 && level <= 15 }

// TestVideoSuperResMatchesNvidiasBinding: overgo runs RTX Video Super Resolution from the runtime OVERGO_NVVFX names,
// on a frame in its own device memory and on its own stream, and gives what NVIDIA's nvidia-vfx binding gives on the
// same frame at every level, to the bit: bicubic, the four upscaling levels for compressed video and the four for
// clean sources at twice the size, and the four denoising and four deblurring levels at the frame's own size.
func TestVideoSuperResMatchesNvidiasBinding(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv(directoryEnvironment) == "" {
		testskip.NotApplicable(t, directoryEnvironment+" names no NVIDIA Video Effects runtime here")
	}
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	testcheck.NoError(t, worker.Do(context.Background(), func(state *device.State) (failed error) {
		runtime, err := openVFX()
		if err != nil {
			return err
		}
		source := planarImage{Width: referenceWidth, Height: referenceHeight}
		host := make([]byte, source.Bytes())
		plane := source.Width * source.Height
		for channel := range rgbChannels {
			for y := range source.Height {
				for x := range source.Width {
					value := float32((x*(channel+1)+2*y)%referencePeriod) / referencePeriod
					binary.LittleEndian.PutUint32(host[float32Bytes*(channel*plane+y*source.Width+x):], math.Float32bits(value))
				}
			}
		}
		larger := planarImage{Width: referenceScale * referenceWidth, Height: referenceScale * referenceHeight}
		for _, image := range []*planarImage{&source, &larger} {
			if image.Pixels, err = state.Driver.MemAlloc(image.Bytes()); err != nil {
				return err
			}
			defer func() { failed = errors.Join(failed, state.Driver.MemFree(image.Pixels)) }()
		}
		if err := state.Driver.MemcpyHtoD(source.Pixels, host); err != nil {
			return err
		}
		sameSized := planarImage{Width: referenceWidth, Height: referenceHeight}
		if sameSized.Pixels, err = state.Driver.MemAlloc(sameSized.Bytes()); err != nil {
			return err
		}
		defer func() { failed = errors.Join(failed, state.Driver.MemFree(sameSized.Pixels)) }()
		for level, want := range bindingDigests {
			target := larger
			if sameSize(level) {
				target = sameSized
			}
			started := time.Now()
			upscaler, err := runtime.newUpscaler(state.Driver, state.Stream, level, source, target)
			if err != nil {
				return err
			}
			built := time.Since(started)
			started = time.Now()
			err = errors.Join(upscaler.Frame(source.Pixels, target.Pixels), state.Driver.StreamSynchronize(state.Stream))
			ran := time.Since(started)
			if err = errors.Join(err, upscaler.Close()); err != nil {
				return err
			}
			written := make([]byte, target.Bytes())
			if err := state.Driver.MemcpyDtoH(written, target.Pixels); err != nil {
				return err
			}
			digest := sha256.Sum256(written)
			t.Logf("level %d: %dx%d to %dx%d, built in %s, a frame in %s", level, source.Width, source.Height, target.Width, target.Height, built, ran)
			if got := hex.EncodeToString(digest[:]); got != want {
				t.Errorf("level %d gave %s, NVIDIA's binding %s", level, got, want)
			}
		}
		return nil
	}))
}
