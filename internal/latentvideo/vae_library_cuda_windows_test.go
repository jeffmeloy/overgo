//go:build windows

package latentvideo

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// clipRoundTripScript runs the video autoencoder's own code over the small
// clip and writes the fixture.
const clipRoundTripScript = "scripts/python_golden_references/wan_vae_clip_round_trip.py"

// codecPathRecord holds what the two paths took on a quiet device: the times,
// the device memory and the distances TestLibraryCodecKeepsWithinTheKernelsMemoryAndLevels
// logs, with what a user installs for the library.
const codecPathRecord = "docs/verification/video-codec-convolution-paths.txt"

const (
	// A path's value may stand this many of a frame's 255 levels from the value
	// the autoencoder's own code gives, and its error against the source this
	// far, as a share, from that code's own: the tolerance a served
	// reconstruction is held to
	// (internal/mediacapability TestVideoAutoencoderRoundTripMatchesItsOwnCode).
	codecLevels     = 2
	codecErrorShare = 0.04
	// signedUnitLevel is one of a byte's 255 levels in a value from -1 to 1.
	signedUnitLevel = 2.0 / 255
	// noCodecLibrary says why a test of the library's path has nothing to hold.
	noCodecLibrary = "this machine's search path holds no cuDNN, so the codec it serves convolves by the kernels"
)

// clipRoundTrip is fixtures/wan/vae_clip_round_trip.json: a clip of five
// frames and the clip the autoencoder's own code makes of it.
type clipRoundTrip struct {
	Source           string              `json:"source"`
	Weights          string              `json:"weights"`
	LatentShape      []int               `json:"latent_shape"`
	MeanSquaredError float64             `json:"mean_squared_error"`
	Clip             clipRoundTripTensor `json:"clip"`
	Image            clipRoundTripTensor `json:"image"`
}

type clipRoundTripTensor struct {
	File   string `json:"file"`
	Shape  []int  `json:"shape"`
	SHA256 string `json:"sha256"`
}

// ownCodeClip reads the clip of five frames and the clip the autoencoder's own
// code makes of it, each held to the digest the reference recorded.
func ownCodeClip(t *testing.T) (reference clipRoundTrip, clip SourceVideo, want []float32) {
	t.Helper()
	fixtures := filepath.Join(testutil.RepoRoot(t), "fixtures", "wan")
	if err := jsonfile.DecodeStrict(filepath.Join(fixtures, "vae_clip_round_trip.json"), &reference); err != nil {
		t.Fatalf("the reference round trip is absent (%v); write it with %s", err, clipRoundTripScript)
	}
	read := func(tensor clipRoundTripTensor) []float32 {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(fixtures, filepath.FromSlash(tensor.File)))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(raw)
		values := 1
		for _, extent := range tensor.Shape {
			values *= extent
		}
		if hex.EncodeToString(sum[:]) != tensor.SHA256 || len(tensor.Shape) != 4 || len(raw) != 4*values {
			t.Fatalf("%s is not the %v tensor the reference recorded; write it again with %s", tensor.File, tensor.Shape, clipRoundTripScript)
		}
		floats := make([]float32, values)
		for index := range floats {
			floats[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*index:]))
		}
		return floats
	}
	source, want := read(reference.Clip), read(reference.Image)
	shape := reference.Clip.Shape
	if !slices.Equal(reference.Image.Shape, shape) {
		t.Fatalf("the reference's clip is %v and its source %v", reference.Image.Shape, shape)
	}
	clip = SourceVideo{Pixels: source, Channels: shape[0], Frames: shape[1], Height: shape[2], Width: shape[3]}
	if clip.Frames*clip.Height*clip.Width >= libraryClipValues {
		t.Fatalf("the reference's clip of %d values is one a served autoencoder convolves through the library, and the tests are of one it does not", clip.Frames*clip.Height*clip.Width)
	}
	return reference, clip, want
}

// holdToOwnCode loads the served autoencoder, routes its halves as route
// does, and holds the clip it gives back, the third time as the first, to the
// reference: no value more than two of 255 levels from it and the error
// against the source within four parts in a hundred of its own. It returns
// the clip and the halves as the round trips left them, and skips a test of
// the library on a machine that has none.
func holdToOwnCode(t *testing.T, reference clipRoundTrip, clip SourceVideo, want []float32, route func(*vaeDecoderCUDASession)) ([]float32, map[string]*vaeDecoderCUDASession) {
	t.Helper()
	runtime, err := LoadAutoencoderRuntime(wanModelDir(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := runtime.Close(context.Background()); err != nil {
			t.Error(err)
		}
	})
	halves := map[string]*vaeDecoderCUDASession{"encoder": runtime.encoder.codec, "decoder": runtime.decoder}
	for _, half := range halves {
		route(half)
	}
	var got SourceVideo
	var latent SourceVideoShape
	for repetition := range 3 {
		started := time.Now()
		if got, latent, err = runtime.RoundTrip(t.Context(), clip); err != nil {
			t.Fatal(err)
		}
		if runtime.encoder.codec.route.absent || runtime.decoder.route.absent {
			testskip.NotApplicable(t, noCodecLibrary)
		}
		t.Logf("run %d: %d frames at %d by %d made the round trip in %.3f s", repetition+1, clip.Frames, clip.Width, clip.Height, time.Since(started).Seconds())
	}
	if extent := []int{latent.Channels, latent.Frames, latent.Height, latent.Width}; !slices.Equal(extent, reference.LatentShape) {
		t.Fatalf("the latent is %v, the reference's %v", extent, reference.LatentShape)
	}
	if len(got.Pixels) != len(want) {
		t.Fatalf("the clip that came back holds %d values of %d", len(got.Pixels), len(want))
	}
	var worst, squared float64
	for index, value := range got.Pixels {
		worst = max(worst, math.Abs(float64(value)-float64(want[index])))
		squared += (float64(value) - float64(clip.Pixels[index])) * (float64(value) - float64(clip.Pixels[index]))
	}
	mean := squared / float64(len(want))
	share := math.Abs(mean-reference.MeanSquaredError) / reference.MeanSquaredError
	t.Logf("furthest value %.3f of a level from the reference's; error against the source %.8f, the reference's %.8f, %.2f parts in 100 apart",
		worst/signedUnitLevel, mean, reference.MeanSquaredError, 100*share)
	if worst > codecLevels*signedUnitLevel {
		t.Errorf("a value stands %.3f levels from the one the autoencoder's own code gives, where %d are allowed", worst/signedUnitLevel, codecLevels)
	}
	if share > codecErrorShare {
		t.Errorf("the clip's error against its source is %.8f, the reference's %.8f: %.2f parts in 100 apart, where %.0f are allowed", mean, reference.MeanSquaredError, 100*share, 100*codecErrorShare)
	}
	return got.Pixels, halves
}

// TestAdoptedCodecPathAgreesWithItsOwnCodeOnASmallClip holds the path a served
// autoencoder convolves a small clip by to the autoencoder's own code, on a
// clip of five frames: no value of the clip it gives back stands more than two
// of 255 levels from the reference's, and its error against the source stands
// within four parts in a hundred of the reference's own. A clip this small is
// one the served autoencoder keeps to the kernels on any machine, since the
// library would not repay its loading: it loads none for it, and its clip is,
// value for value, the clip of an autoencoder that was never given the
// library. The library's own path over this clip is held by
// TestLibraryCodecAgreesWithItsOwnCodeOnASmallClip, on a machine that has it.
func TestAdoptedCodecPathAgreesWithItsOwnCodeOnASmallClip(t *testing.T) {
	cudatest.Require(t)
	reference, clip, want := ownCodeClip(t)
	var byKernels []float32
	t.Run("kernels", func(t *testing.T) {
		byKernels, _ = holdToOwnCode(t, reference, clip, want, func(half *vaeDecoderCUDASession) { half.route = libraryRoute{} })
	})
	t.Run("served", func(t *testing.T) {
		served, halves := holdToOwnCode(t, reference, clip, want, func(*vaeDecoderCUDASession) {})
		for name, half := range halves {
			if !half.route.adopted || half.route.taken || half.library != nil {
				t.Errorf("the served %s took the library for a clip of %d values, where one of %d repays it", name, clip.Frames*clip.Height*clip.Width, libraryClipValues)
			}
		}
		if !slices.Equal(served, byKernels) {
			t.Error("the served autoencoder's clip is not the kernels' value for value")
		}
	})
}

// TestLibraryCodecAgreesWithItsOwnCodeOnASmallClip holds cuDNN's path, through
// its binding, to the autoencoder's own code on the clip of five frames, with
// the library given to both halves for a clip of any size. More than one frame
// is the point: a frame after the first is convolved behind the frames before
// it, which the library's path keeps in half precision from one chunk to the
// next, and a single frame's round trip reads none.
func TestLibraryCodecAgreesWithItsOwnCodeOnASmallClip(t *testing.T) {
	cudatest.Require(t)
	reference, clip, want := ownCodeClip(t)
	_, halves := holdToOwnCode(t, reference, clip, want, func(half *vaeDecoderCUDASession) { half.adoptLibrary(0) })
	for name, half := range halves {
		if !half.route.taken || len(half.library.plans) == 0 || len(half.library.refused) != 0 {
			t.Errorf("the %s convolved %d shapes through the library and was refused %d: its clip is not the library's", name, len(half.library.plans), len(half.library.refused))
		}
	}
}

// codecRun is what one path took over a clip: each repetition's time to encode
// and to decode, the clip it decoded, the most of its own allocations each
// half held, and the device memory both halves held at the end beyond their
// weights.
type codecRun struct {
	encoded, decoded          []time.Duration
	clip                      []float32
	encoderPeak, decoderPeak  uint64
	deviceHeld                int64
	convolved, refusedByCuDNN int
}

// TestLibraryCodecKeepsWithinTheKernelsMemoryAndLevels runs the autoencoder
// over clips of the sizes a served request carries, 21 frames and 81 at 832
// by 480, by the kernels and by the library, three times each, and holds what
// adopting the library was bound by that is no time: the library's clip
// stands within two of 255 levels of the kernels', which the host reference
// holds, neither half holds more of its own allocations than under the
// kernels, and the library is refused no shape. Each half is given the library
// as a served one is, so the first repetition loads it, plans its shapes and
// packs its filters, and its time has all of that in it. The test logs each
// repetition's time and the device memory the halves held; the figures of a
// quiet device are in the record.
func TestLibraryCodecKeepsWithinTheKernelsMemoryAndLevels(t *testing.T) {
	cudatest.Require(t)
	directory := wanModelDir(t)
	checkpoint := AutoencoderWeights(directory)
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := ResolveProfile(directory)
	if err != nil {
		t.Fatal(err)
	}
	encoderGraph, err := CompileVAEEncoderPlan(catalog.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	decoderPlan, err := CompileVAEDecoderPlan(catalog.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	// A half asked to route a clip through the library says whether the machine
	// has one. The library it loads is released with it, and the process loads
	// it again, so the first repetition's time below still has the loading in it.
	probe, err := NewVAEDecoderCUDASession(checkpoint, decoderPlan, device.DefaultOrdinal())
	if err != nil {
		t.Fatal(err)
	}
	probe.adoptLibrary(0)
	err = probe.routeClip(t.Context(), 0)
	absent := probe.route.absent
	if err := errors.Join(err, probe.Close()); err != nil {
		t.Fatal(err)
	}
	if absent {
		testskip.NotApplicable(t, noCodecLibrary)
	}
	const height, width, repetitions = 480, 832, 3
	for _, frames := range []int{21, 81} {
		t.Run(fmt.Sprintf("%d frames", frames), func(t *testing.T) {
			plan, err := CompileSourceCodecBoundary(SourceCodecProfile{
				InputChannels: encoderGraph.InputChannels, LatentChannels: encoderGraph.LatentChannels,
				Stride: encoderGraph.Stride, LatentStats: profile.LatentStats,
			}, SourceVideoShape{Channels: media.RGBChannels, Frames: frames, Height: height, Width: width})
			if err != nil {
				t.Fatal(err)
			}
			plane := height * width
			source := make([]float32, media.RGBChannels*frames*plane)
			for channel := range media.RGBChannels {
				for frame := range frames {
					for at := range plane {
						x, y := at%width, at/width
						source[(channel*frames+frame)*plane+at] = float32((x+2*y+7*frame+40*channel)%width) / float32(width)
					}
				}
			}
			free := func(session *vaeDecoderCUDASession) (bytes uint64) {
				t.Helper()
				if err := session.worker.Do(t.Context(), func(state *device.State) (err error) {
					bytes, _, err = state.Driver.MemInfo()
					return err
				}); err != nil {
					t.Fatal(err)
				}
				return bytes
			}
			var latent []float32
			run := func(library bool) (result codecRun) {
				t.Helper()
				encoder, err := newVAEEncoderCUDASession(checkpoint, encoderGraph, device.DefaultOrdinal())
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := encoder.Close(); err != nil {
						t.Error(err)
					}
				}()
				decoder, err := NewVAEDecoderCUDASession(checkpoint, decoderPlan, device.DefaultOrdinal())
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := decoder.Close(); err != nil {
						t.Error(err)
					}
				}()
				before := free(decoder)
				if library {
					// As a served autoencoder is given it: the first clip large enough loads it.
					encoder.codec.adoptLibrary(libraryClipValues)
					decoder.adoptLibrary(libraryClipValues)
				}
				for repetition := range repetitions {
					started := time.Now()
					encoded, _, err := encoder.Encode(t.Context(), plan, source)
					if err != nil {
						t.Fatal(err)
					}
					result.encoded = append(result.encoded, time.Since(started))
					// Both paths decode the kernels' latent, so their clips differ by the decoder alone.
					if latent == nil {
						latent = encoded
					}
					var clip []float32
					started = time.Now()
					if _, err := decoder.DecodeContext(t.Context(), profile.LatentStats, latent, plan.Latent.Frames, plan.Latent.Height, plan.Latent.Width,
						func(_ int, frame []float32, _, _ int) error {
							if repetition == 0 {
								clip = append(clip, frame...)
							}
							return nil
						}); err != nil {
						t.Fatal(err)
					}
					result.decoded = append(result.decoded, time.Since(started))
					if repetition == 0 {
						result.clip = clip
					}
				}
				result.deviceHeld = int64(before) - int64(free(decoder))
				for session, peak := range map[*vaeDecoderCUDASession]*uint64{encoder.codec: &result.encoderPeak, decoder: &result.decoderPeak} {
					if err := session.worker.Do(t.Context(), func(state *device.State) error {
						*peak = state.Driver.MemoryStats().PeakBytes
						return nil
					}); err != nil {
						t.Fatal(err)
					}
					if library {
						if !session.route.taken {
							t.Fatalf("a clip of %d values did not take the library, where one of %d does", frames*plane, libraryClipValues)
						}
						result.convolved, result.refusedByCuDNN = result.convolved+len(session.library.plans), result.refusedByCuDNN+len(session.library.refused)
					}
				}
				return result
			}
			seconds := func(times []time.Duration) (text string) {
				for _, taken := range times {
					text += fmt.Sprintf(" %.2f", taken.Seconds())
				}
				return text
			}
			const gibibyte = 1 << 30
			kernels, library := run(false), run(true)
			for name, path := range map[string]codecRun{"kernels": kernels, "library": library} {
				t.Logf("%s: encoded in%s s, decoded in%s s; own allocations at most %.2f GiB encoding and %.2f GiB decoding; the device held %.2f GiB beyond the weights",
					name, seconds(path.encoded), seconds(path.decoded), float64(path.encoderPeak)/gibibyte, float64(path.decoderPeak)/gibibyte, float64(path.deviceHeld)/gibibyte)
			}
			var worst float64
			for index, value := range library.clip {
				worst = max(worst, math.Abs(float64(value)-float64(kernels.clip[index])))
			}
			t.Logf("the library convolved %d shapes and was refused %d; its clip stands at most %.3f of a level from the kernels'; a quiet device's figures are in %s",
				library.convolved, library.refusedByCuDNN, worst/signedUnitLevel, codecPathRecord)
			if len(library.clip) != len(kernels.clip) || worst > codecLevels*signedUnitLevel {
				t.Errorf("the library's clip of %d values stands %.3f levels from the kernels' of %d, where %d are allowed", len(library.clip), worst/signedUnitLevel, len(kernels.clip), codecLevels)
			}
			if library.encoderPeak > kernels.encoderPeak || library.decoderPeak > kernels.decoderPeak {
				t.Errorf("the library's halves held %d and %d bytes of their own allocations, the kernels' %d and %d: adopting it was bound by the codec's peak",
					library.encoderPeak, library.decoderPeak, kernels.encoderPeak, kernels.decoderPeak)
			}
			if library.convolved == 0 || library.refusedByCuDNN != 0 {
				t.Errorf("the library convolved %d shapes and was refused %d", library.convolved, library.refusedByCuDNN)
			}
		})
	}
}
