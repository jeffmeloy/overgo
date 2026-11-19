//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// goldenFloorMargin is how far below its first measured run a section's floor is set (moss-ultra-compare/golden-floor),
// the margin the section checks set theirs by.
const goldenFloorMargin = 3

// pictureSquare is the side, in pixels, of the squares a picture's agreement is read over, as the Krea golden reads
// its (fixtures/krea/own_picture.json).
const pictureSquare = 8

// goldenFloors are each case's floors, its final latent's and its finished picture's agreement with ComfyUI's in dB
// over squares, each its first measured run's (2026-11-17) less goldenFloorMargin. ComfyUI runs the Base's projections
// in int8 with its activations quantized per row, overgo the weights decoded to bfloat16, and ComfyUI repeats itself
// exactly (pass-1 against pass-2, 354 dB), so the agreement is how far the two arithmetics part over 50 steps. A section
// drawn from noise on the workflow's saved empty prompt, or mode 3's eight empty lines, parts the most: there overgo's
// own attention in 8 bits moves overgo to 2.38 dB of itself, where on mode 2's long prompt it keeps 17.00 dB, so
// ComfyUI's arithmetic reaching 1.55 dB is that sensitivity, not a fault; a picture or depth map holds its section.
var goldenFloors = map[string]goldenFloor{
	"text to image":                 {latent: 1.55 - goldenFloorMargin, picture: 15.11 - goldenFloorMargin},
	"prompt mode 2":                 {latent: 15.57 - goldenFloorMargin, picture: 32.73 - goldenFloorMargin},
	"prompt mode 3":                 {latent: 0.88 - goldenFloorMargin, picture: 16.50 - goldenFloorMargin},
	"image to image":                {latent: 1.55 - goldenFloorMargin, picture: 15.11 - goldenFloorMargin},
	"image to image at denoise 0.6": {latent: 17.91 - goldenFloorMargin, picture: 32.94 - goldenFloorMargin},
	"depth control":                 {latent: 14.13 - goldenFloorMargin, picture: 28.08 - goldenFloorMargin},
}

// goldenFloor is a case's floors: its final latent's and its finished picture's least agreement with ComfyUI's, in dB.
type goldenFloor struct{ latent, picture float64 }

// goldenCase is one section or prompt mode of the MOSS Ultra workflow as a recording of it holds it: the run and
// section it was recorded in, its sampler (whose final latent the recorder saved), the node that saved its finished
// picture, the request that asks it, and that request with its input changed.
type goldenCase struct {
	name, run, slug   string
	sampler, finished string
	request           func(t *testing.T, recording comfySection) Request
	control           func(t *testing.T, request Request) Request
}

// reseeded is a request drawn from other noise, the control of a section drawn from noise alone.
func reseeded(_ *testing.T, request Request) Request {
	request.Seed++
	return request
}

// textSection is a text to image section's request as its recording asks it at 1024 x 576: the prompts its preview
// nodes show, its sampler's guidance, steps and seed on ER-SDE and bong_tangent, and its finishing's sharpening.
func textSection(t *testing.T, recording comfySection, sampler, positive, negative, sharpen string) Request {
	t.Helper()
	inputs := recording.inputs(sampler)
	if inputs["sampler_name"] != "er_sde" || inputs["scheduler"] != sampling.ScheduleBongTangent || len(recording.Texts[positive]) != 1 || len(recording.Texts[negative]) != 1 {
		t.Fatalf("the recording's sampler is %v with prompts %v; want ER-SDE on bong_tangent and one prompt of each", inputs, recording.Texts)
	}
	guidance := float32(inputs["cfg"].(float64))
	finishing := recording.inputs(sharpen)
	return Request{Prompt: recording.Texts[positive][0], NegativePrompt: recording.Texts[negative][0], GuidanceScale: &guidance, Width: 1024, Height: 576,
		Steps: int(inputs["steps"].(float64)), Seed: int64(inputs["seed"].(float64)),
		Schedule: &sampling.Schedule{Sampler: sampling.SamplerERSDE, Name: sampling.ScheduleBongTangent},
		Sharpen:  &media.Sharpening{Radius: int(finishing["sharpen_radius"].(float64)), Sigma: finishing["sigma"].(float64), Alpha: finishing["alpha"].(float64)}}
}

// goldenCases are the workflow's sections that run MOSS's 50 steps and its prompt modes, each recorded at 1024 x 576.
var goldenCases = []goldenCase{
	{name: "text to image", run: "pass-1", slug: "text-to-image", sampler: "598", finished: "482", control: reseeded,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			if recording.Set["670.megapixels"] != 0.5625 {
				t.Fatalf("the recording set %v; want the canvas at 1024 x 576", recording.Set)
			}
			return textSection(t, recording, "598", "436", "592", "599")
		}},
	{name: "prompt mode 2", run: "mode-2", slug: "text-to-image", sampler: "598", finished: "482", control: reseeded,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			if recording.Set["483.value"] != float64(2) {
				t.Fatalf("the recording set %v; want prompt mode 2", recording.Set)
			}
			// Mode 2's text is sampled, so the section is held as drawn from the text ComfyUI's enhancer wrote.
			return textSection(t, recording, "598", "436", "592", "599")
		}},
	{name: "prompt mode 3", run: "mode-3", slug: "text-to-image", sampler: "598", finished: "482", control: reseeded,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			builder := recording.inputs("450")
			if recording.Set["483.value"] != float64(3) || builder["style_palette_data"] != "" || builder["elements_data"] != "" {
				t.Fatalf("the recording set %v over a builder of %v; want mode 3 with neither a palette nor a region", recording.Set, builder)
			}
			field := func(name string) string { value, _ := builder[name].(string); return value }
			request := textSection(t, recording, "598", "436", "592", "599")
			request.Prompt = ""
			request.Layout = &LayoutDraft{Description: field("high_level_description"), Background: field("background"), Style: field("style"),
				Photo: field("style.photo"), ArtStyle: field("style.art_style"), Aesthetics: field("aesthetics"), Lighting: field("lighting"), Medium: field("medium")}
			return request
		}},
	{name: "image to image", run: "pass-1", slug: "image-to-image", sampler: "506", finished: "604", control: reseeded,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			if recording.Set["428.image"] != "overgo-recorder-input.png" || recording.Set["680.megapixels"] != 0.5625 {
				t.Fatalf("the recording set %v; want the recorder's picture at 1024 x 576", recording.Set)
			}
			// At the request's denoise of 1 the section starts from the noise alone, at sigma 1, so its control draws
			// other noise.
			request := textSection(t, recording, "506", "585", "586", "603")
			request.Image, request.Denoise = base64.StdEncoding.EncodeToString(ultraPictureBytes(t)), recording.inputs("506")["denoise"].(float64)
			return request
		}},
	{name: "image to image at denoise 0.6", run: "denoise", slug: "image-to-image", sampler: imageToImageSampler, finished: "604", control: mirroredPicture,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			request := denoiseSection(t, recording)
			request.Sharpen = textSection(t, recording, imageToImageSampler, "585", "586", "603").Sharpen
			return request
		}},
	{name: "depth control", run: "pass-1", slug: "depth-control", sampler: depthControlSampler, finished: "610",
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			request, _ := depthControlSection(t, recording)
			request.Sharpen = textSection(t, recording, depthControlSampler, "607", "608", "609").Sharpen
			return request
		},
		control: func(t *testing.T, request Request) Request {
			t.Helper()
			control := *request.Control
			depthMap := testcheck.Must(base64.StdEncoding.DecodeString(control.Map))(t)
			control.Map = base64.StdEncoding.EncodeToString(mirrored(t, depthMap))
			request.Control = &control
			return request
		}},
}

// TestUltraSectionsHoldTheirGoldenFloors holds overgo's MOSS on the MOSS Ultra workflow's own files (ultraModelRoot) to
// the workflow's recordings (moss-ultra-compare/golden-floor): each section that runs MOSS's 50 steps, and each prompt
// mode, asked as its recording asks it, keeps its final latent and its finished picture above their floors of agreement
// with ComfyUI's, in dB over squares; and the same request with its input changed, other noise for a section drawn
// from noise and the picture or map mirrored for one drawn from them, keeps its final latent below, so each floor
// holds the input's part.
func TestUltraSectionsHoldTheirGoldenFloors(t *testing.T) {
	cudatest.Require(t)
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	holdGoldenFloors(t, root, profile, goldenCases, goldenFloors)
	t.Run("own size", func(t *testing.T) { ownSizeReading(t, root, profile) })
}

// holdGoldenFloors draws each case as its recording asks it, and with its input changed, on the root, and holds its
// final latent and finished picture to its floors, its changed draw's latent below.
func holdGoldenFloors(t *testing.T, root string, profile Profile, cases []goldenCase, floors map[string]goldenFloor) {
	t.Helper()
	readings := readGoldenCases(t, root, profile, cases)
	for _, golden := range cases {
		reading := readings[golden.name]
		floor, declared := floors[golden.name]
		if !declared {
			t.Errorf("%s declares no floor", golden.name)
			continue
		}
		if !(reading.latent >= floor.latent) || !(reading.picture >= floor.picture) {
			t.Errorf("%s keeps %.2f dB in its latent and %.2f dB in its picture; want at least its floors, %.2f and %.2f dB",
				golden.name, reading.latent, reading.picture, floor.latent, floor.picture)
		}
		if reading.control >= floor.latent {
			t.Errorf("%s with its input changed keeps %.2f dB in its latent, its floor %.2f dB or more, so the floor does not hold the input's part",
				golden.name, reading.control, floor.latent)
		}
	}
}

// goldenReading is a case's agreement with its recording, in dB: its final latent's, its finished picture's, and its
// final latent's with its input changed.
type goldenReading struct{ latent, picture, control float64 }

// readGoldenCases draws each case as its recording asks it, and with its input changed, on the root, and reads its
// agreement with its recording.
func readGoldenCases(t *testing.T, root string, profile Profile, cases []goldenCase) map[string]goldenReading {
	t.Helper()
	ctx := t.Context()
	readings := map[string]goldenReading{}
	for _, golden := range cases {
		recording := readComfySection(t, golden.run, golden.slug)
		request := golden.request(t, recording)
		wantLatent, wantPicture := recording.latent(t, golden.sampler), recording.picture(t, golden.finished)
		generator := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
		drawn := func(request Request) ([]float32, []byte) {
			t.Helper()
			session := testcheck.Must(generator.prepare(ctx, request))(t)
			testcheck.Must(generator.integrate(ctx, session))(t)
			final := testcheck.Must(generator.rotary.latent.outputs.CopyToHost(ctx, generator.rotary.latent.node))(t)
			picture := testcheck.Must(generator.decode(ctx, session))(t)
			return final.Data, picture.Data
		}
		latent, picture := drawn(request)
		latentAgreement, pictureAgreement := decibelsOverSquares(latent, wantLatent), pictureDecibels(t, picture, wantPicture)
		changed := golden.control(t, request)
		testcheck.NoError(t, generator.Reset(ctx, changed))
		controlLatent, _ := drawn(changed)
		controlAgreement := decibelsOverSquares(controlLatent, wantLatent)
		closeReporting(t, context.WithoutCancel(ctx), generator)
		t.Logf("%s: the final latent keeps %.2f dB, the finished picture %.2f dB over %d-pixel squares; with its input changed the latent keeps %.2f dB",
			golden.name, latentAgreement, pictureAgreement, pictureSquare, controlAgreement)
		readings[golden.name] = goldenReading{latent: latentAgreement, picture: pictureAgreement, control: controlAgreement}
	}
	return readings
}

// ownSizeReading reads the workflow's text to image section at its own size (the own-size recording: its canvas 16:9
// at 3 megapixels in multiples of 32, as ResolutionSelector 670 sizes it) on the card ComfyUI drew it on: overgo's time
// per step, its time from the prompt to the picture and its peak device memory, its weights loaded, beside ComfyUI's
// (the sampler's progress stamps, the run's milliseconds and the card's peak). A reading, not a floor.
func ownSizeReading(t *testing.T, root string, profile Profile) {
	ctx := t.Context()
	recording := readComfySection(t, "own-size", "text-to-image")
	canvas := recording.inputs("670")
	if len(recording.Set) != 0 || canvas["megapixels"] != float64(3) || canvas["multiple"] != float64(32) || canvas["aspect_ratio"] != "16:9 (Widescreen)" {
		t.Fatalf("the recording set %v over a canvas of %v; want the workflow's own 16:9 at 3 megapixels", recording.Set, canvas)
	}
	request := textSection(t, recording, "598", "436", "592", "599")
	request.Width, request.Height, request.Canvas = 0, 0, &media.Canvas{Aspect: "16:9", Megapixels: 3, Multiple: 32}
	driver.ResetStagePeakBytes()
	generator := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	start := time.Now()
	session := testcheck.Must(generator.prepare(ctx, request))(t)
	samplingFrom := time.Now()
	testcheck.Must(generator.integrate(ctx, session))(t)
	sampled := time.Since(samplingFrom)
	picture := testcheck.Must(generator.decode(ctx, session))(t)
	total, peak := time.Since(start), driver.StagePeakBytes()
	stamps := recording.Progress["598"]
	if len(stamps) < 2 || picture.Width == 0 {
		t.Fatalf("the recording stamps %d of the sampler's steps and overgo drew %dx%d; want the steps and a picture", len(stamps), picture.Width, picture.Height)
	}
	comfyStep := time.Duration(stamps[len(stamps)-1].Microseconds-stamps[0].Microseconds) * time.Microsecond / time.Duration(len(stamps)-1)
	t.Logf("at %dx%d: overgo %v a step, %v from the prompt to the picture, %.2f GiB of the card at most; ComfyUI %v a step, %v, %.2f GiB",
		picture.Width, picture.Height, sampled/time.Duration(request.Steps), total.Round(time.Millisecond), float64(peak)/(1<<30),
		comfyStep, time.Duration(recording.Milliseconds)*time.Millisecond, float64(recording.PeakCardBytes)/(1<<30))
}

// picture is the PNG the recorder saved of node, by the SHA-256 the recording names.
func (s comfySection) picture(t *testing.T, node string) []byte {
	t.Helper()
	for _, file := range s.Files {
		if file.Node != node {
			continue
		}
		path := filepath.Join(s.folder, file.Name)
		data := testcheck.Must(os.ReadFile(path))(t)
		if sum := sha256.Sum256(data); hex.EncodeToString(sum[:]) != file.SHA256 {
			t.Fatalf("%s hashes %x; want the recorded %s", path, sum, file.SHA256)
		}
		return data
	}
	t.Fatalf("the recording keeps no picture of %s", node)
	return nil
}

// pictureDecibels is how far a picture agrees with another, both PNGs of one size, in dB over the means of each colour
// in squares of pictureSquare pixels: the peak level over the root mean square of the means' differences.
func pictureDecibels(t *testing.T, got, want []byte) float64 {
	t.Helper()
	gotMeans, wantMeans := squareColourMeans(t, got), squareColourMeans(t, want)
	if len(gotMeans) != len(wantMeans) || len(wantMeans) == 0 {
		t.Fatalf("the pictures hold %d and %d square means; want one size", len(gotMeans), len(wantMeans))
	}
	var squared float64
	for index, mean := range gotMeans {
		apart := mean - wantMeans[index]
		squared += apart * apart
	}
	return 20 * math.Log10(math.MaxUint8/math.Sqrt(squared/float64(len(wantMeans))))
}

// squareColourMeans is a PNG's mean red, green and blue level in each square of pictureSquare pixels, row by row.
func squareColourMeans(t *testing.T, data []byte) []float64 {
	t.Helper()
	decoded := testcheck.Must(png.Decode(bytes.NewReader(data)))(t)
	bounds := decoded.Bounds()
	var means []float64
	for top := bounds.Min.Y; top+pictureSquare <= bounds.Max.Y; top += pictureSquare {
		for left := bounds.Min.X; left+pictureSquare <= bounds.Max.X; left += pictureSquare {
			var sums [3]float64
			for y := top; y < top+pictureSquare; y++ {
				for x := left; x < left+pictureSquare; x++ {
					// The pictures are opaque, so the levels RGBA gives are the colours' own, in 16 bits.
					r, g, b, _ := decoded.At(x, y).RGBA()
					sums[0], sums[1], sums[2] = sums[0]+float64(r>>8), sums[1]+float64(g>>8), sums[2]+float64(b>>8)
				}
			}
			for _, sum := range sums {
				means = append(means, sum/(pictureSquare*pictureSquare))
			}
		}
	}
	return means
}
