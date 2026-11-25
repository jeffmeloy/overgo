//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/cuda/device"
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
// over squares, each its first measured run's less goldenFloorMargin (2026-11-17; text to image and prompt mode 3
// 2026-11-22). ComfyUI runs the Base's projections in int8 with its activations quantized per row, overgo the weights
// decoded to bfloat16, and ComfyUI repeats itself exactly (pass-1 against pass-2, 354 dB, and the real-1 and builder-1
// runs against real-2 and builder-2, file for file), so the agreement is how far the two arithmetics part over 50
// steps. Text to image is held on a real prompt (realPrompt, the real-1 run) and mode 3 on the builder filled, its
// fields, palette, regions and characters (the builder-1 run), whose layout prompt overgo renders to ComfyUI's byte for
// byte (moss-real-prompt): on the workflow's saved empty prompt and an empty builder they had kept 1.61 and 0.93 dB of
// their latents, near a reseeded draw's, where the real prompt keeps 16.62 dB. Mode 3's layout prompt parts its draw
// the most of the sections (3.78 dB), well over its reseeded draw (-2.46 dB); a picture or depth map holds its section.
var goldenFloors = map[string]goldenFloor{
	"text to image":                 {latent: 16.62 - goldenFloorMargin, picture: 38.47 - goldenFloorMargin},
	"prompt mode 2":                 {latent: 15.57 - goldenFloorMargin, picture: 32.73 - goldenFloorMargin},
	"prompt mode 3":                 {latent: 3.78 - goldenFloorMargin, picture: 17.98 - goldenFloorMargin},
	"image to image at denoise 0.6": {latent: 17.91 - goldenFloorMargin, picture: 32.94 - goldenFloorMargin},
	"depth control":                 {latent: 14.13 - goldenFloorMargin, picture: 28.08 - goldenFloorMargin},
}

// realPrompt is the prompt text to image and its Turbo section were recorded on (the real-1 and real-2 runs).
const realPrompt = "A red fox sitting in fresh snow at the edge of a birch forest at dawn, soft golden light on its fur, its breath visible in the cold air"

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

// goldenCases are the workflow's sections that run MOSS's 50 steps and its prompt modes, each recorded at 1024 x 576;
// image to image at denoise 1 starts from the noise alone, the same run as text to image, so it is not a case of its own
// (moss-real-prompt).
var goldenCases = []goldenCase{
	{name: "text to image", run: "real-1", slug: "text-to-image", sampler: "598", finished: "482", control: reseeded,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			if recording.Set["670.megapixels"] != 0.5625 || recording.Set["459.value"] != realPrompt || len(recording.Texts["436"]) != 1 || recording.Texts["436"][0] != realPrompt {
				t.Fatalf("the recording set %v and used the prompt %v; want the canvas at 1024 x 576 and the real prompt", recording.Set, recording.Texts["436"])
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
	{name: "prompt mode 3", run: "builder-1", slug: "text-to-image", sampler: "598", finished: "482", control: reseeded,
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			builder := recording.inputs("450")
			field := func(name string) string { value, _ := builder[name].(string); return value }
			var palette []string
			var regions []LayoutRegion
			testcheck.NoError(t, json.Unmarshal([]byte(field("style_palette_data")), &palette))
			testcheck.NoError(t, json.Unmarshal([]byte(field("elements_data")), &regions))
			characters, _ := recording.inputs("707:702:435")["character_boxes"].(string)
			if recording.Set["483.value"] != float64(3) || field("high_level_description") == "" || len(palette) == 0 || len(regions) == 0 || characters == "" {
				t.Fatalf("the recording set %v; want mode 3 with the builder's fields, palette, regions and characters filled", recording.Set)
			}
			request := textSection(t, recording, "598", "436", "592", "599")
			request.Prompt = ""
			request.Layout = &LayoutDraft{Description: field("high_level_description"), Background: field("background"), Style: field("style"),
				Photo: field("style.photo"), ArtStyle: field("style.art_style"), Aesthetics: field("aesthetics"), Lighting: field("lighting"), Medium: field("medium"),
				Palette: palette, Regions: regions, Characters: characters}
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
	// What needs no draw is held first, on any machine (SQA-410): a reference of no signal is no reading, and a case
	// that declares no floor is refused before any case is drawn.
	t.Run("refusals", func(t *testing.T) {
		if _, err := decibels([]float32{0.5, -0.25}, []float32{0, 0}); err == nil {
			t.Error("a reference of no signal was read as an agreement")
		}
		if agreement, err := decibels([]float32{0.5, -0.25}, []float32{0.5, -0.25}); err != nil || !math.IsInf(agreement, 1) {
			t.Errorf("agreement with a reference that has signal read %v (%v); want +Inf where nothing differs", agreement, err)
		}
		if missing := undeclaredFloors(append(slices.Clone(goldenCases), goldenCase{name: "a case with no floor"}), goldenFloors); !slices.Equal(missing, []string{"a case with no floor"}) {
			t.Errorf("the cases with no floor read %v; want the one added", missing)
		}
	})
	cudatest.Require(t)
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	holdGoldenFloors(t, root, profile, goldenCases, goldenFloors)
	t.Run("own size", func(t *testing.T) { ownSizeReading(t, root, profile) })
}

// undeclaredFloors names each case that declares no floor.
func undeclaredFloors(cases []goldenCase, floors map[string]goldenFloor) []string {
	var missing []string
	for _, golden := range cases {
		if _, declared := floors[golden.name]; !declared {
			missing = append(missing, golden.name)
		}
	}
	return missing
}

// holdGoldenFloors draws each case as its recording asks it, and with its input changed, on the root, and holds its
// final latent and finished picture to its floors, its changed draw's latent below. Every case's floor is read before
// any case is drawn, and each case is held as soon as it is read (SQA-410: a missing floor had cost five draws, and a
// later case's failure had hidden an earlier case's reading).
func holdGoldenFloors(t *testing.T, root string, profile Profile, cases []goldenCase, floors map[string]goldenFloor) {
	t.Helper()
	if missing := undeclaredFloors(cases, floors); len(missing) > 0 {
		t.Fatalf("%v declare no floor", missing)
	}
	for _, golden := range cases {
		reading, floor := readGoldenCase(t, root, profile, golden), floors[golden.name]
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

// readGoldenCases reads each case in turn (readGoldenCase).
func readGoldenCases(t *testing.T, root string, profile Profile, cases []goldenCase) map[string]goldenReading {
	t.Helper()
	readings := map[string]goldenReading{}
	for _, golden := range cases {
		readings[golden.name] = readGoldenCase(t, root, profile, golden)
	}
	return readings
}

// readGoldenCase draws a case as its recording asks it, and with its input changed, on the root, and reads its
// agreement with its recording. Its generator is closed as it returns, whatever stops it, and the changed draw keeps
// its final latent alone, its picture never decoded (SQA-410).
func readGoldenCase(t *testing.T, root string, profile Profile, golden goldenCase) goldenReading {
	t.Helper()
	ctx := t.Context()
	recording := readComfySection(t, golden.run, golden.slug)
	request := golden.request(t, recording)
	wantLatent, wantPicture := recording.latent(t, golden.sampler), recording.picture(t, golden.finished)
	generator := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	drawn := func(request Request, decode bool) (latent []float32, picture []byte) {
		t.Helper()
		session := testcheck.Must(generator.prepare(ctx, request))(t)
		testcheck.Must(generator.integrate(ctx, session))(t)
		latent = testcheck.Must(generator.rotary.latent.outputs.CopyToHost(ctx, generator.rotary.latent.node))(t).Data
		if decode {
			picture = testcheck.Must(generator.decode(ctx, session))(t).Data
		}
		return latent, picture
	}
	latent, picture := drawn(request, true)
	reading := goldenReading{latent: decibelsOverSquares(t, latent, wantLatent), picture: pictureDecibels(t, picture, wantPicture)}
	changed := golden.control(t, request)
	testcheck.NoError(t, generator.Reset(ctx, changed))
	controlLatent, _ := drawn(changed, false)
	reading.control = decibelsOverSquares(t, controlLatent, wantLatent)
	t.Logf("%s: the final latent keeps %.2f dB, the finished picture %.2f dB over %d-pixel squares; with its input changed the latent keeps %.2f dB",
		golden.name, reading.latent, reading.picture, pictureSquare, reading.control)
	return reading
}

// ownSizeReading reads the workflow's text to image section at its own size (the own-size recording: its canvas 16:9
// at 3 megapixels in multiples of 32, as ResolutionSelector 670 sizes it) on the card ComfyUI drew it on: overgo's time
// per step, read between samplings of 2 and 6 steps, its time from the prompt to the picture and the card's memory in
// use at most, total less free as ComfyUI's recorder reads it as the draw starts and the most its allocations rose by
// after it, beside ComfyUI's (the sampler's progress stamps, the run's milliseconds and the card's peak). A reading, not
// a floor.
func ownSizeReading(t *testing.T, root string, profile Profile) {
	ctx := t.Context()
	recording := readComfySection(t, "own-size", "text-to-image")
	canvas := recording.inputs("670")
	if len(recording.Set) != 0 || canvas["megapixels"] != float64(3) || canvas["multiple"] != float64(32) || canvas["aspect_ratio"] != "16:9 (Widescreen)" {
		t.Fatalf("the recording set %v over a canvas of %v; want the workflow's own 16:9 at 3 megapixels", recording.Set, canvas)
	}
	request := textSection(t, recording, "598", "436", "592", "599")
	request.Width, request.Height, request.Canvas = 0, 0, &media.Canvas{Aspect: "16:9", Megapixels: 3, Multiple: 32}
	// ComfyUI's step is read before the generator loads, so a recording that cannot give one costs no draw (SQA-477).
	comfyStep := recording.stepTime(t, "598")
	generator := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	// The card's use is read as ComfyUI's recorder reads it, its total less its free memory (comfy-reference's await,
	// from ComfyUI's /system_stats), as the drawing starts, and its peak as that and the most the process's allocations
	// rose by after it (driver.StagePeakBytes): the process's allocations alone, read beside ComfyUI's whole card, had
	// left out what the card's context and the card's other users hold (SQA-388), and total less free read only at a
	// stage's end misses the sampling's peak inside it.
	var before uint64
	driver.ResetStagePeakBytes()
	allocated := driver.StagePeakBytes()
	testcheck.NoError(t, generator.rotary.runtime.Do(ctx, func(state *device.State) error {
		free, total, err := state.Driver.MemInfo()
		before = total - free
		return err
	}))
	start := time.Now()
	session := testcheck.Must(generator.prepare(ctx, request))(t)
	testcheck.Must(generator.integrate(ctx, session))(t)
	picture := testcheck.Must(generator.decode(ctx, session))(t)
	total, used := time.Since(start), before+driver.StagePeakBytes()-allocated
	// A step is read as TestWhereAMossStepsTimeGoes reads it, between samplings of 2 and 6 steps, so the request's own
	// part, its weights' upload among it, is not charged to its steps (SQA-388: a whole sampling over its steps had read
	// 33.4 s of setup into 5.30 s a step).
	sampled := func(steps int) time.Duration {
		t.Helper()
		asked := testcheck.Must(request.Sized(CatalogIn(root)))(t)
		asked.Steps = steps
		testcheck.NoError(t, generator.Reset(ctx, asked))
		session := testcheck.Must(generator.prepare(ctx, asked))(t)
		from := time.Now()
		testcheck.Must(generator.integrate(ctx, session))(t)
		wall := time.Since(from)
		testcheck.Must(generator.decode(ctx, session))(t)
		return wall
	}
	step := (sampled(6) - sampled(2)) / 4
	if picture.Width == 0 || step <= 0 {
		t.Fatalf("overgo drew %dx%d and read a step of %v; want a picture and a step", picture.Width, picture.Height, step)
	}
	t.Logf("at %dx%d: overgo %v a step, %v from the prompt to the picture, %.2f GiB of the card in use at most; ComfyUI %v a step, %v, %.2f GiB",
		picture.Width, picture.Height, step.Round(time.Millisecond), total.Round(time.Millisecond), float64(used)/(1<<30),
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
	// Two pictures of different sizes are refused before they are scored (SQA-410): squareColourMeans drops a picture's
	// partial squares, so one a column wider had held as many means and agreed.
	gotSize, wantSize := testcheck.Must(png.DecodeConfig(bytes.NewReader(got)))(t), testcheck.Must(png.DecodeConfig(bytes.NewReader(want)))(t)
	if gotSize.Width != wantSize.Width || gotSize.Height != wantSize.Height {
		t.Fatalf("the pictures are %dx%d and %dx%d; want one size", gotSize.Width, gotSize.Height, wantSize.Width, wantSize.Height)
	}
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
