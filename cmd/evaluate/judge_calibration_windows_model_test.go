//go:build windows && modeltest

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/runrecord"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const (
	// calibrationLevel is the one-sided tail at which a degradation is held to lose to the outputs it came from,
	// declared before any run.
	calibrationLevel = 0.05
	// calibrationPictures and calibrationClips are how many of a suite's own outputs are degraded: the first of each
	// suite, every question asked of each kept.
	calibrationPictures, calibrationClips = 12, 10
	// blurShare is the box radius of the blur as a share of a picture's shorter side, three passes making it close to a
	// Gaussian: a sixteenth keeps a picture's layout and takes its detail.
	blurShare = 16
	// blurPasses is how many box passes the blur makes.
	blurPasses = 3
	// noiseDeviation is the deviation of the noise added to each channel, of a channel's full 255.
	noiseDeviation = 64
	// channelFull is a channel's full value as image colours report it.
	channelFull = 0xffff
)

// degradation is one way an output is made worse: each frame changed alike, the clip as a whole (frozen), or the
// output replaced by another prompt's (swapped). seenInPictures and seenInClips say the judge is held to scoring it
// lower on the suite's own questions: what it was measured to see on 2026-11-09, 12 pictures and 10 clips
// (judge-calibration/sensitivity). One it was measured blind to is measured again and logged, and is left to the
// instruments that see it (judge-calibration/fidelity, judge-calibration/stillness), so a judge that comes to see it
// fails nothing.
type degradation struct {
	name                        string
	frame                       func(picture image.Image, seed uint64) image.Image
	clip                        bool
	swapped                     bool
	seenInPictures, seenInClips bool
}

var degradations = []degradation{
	// Pictures 11 lower, 0 higher, 1 the same; clips 3, 0, 7.
	{name: "blurred", frame: func(picture image.Image, _ uint64) image.Image { return blurred(picture) }, seenInPictures: true},
	// Pictures 1, 1, 10; clips 0, 0, 10.
	{name: "noised", frame: noised},
	// Clips 1, 0, 9: a clip's caption questions ask what it shows, which one frame shows.
	{name: "frozen", clip: true},
	// Pictures 11, 0, 1; clips 8, 0, 2.
	{name: "swapped", swapped: true, seenInPictures: true, seenInClips: true},
}

// unchanged names the outputs the degradations are read against: each decoded and written again as it was, so a
// degraded output differs from it only by its degradation.
const unchanged = "unchanged"

// calibrationSuites are the suites whose own outputs are degraded: the first 12 pictures Krea-2-Turbo drew for the
// TIFA suite and the first 10 clips Wan2.1-T2V-1.3B drew at 81 frames and 832 x 480.
func calibrationSuites(roots dataroot.Roots) (pictures, clips [3]string) {
	return [3]string{filepath.Join(roots.Datasets, "tifa_v1.0", "krea-2-turbo-suite.json"), filepath.Join(roots.Datasets, "tifa_v1.0", "krea-2-turbo"), "pictures"},
		[3]string{filepath.Join(roots.Datasets, "VidGen", "wan2.1-generated-832-suite.json"), filepath.Join(roots.Datasets, "VidGen", "wan2.1-generated-832"), "clips"}
}

// TestTheJudgeScoresDegradedOutputsLower measures what the judge sees of what it is used to measure. The calibration
// suites' outputs are each written again unchanged, blurred, noised, every frame frozen on the first (a clip only), and
// swapped for the next output, another prompt's; Qwen3.8-27B answers each output's questions of every version. A
// version loses to the unchanged one at a picture or clip whose questions it answers fewer of, and wins where more,
// the questions of one output read together, since they ask about one picture. A degradation the judge was measured
// to see loses by the exact paired test at calibrationLevel, declared before the run: another prompt's picture or
// clip, and a blurred picture. It was measured blind to noise in either, to a frozen clip and to a blurred clip, since
// its questions ask what an output shows, not how well: its quality cells are prompt adherence, and those losses are
// read by instruments that see them. The judge's answers are kept in a store of the test's own, so no reading of these
// suites joins the store's quality cells.
func TestTheJudgeScoresDegradedOutputsLower(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	root := t.TempDir()
	pictureSuite, clipSuite := calibrationSuites(roots)
	pictures := degradedSuite(t, ctx, ffmpeg, pictureSuite, root, calibrationPictures, nil, degradationNames(true))
	clips := degradedSuite(t, ctx, ffmpeg, clipSuite, root, calibrationClips, nil, degradationNames(true))
	judge := openCalibration(t, ctx, roots, root)
	for video, suite := range []evaluation.GeneratedAnswerSuite{pictures, clips} {
		right := judge.answer(t, ctx, suite)
		for _, worse := range degradations {
			if _, asked := right[worse.name]; !asked {
				continue
			}
			lost, gained, same, tail := lowerAt(right, worse.name)
			seen := worse.seenInPictures
			if video == 1 {
				seen = worse.seenInClips
			}
			t.Logf("%s %s: lower at %d outputs, higher at %d, the same at %d; chance %.3g where either way is as likely (held to: %t)",
				suite.Source, worse.name, lost, gained, same, tail, seen)
			if seen && tail > calibrationLevel {
				t.Errorf("the judge does not score %s %s outputs lower: %d lower, %d higher, %d the same, chance %.3g against %.3g",
					worse.name, suite.Source, lost, gained, same, tail, calibrationLevel)
			}
		}
	}
}

// fidelityQuestions are what the judge is asked of how well an output is drawn, each answered yes of a clean one:
// whether it is sharp and whether it is clean, and of a clip whether its scene moves.
var fidelityQuestions = []string{
	"Is the image sharp, with crisp edges and clear fine detail?",
	"Is the image clean, free of noise, grain and speckle?",
}

// motionQuestion is asked of a clip alone.
const motionQuestion = "Does the scene move over the course of the video, rather than staying one still frame?"

// TestTheJudgeSeesFidelityWhenAsked measures whether the judge sees, when asked directly, the losses its content
// questions are blind to (TestTheJudgeScoresDegradedOutputsLower): the calibration suites' outputs, unchanged, blurred,
// noised and frozen (a clip only), are each asked fidelityQuestions, and a clip motionQuestion, each answered yes of
// the unchanged output. Each degradation answers fewer of them yes than the unchanged output, output by output, by the
// exact paired test at calibrationLevel, declared before the run.
func TestTheJudgeSeesFidelityWhenAsked(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	root := t.TempDir()
	pictureSuite, clipSuite := calibrationSuites(roots)
	worse := []string{"blurred", "noised", "frozen"}
	pictures := degradedSuite(t, ctx, ffmpeg, pictureSuite, root, calibrationPictures, fidelityCases(fidelityQuestions), worse)
	clips := degradedSuite(t, ctx, ffmpeg, clipSuite, root, calibrationClips, fidelityCases(append(slices.Clone(fidelityQuestions), motionQuestion)), worse)
	judge := openCalibration(t, ctx, roots, root)
	for _, suite := range []evaluation.GeneratedAnswerSuite{pictures, clips} {
		right := judge.answer(t, ctx, suite)
		for _, version := range worse {
			if _, asked := right[version]; !asked {
				continue
			}
			lost, gained, same, tail := lowerAt(right, version)
			t.Logf("%s %s, asked how well it is drawn: lower at %d outputs, higher at %d, the same at %d; chance %.3g",
				suite.Source, version, lost, gained, same, tail)
			if tail > calibrationLevel {
				t.Errorf("asked directly, the judge does not score %s %s outputs lower: %d lower, %d higher, %d the same, chance %.3g against %.3g",
					version, suite.Source, lost, gained, same, tail, calibrationLevel)
			}
		}
	}
}

// fidelityCases asks each output questions in place of its suite's own: each a yes-or-no case about the output its
// first case shows, answered yes, with choices where the suite's cases have them.
func fidelityCases(questions []string) func(asked []evaluation.GeneratedAnswerCase) []evaluation.GeneratedAnswerCase {
	return func(asked []evaluation.GeneratedAnswerCase) []evaluation.GeneratedAnswerCase {
		first := asked[0]
		header, _, _ := strings.Cut(first.Prompt, "\n")
		base := first.Name[:strings.LastIndex(first.Name, "/")]
		cases := make([]evaluation.GeneratedAnswerCase, len(questions))
		for index, question := range questions {
			cases[index] = evaluation.GeneratedAnswerCase{
				Name: base + "/fidelity/" + string(rune('a'+index)), MaxTokens: 1, Answers: []string{"A"},
				Prompt: header + "\n" + question + "\nA. yes\nB. no\nAnswer with the letter only.",
			}
			if len(first.Choices) > 0 {
				cases[index].Choices = []string{"A", "B"}
			}
		}
		return cases
	}
}

// calibration is the judge opened over the outputs written under a root, its answers kept in a store of the test's
// own.
type calibration struct {
	session *projectionSession
	answers *overgodb.Store
}

// openCalibration opens Qwen3.8-27B with its projector over the outputs under root, with the models and recipes the
// data root's store holds, and its answers kept apart: their records name the judge's definition, recipes and model as
// their parents, which the store of the test's own holds as bare authorities.
func openCalibration(t *testing.T, ctx context.Context, roots dataroot.Roots, root string) calibration {
	t.Helper()
	store := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
	answers := overgodbtest.Open(t, t.TempDir())
	judge := filepath.Join(roots.ResolveModelPath("Qwen3.8-27B-UD-Q6_K"), "Qwen3.8-27B-UD-Q6_K.gguf")
	session := testcheck.Must(openProjectionInto(ctx, store, answers, manifest{Repository: roots.Store, CodeCommit: runrecord.RunningCommit()},
		modelRequest{Path: judge, Projector: filepath.Join(filepath.Dir(judge), "mmproj-BF16.gguf"), MediaRoot: root}))(t)
	t.Cleanup(func() { testcheck.NoError(t, session.Close()) })
	authorities := session.campaign.Authorities()
	judged := testcheck.Must(projectionIdentity(ctx, store, projectionRuntime{runner: session.runner}))(t)
	var parents []artifact.Descriptor
	for _, id := range []artifact.ID{authorities.ModelDefinition, authorities.RuntimeRecipe, session.model, judged.Model, judged.Recipe, judged.Projection} {
		if id.Valid() && !slices.ContainsFunc(parents, func(held artifact.Descriptor) bool { return held.ID == id }) {
			parents = append(parents, artifact.Descriptor{ID: id})
		}
	}
	testcheck.Must(answers.Commit(ctx, artifact.Batch{Key: "judge-calibration/authorities", Artifacts: parents}))(t)
	return calibration{session: session, answers: answers}
}

// answer has the judge answer every case of suite and counts, by version and output, the cases answered as their
// answer.
func (judge calibration) answer(t *testing.T, ctx context.Context, suite evaluation.GeneratedAnswerSuite) map[string]map[string]int {
	t.Helper()
	compiled := testcheck.Must(judge.session.campaign.CompileSuite(ctx, testcheck.Must(json.Marshal(suite))(t)))(t)
	result := testcheck.Must(judge.session.campaign.Evaluate(ctx, compiled))(t)
	outcomes, read := evaluation.CaseOutcomes(ctx, judge.answers, result.Evaluation)
	if !read || len(outcomes) != len(suite.Cases) {
		t.Fatalf("the judge's %d answers to the %d cases of %s cannot be read", len(outcomes), len(suite.Cases), suite.Source)
	}
	right := map[string]map[string]int{}
	for _, testCase := range suite.Cases {
		version, output, _ := strings.Cut(testCase.Group, "/")
		if right[version] == nil {
			right[version] = map[string]int{}
		}
		// Every output is counted, those answered nothing right at zero.
		answered := 0
		if outcomes[testCase.Name] {
			answered = 1
		}
		right[version][output] += answered
	}
	return right
}

// lowerAt compares version with the unchanged outputs output by output: where it answers fewer cases right, more, or
// as many, and the exact paired test's one-sided tail of so many fewer.
func lowerAt(right map[string]map[string]int, version string) (lost, gained, same uint64, tail float64) {
	for output, kept := range right[unchanged] {
		switch answered := right[version][output]; {
		case answered < kept:
			lost++
		case answered > kept:
			gained++
		default:
			same++
		}
	}
	return lost, gained, same, tensorstats.ExactPairedBinary(lost, gained, same).Tail
}

// degradedSuite writes, under root, each of the first count outputs the suite at suite[0] asks about, read from the
// directory suite[1], unchanged and degraded each of the ways named that applies, in the folder suite[2]; it returns
// the suite asking each version of every output the questions ask gives of the output's own cases, or those cases
// where ask is nil, the version and output in each case's group.
func degradedSuite(t *testing.T, ctx context.Context, ffmpeg string, suite [3]string, root string, count int,
	ask func([]evaluation.GeneratedAnswerCase) []evaluation.GeneratedAnswerCase, named []string,
) evaluation.GeneratedAnswerSuite {
	t.Helper()
	path, directory, kind := suite[0], suite[1], suite[2]
	var source evaluation.GeneratedAnswerSuite
	testcheck.NoError(t, jsonfile.DecodeStrict(path, &source))
	var outputs []string
	asked := map[string][]evaluation.GeneratedAnswerCase{}
	video := false
	for _, testCase := range source.Cases {
		mediaKind, name, _, framed := evaluation.ParseMediaPrompt(testCase.Prompt)
		if !framed {
			t.Fatalf("%s asks %s about no output", path, testCase.Name)
		}
		video = mediaKind == evaluation.MediaVideo
		if _, kept := asked[name]; !kept && len(outputs) == count {
			continue
		}
		if !slices.Contains(outputs, name) {
			outputs = append(outputs, name)
		}
		asked[name] = append(asked[name], testCase)
	}
	if ask != nil {
		for output, cases := range asked {
			asked[output] = ask(cases)
		}
	}
	versions := []string{unchanged}
	for _, name := range degradationNames(video) {
		if slices.Contains(named, name) {
			versions = append(versions, name)
		}
	}
	written := map[string]map[string]string{}
	for _, output := range outputs {
		file := testcheck.Must(clipFile(directory, output))(t)
		var frames []image.Image
		if video {
			frames = testcheck.Must(media.DecodeVideoFile(ctx, file, ffmpeg, motionFrameRate, motionFrames))(t)
		} else {
			frames = []image.Image{testcheck.Must(png.Decode(bytes.NewReader(testcheck.Must(os.ReadFile(file))(t))))(t)}
		}
		write := func(version string, frames []image.Image) {
			name := filepath.ToSlash(filepath.Join(kind, version, output))
			if video {
				name += ".gif"
				writeCalibrationClip(t, ctx, filepath.Join(root, name), frames)
			} else {
				if !strings.HasSuffix(name, ".png") {
					name += ".png"
				}
				var encoded bytes.Buffer
				testcheck.NoError(t, png.Encode(&encoded, frames[0]))
				testcheck.NoError(t, os.MkdirAll(filepath.Dir(filepath.Join(root, name)), renderDirectoryMode))
				testcheck.NoError(t, os.WriteFile(filepath.Join(root, name), encoded.Bytes(), renderFileMode))
			}
			if written[version] == nil {
				written[version] = map[string]string{}
			}
			written[version][output] = name
		}
		write(unchanged, frames)
		for _, worse := range degradations {
			switch {
			case worse.swapped || !slices.Contains(versions, worse.name):
			case worse.clip:
				frozen := make([]image.Image, len(frames))
				for index := range frozen {
					frozen[index] = frames[0]
				}
				write(worse.name, frozen)
			default:
				changed := make([]image.Image, len(frames))
				for index, frame := range frames {
					changed[index] = worse.frame(frame, uint64(index))
				}
				write(worse.name, changed)
			}
		}
	}
	// The judge answers these as itself, a vision-language model shown fixed media, so the suite names no judge.
	degraded := source
	degraded.Cases, degraded.Motion, degraded.Readings, degraded.Judge = nil, nil, nil, ""
	for _, version := range versions {
		for at, output := range outputs {
			shown := written[version][output]
			if version == "swapped" {
				shown = written[unchanged][outputs[(at+1)%len(outputs)]]
			}
			for _, testCase := range asked[output] {
				testCase.Name = version + "/" + testCase.Name
				testCase.Prompt = strings.Replace(testCase.Prompt, "name="+output+">", "name="+shown+">", 1)
				testCase.Group = version + "/" + output
				degraded.Cases = append(degraded.Cases, testCase)
			}
		}
	}
	return degraded
}

// degradationNames are the degradations that apply to a picture, or to a clip.
func degradationNames(video bool) []string {
	var names []string
	for _, worse := range degradations {
		if !worse.clip || video {
			names = append(names, worse.name)
		}
	}
	return names
}

// writeCalibrationClip writes frames as a GIF, as the clips the suite asks about were written, through the encoder a
// served clip is.
func writeCalibrationClip(t *testing.T, ctx context.Context, path string, frames []image.Image) {
	t.Helper()
	video := testcheck.Must(latentvideo.WriteClip(ctx, "", latentvideo.ClipFinish{}, motionFrameRate, latentvideo.UnitPixels, func(sink latentvideo.VideoFrameSink) error {
		for index, frame := range frames {
			bounds := frame.Bounds()
			width, height := bounds.Dx(), bounds.Dy()
			plane := width * height
			pixels := make([]float32, media.RGBChannels*plane)
			for at := range plane {
				red, green, blue, _ := frame.At(bounds.Min.X+at%width, bounds.Min.Y+at/width).RGBA()
				for channel, value := range []uint32{red, green, blue} {
					pixels[channel*plane+at] = float32(value) / channelFull
				}
			}
			if err := sink(index, pixels, height, width); err != nil {
				return err
			}
		}
		return nil
	}))(t)
	testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), renderDirectoryMode))
	testcheck.NoError(t, os.WriteFile(path, video.Data, renderFileMode))
}

// blurred is the picture smoothed by blurPasses passes of a box whose radius is its shorter side over blurShare.
func blurred(picture image.Image) image.Image {
	bounds := picture.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	radius := max(min(width, height)/blurShare, 1)
	channels := make([][]float64, media.RGBChannels)
	for channel := range channels {
		channels[channel] = make([]float64, width*height)
	}
	for at := range width * height {
		red, green, blue, _ := picture.At(bounds.Min.X+at%width, bounds.Min.Y+at/width).RGBA()
		for channel, value := range []uint32{red, green, blue} {
			channels[channel][at] = float64(value)
		}
	}
	// box averages one line of values over a window of radius either side, cut at the line's ends, from running sums.
	box := func(line []float64) []float64 {
		sums := make([]float64, len(line)+1)
		for at, value := range line {
			sums[at+1] = sums[at] + value
		}
		out := make([]float64, len(line))
		for at := range line {
			low, high := max(at-radius, 0), min(at+radius+1, len(line))
			out[at] = (sums[high] - sums[low]) / float64(high-low)
		}
		return out
	}
	for range blurPasses {
		for _, values := range channels {
			for y := range height {
				copy(values[y*width:(y+1)*width], box(values[y*width:(y+1)*width]))
			}
			column := make([]float64, height)
			for x := range width {
				for y := range height {
					column[y] = values[y*width+x]
				}
				for y, value := range box(column) {
					values[y*width+x] = value
				}
			}
		}
	}
	out := image.NewNRGBA(image.Rect(0, 0, width, height))
	for at := range width * height {
		out.SetNRGBA(at%width, at/width, color.NRGBA{
			R: uint8(channels[0][at] / channelFull * math.MaxUint8), G: uint8(channels[1][at] / channelFull * math.MaxUint8),
			B: uint8(channels[2][at] / channelFull * math.MaxUint8), A: math.MaxUint8,
		})
	}
	return out
}

// noised is the picture with Gaussian noise of noiseDeviation added to each channel, drawn from a stream seeded by seed.
func noised(picture image.Image, seed uint64) image.Image {
	bounds := picture.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	random := rand.New(rand.NewPCG(seed, seed+1))
	shift := func(value uint32) uint8 {
		return uint8(min(max(float64(value)/channelFull*math.MaxUint8+random.NormFloat64()*noiseDeviation, 0), math.MaxUint8))
	}
	for y := range bounds.Dy() {
		for x := range bounds.Dx() {
			red, green, blue, _ := picture.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			out.SetNRGBA(x, y, color.NRGBA{R: shift(red), G: shift(green), B: shift(blue), A: math.MaxUint8})
		}
	}
	return out
}
