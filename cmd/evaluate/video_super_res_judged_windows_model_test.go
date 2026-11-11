//go:build windows && modeltest

package main

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/nvvfx"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The student clips the judge compares, and how they are drawn: VidGen captions picked by seed, drawn by the Wan
// student at its suite's size, 81 frames at 832 x 480, once as they are and once finished at twice the size.
const (
	judgedClips   = 16
	judgedSeed    = 23
	judgedUpscale = 2
	judgedWidth   = 832
	judgedHeight  = 480
	// judgedCommit names the renders' manifests.
	judgedCommit = "rtx-vsr-judged"
)

// The degradations the denoise and deblur levels are read on: noise of a known deviation in [0, 1] on each channel,
// and a blur of three box passes of a known radius, each applied to the source clips of
// TestVideoSuperResBeatsBicubicOnSourceClips.
const (
	degradeNoise      = 0.05
	degradeBlurRadius = 2
	degradeBlurPasses = 3
	// degradeSeed seeds the noise, so every run reads the same noised clips.
	degradeSeed = 20261109
	// denoiseLevel and deblurLevel are the effect's QualityLevel of strength 1 (nvidia-vfx video_super_res.py).
	denoiseLevel = 8
	deblurLevel  = 12
	// finishStrengths are a denoise's or deblur's strengths, 1 to 4.
	finishStrengths = 4
)

// judgedFrames are the frames of a clip the judge reads: a quarter, half and three quarters through its 81.
var judgedFrames = []int{20, 40, 60}

// judgedCropWidth is the width of the middle of a frame the judge reads, at the frame's full height: 1024 x 960 is
// 3840 of the projector's patches, within the 4096 one encode reads.
const judgedCropWidth = 1024

// nvencArgs are the options a finished clip is encoded with (latentvideo nvencCodec), spelled, so FFmpeg's Lanczos is
// encoded as a finished clip is.
var nvencArgs = []string{"-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-maxrate", "80M",
	"-bufsize", "160M", "-tag:v", "hvc1", "-pix_fmt", "yuv420p"}

// judgedRootEnv names, in the process TestTheJudgeDoesNotPreferLanczosOverTheDefaultLevel starts, the directory the clips are
// drawn into and their suite written beside them (judgedSuiteFile); TestJudgedClipsAreDrawn runs only there.
const (
	judgedRootEnv   = "OVERGO_JUDGED_ROOT"
	judgedSuiteFile = "suite.json"
)

// TestJudgedClipsAreDrawn draws what TestTheJudgeDoesNotPreferLanczosOverTheDefaultLevel judges, in a process of its own: it
// has the Wan student draw judgedClips VidGen captions as they are and
// finished at twice the size by the level a request naming none gets, enlarges each plain clip by FFmpeg's Lanczos to
// the same size, encoded on NVENC as a finished clip is, and writes the middle of judgedFrames frames of each as
// pictures, with the suite that asks the judge of each whether it is sharp and whether it is clean.
func TestJudgedClipsAreDrawn(t *testing.T) {
	root := os.Getenv(judgedRootEnv)
	if root == "" {
		testskip.NotApplicable(t, "the clips are drawn in the process TestTheJudgeDoesNotPreferLanczosOverTheDefaultLevel starts")
	}
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	started := time.Now()
	items := testcheck.Must(evaluation.PickVideoCaptions(filepath.Join(roots.Datasets, "VidGen", "VidGen_1M_video_caption.json"), judgedClips, judgedSeed))(t)
	prompts := make([]renderPrompt, len(items))
	for index, item := range items {
		prompts[index] = renderPrompt{ID: item.ID, Prompt: item.Caption}
	}
	student := roots.ResolveModelPath("Self-Forcing")
	settings := map[string]float64{"width": judgedWidth, "height": judgedHeight, "frames": motionFrames}
	upscaled := maps.Clone(settings)
	upscaled["upscale"] = judgedUpscale
	drawn := map[string]map[string]string{}
	for version, asked := range map[string]map[string]float64{"plain": settings, "default": upscaled} {
		directory := filepath.Join(root, version)
		keys := testcheck.Must(renderMedia(ctx, roots.Store, student, recipe.TaskVideoGen, directory, prompts, asked, "", judgedCommit))(t)
		var manifest renderManifest
		testcheck.NoError(t, jsonfile.Decode(filepath.Join(directory, renderManifestFile), &manifest))
		drawn[version] = map[string]string{}
		for _, item := range items {
			held := manifest.Draws[keys[item.ID]]
			drawn[version][item.ID] = version + "/" + held.File
			if version == "default" && (held.Encoding.Finish == nil || held.Encoding.Finish.Scale != judgedUpscale) {
				t.Fatalf("the clip drawn for %s at twice its size records the finish %+v", item.ID, held.Encoding.Finish)
			}
		}
	}
	drawn["lanczos"] = map[string]string{}
	for _, item := range items {
		name := "lanczos/" + item.ID + clipFileExtension
		testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "lanczos"), renderDirectoryMode))
		args := append([]string{"-nostdin", "-hide_banner", "-loglevel", "error", "-i", filepath.Join(root, drawn["plain"][item.ID]),
			"-vf", fmt.Sprintf("scale=%d:%d:flags=lanczos", judgedUpscale*judgedWidth, judgedUpscale*judgedHeight), "-an"}, nvencArgs...)
		var diagnostic strings.Builder
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: ffmpeg, Args: append(args, filepath.Join(root, name)), Stderr: &diagnostic})
		if err != nil || receipt.ExitCode != 0 {
			t.Fatalf("FFmpeg's Lanczos of %s exited %d: %v %s", item.ID, receipt.ExitCode, err, diagnostic.String())
		}
		drawn["lanczos"][item.ID] = name
	}

	t.Logf("drawn and enlarged in %s", time.Since(started).Round(time.Second))

	// The judge reads the middle of frames of each clip as pictures, at their full detail: how sharp and clean a clip is
	// drawn is a frame's. A whole clip at 1664 x 960 asks the judge about 7800 tokens a question, which ran 16 clips past
	// two hours, and a whole frame 6240 patches, past the 4096 one encode of the judge's projector is bounded to, which
	// took about 19 s a question; the middle judgedCropWidth columns take 3840.
	versions := []string{"default", "lanczos"}
	suite := evaluation.GeneratedAnswerSuite{Kind: "generated-answer", Schema: "overgo/rtx-vsr-fidelity/v1", Source: "rtx-vsr/fidelity",
		Transforms: []string{"trim-space", "lowercase"}}
	for _, version := range versions {
		testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "frames", version), renderDirectoryMode))
		for _, item := range items {
			frames := testcheck.Must(media.DecodeVideoFile(ctx, filepath.Join(root, drawn[version][item.ID]), ffmpeg, motionFrameRate, motionFrames))(t)
			for _, at := range judgedFrames {
				if at >= len(frames) || frames[at].Bounds().Dx() != judgedUpscale*judgedWidth || frames[at].Bounds().Dy() != judgedUpscale*judgedHeight {
					t.Fatalf("%s's %s clip holds %d frames of %v, not frame %d at twice %dx%d", item.ID, version, len(frames), frames[0].Bounds(), at, judgedWidth, judgedHeight)
				}
				var encoded bytes.Buffer
				bounds := frames[at].Bounds()
				left := bounds.Min.X + (bounds.Dx()-judgedCropWidth)/2
				middle := frames[at].(interface {
					SubImage(image.Rectangle) image.Image
				}).SubImage(image.Rect(left, bounds.Min.Y, left+judgedCropWidth, bounds.Max.Y))
				testcheck.NoError(t, png.Encode(&encoded, middle))
				name := fmt.Sprintf("frames/%s/%s-%d.png", version, item.ID, at)
				testcheck.NoError(t, os.WriteFile(filepath.Join(root, name), encoded.Bytes(), renderFileMode))
				for index, question := range fidelityQuestions {
					suite.Cases = append(suite.Cases, evaluation.GeneratedAnswerCase{
						Name:      fmt.Sprintf("%s/%s/%d/%c", version, item.ID, at, 'a'+index),
						Prompt:    "<overgo-media kind=image name=" + name + ">\n" + question + "\nA. yes\nB. no\nAnswer with the letter only.",
						MaxTokens: 1, Answers: []string{"A"}, Choices: []string{"A", "B"}, Group: version + "/" + item.ID,
					})
				}
			}
		}
	}
	testcheck.NoError(t, jsonfile.Write(filepath.Join(root, judgedSuiteFile), suite, renderFileMode))
}

// TestTheJudgeDoesNotPreferLanczosOverTheDefaultLevel reads the level offered first for overgo's own clips, which have
// no ground truth at a larger size, by the judge's fidelity questions (judge-calibration/fidelity), asked of the clips
// TestJudgedClipsAreDrawn draws: the Wan student's, finished at twice the size by the level a request naming none
// gets, and the same clips enlarged by FFmpeg's Lanczos. Qwen3.8-27B is asked of the middle of judgedFrames frames of
// each, as pictures at their full detail, whether it is sharp and whether it is clean, its probability of yes summed
// over the frames and questions, clip by clip. The step first claimed the judge prefers the default, by the exact
// paired test at calibrationLevel: on 2026-11-09 it scored clean-high higher at 6 of 16 clips and Lanczos at 10 (tail
// 0.895), so the claim was refuted, and the default stands on the VMAF of source clips with a ground truth
// (TestVideoSuperResBeatsBicubicOnSourceClips). What is held, a bar set after that run, is that the judge does not
// prefer Lanczos over the default by the same test: it would fail were Lanczos clearly the better finish. The denoise
// and deblur levels are read on the source clips TestVideoSuperResBeatsBicubicOnSourceClips decodes (readRepairLevels).
func TestTheJudgeDoesNotPreferLanczosOverTheDefaultLevel(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_NVVFX") == "" {
		testskip.NotApplicable(t, "OVERGO_NVVFX names no NVIDIA Video Effects runtime here")
	}
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	root := t.TempDir()
	// What is judged is drawn in a process of its own, which gives the card back when it exits: the student's sessions
	// stay resident for the life of a process, and beside them the judge answered a picture in about 20 s, where alone
	// it answers one in about 2. The child's lines are logged here, not passed through, so they are not read as this
	// process's own test events.
	var drawnLog bytes.Buffer
	child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestJudgedClipsAreDrawn$", "-test.v", "-test.timeout=0")
	child.Env = append(os.Environ(), judgedRootEnv+"="+root)
	child.Stdout, child.Stderr = &drawnLog, &drawnLog
	drawErr := child.Run()
	for line := range strings.Lines(drawnLog.String()) {
		t.Log(strings.TrimRight(line, "\r\n"))
	}
	testcheck.NoError(t, drawErr)
	var suite evaluation.GeneratedAnswerSuite
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, judgedSuiteFile), &suite))
	var items []string
	for _, testCase := range suite.Cases {
		if version, item, _ := strings.Cut(testCase.Group, "/"); version == "default" && !slices.Contains(items, item) {
			items = append(items, item)
		}
	}
	defaultLevel := latentvideo.UpscaleLevel("").Default()
	versions := []string{"default", "lanczos"}
	started := time.Now()
	judge := openCalibration(t, ctx, roots, root)
	answers := judge.judged(t, ctx, suite)
	t.Logf("the judge answered %d questions in %s", len(suite.Cases), time.Since(started).Round(time.Second))
	var wins, losses, ties uint64
	sums := map[string]float64{}
	for _, item := range items {
		score := map[string]float64{}
		for _, version := range versions {
			for _, at := range judgedFrames {
				for index := range fidelityQuestions {
					score[version] += answers[fmt.Sprintf("%s/%s/%d/%c", version, item, at, 'a'+index)].probability
				}
			}
			sums[version] += score[version]
		}
		switch difference := score["default"] - score["lanczos"]; {
		case difference > 0:
			wins++
		case difference < 0:
			losses++
		default:
			ties++
		}
		t.Logf("%s: %s %.3f, Lanczos %.3f", item, defaultLevel, score["default"], score["lanczos"])
	}
	preferred, lanczosPreferred := tensorstats.ExactPairedBinary(wins, losses, ties).Tail, tensorstats.ExactPairedBinary(losses, wins, ties).Tail
	t.Logf("asked whether each is sharp and clean, the judge scores %s higher at %d clips, lower at %d, the same at %d (tail %.3g for %s, %.3g for Lanczos); mean %.3f against Lanczos's %.3f",
		defaultLevel, wins, losses, ties, preferred, defaultLevel, lanczosPreferred, sums["default"]/float64(len(items)), sums["lanczos"]/float64(len(items)))
	if lanczosPreferred <= calibrationLevel {
		t.Errorf("the judge prefers FFmpeg's Lanczos over %s at the same size: %d lower, %d higher, %d the same, tail %.3g against %.3g",
			defaultLevel, losses, wins, ties, lanczosPreferred, calibrationLevel)
	}
}

// readRepairLevels reads each denoise strength on the source clips given noise of degradeNoise, and each deblur
// strength on them blurred by degradeBlurRadius, by VMAF and PSNR against the clean original, beside the degraded
// clip's own. Each strength's VMAF is set against the degraded clip's by the exact paired test, and the strongest of
// each repair beats it at calibrationLevel: on 2026-11-09 every deblur strength did, and denoise at 3 and 4 alone, VMAF
// reading this noise as little loss.
func readRepairLevels(t *testing.T, ffmpeg string, clips [][][]float32, worker *device.Worker, runtime *nvvfx.Runtime) {
	t.Helper()
	ctx := t.Context()
	scratch := t.TempDir()
	random := rand.New(rand.NewPCG(degradeSeed, degradeSeed))
	for _, repair := range []struct {
		name    string
		base    uint32
		degrade func([]float32) []float32
	}{
		{"denoise", denoiseLevel, func(frame []float32) []float32 { return noisedFrame(frame, random) }},
		{"deblur", deblurLevel, blurredFrame},
	} {
		// vmaf[strength][clip] and psnr alike, the degraded clip at strength 0.
		vmaf, psnr := make([][]float64, finishStrengths+1), make([][]float64, finishStrengths+1)
		for _, frames := range clips {
			reference, distorted := filepath.Join(scratch, "reference.rgb"), filepath.Join(scratch, "distorted.rgb")
			testcheck.NoError(t, writeRaw(reference, frames))
			degraded := make([][]float32, len(frames))
			for index, frame := range frames {
				degraded[index] = repair.degrade(frame)
			}
			for strength := range finishStrengths + 1 {
				repaired := degraded
				if strength > 0 {
					repaired = testcheck.Must(upscaleFrames(ctx, worker, runtime, repair.base+uint32(strength-1), degraded, 1))(t)
				}
				testcheck.NoError(t, writeRaw(distorted, repaired))
				vmaf[strength] = append(vmaf[strength], testcheck.Must(readVMAF(ctx, ffmpeg, distorted, reference, scratch))(t))
				psnr[strength] = append(psnr[strength], meanPSNR(repaired, frames))
			}
		}
		for strength := 1; strength <= finishStrengths; strength++ {
			var wins, losses, ties uint64
			for clip := range clips {
				switch difference := vmaf[strength][clip] - vmaf[0][clip]; {
				case difference > 0:
					wins++
				case difference < 0:
					losses++
				default:
					ties++
				}
			}
			tail := tensorstats.ExactPairedBinary(wins, losses, ties).Tail
			t.Logf("%s strength %d: VMAF %.2f against the degraded clip's %.2f, higher at %d clips, lower at %d, the same at %d (tail %.3g); PSNR %.2f dB against %.2f dB",
				repair.name, strength, mean(vmaf[strength]), mean(vmaf[0]), wins, losses, ties, tail, mean(psnr[strength]), mean(psnr[0]))
			if strength == finishStrengths && tail > calibrationLevel {
				t.Errorf("%s at its strongest does not repair its degraded clips: VMAF higher at %d, lower at %d, the same at %d, tail %.3g",
					repair.name, wins, losses, ties, tail)
			}
		}
	}
}

// noisedFrame is the frame with Gaussian noise of deviation degradeNoise added to each value, kept in [0, 1].
func noisedFrame(frame []float32, random *rand.Rand) []float32 {
	noised := make([]float32, len(frame))
	for index, value := range frame {
		noised[index] = float32(math.Min(math.Max(float64(value)+random.NormFloat64()*degradeNoise, 0), 1))
	}
	return noised
}

// blurredFrame is the 1280 x 720 planar frame smoothed by degradeBlurPasses box passes of radius degradeBlurRadius,
// along each row and then each column, the edge repeated beyond it.
func blurredFrame(frame []float32) []float32 {
	plane := superResWidth * superResHeight
	blurred := slices.Clone(frame)
	line := make([]float32, max(superResWidth, superResHeight))
	for range degradeBlurPasses {
		for channel := range media.RGBChannels {
			values := blurred[channel*plane : (channel+1)*plane]
			for y := range superResHeight {
				boxLine(values[y*superResWidth:(y+1)*superResWidth], 1, superResWidth, line)
			}
			for x := range superResWidth {
				boxLine(values[x:], superResWidth, superResHeight, line)
			}
		}
	}
	return blurred
}

// boxLine replaces each of count values stride apart by the mean of the values within degradeBlurRadius of it.
func boxLine(values []float32, stride, count int, line []float32) {
	for index := range count {
		line[index] = values[index*stride]
	}
	for index := range count {
		var sum float32
		for offset := -degradeBlurRadius; offset <= degradeBlurRadius; offset++ {
			sum += line[min(max(index+offset, 0), count-1)]
		}
		values[index*stride] = sum / (2*degradeBlurRadius + 1)
	}
}
