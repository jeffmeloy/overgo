//go:build windows && modeltest

package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/nvvfx"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/processcontrol"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The source clips RTX Video Super Resolution is read on, and how: VidGen clips the harder video suite describes (the
// names VidGen_frames holds), sixteen frames of each at 1280 x 720, each shrunk by two and by four by averaging each
// block (FFmpeg's area filter on whole factors), upscaled back by each level, and read against its original.
const (
	superResClips     = 24
	superResFrames    = 16
	superResFPS       = 4
	superResWidth     = 1280
	superResHeight    = 720
	superResPairedBar = 0.05
	float32Size       = 4
)

// superResFactors are the factors a clip is shrunk by and upscaled back.
var superResFactors = []int{2, 4}

// superResLevels are the upscaling levels read, by the names a request gives them, bicubic first as the bar, each with
// the effect's QualityLevel (nvidia-vfx video_super_res.py).
var superResLevels = []latentvideo.UpscaleLevel{"bicubic", "low", "medium", "high", "ultra", "clean-low", "clean-medium", "clean-high", "clean-ultra"}

// superResQuality is the effect's QualityLevel of each name.
var superResQuality = map[latentvideo.UpscaleLevel]uint32{
	"bicubic": 0, "low": 1, "medium": 2, "high": 3, "ultra": 4, "clean-low": 16, "clean-medium": 17, "clean-high": 18, "clean-ultra": 19,
}

// TestVideoSuperResBeatsBicubicOnSourceClips reads each upscaling level against bicubic, level 0 of the same effect, on
// source clips with a ground truth: each clip shrunk by two and by four and upscaled back, scored by VMAF (FFmpeg's
// libvmaf) and PSNR against its original. A level beats bicubic where, clip by clip, it scores higher more often than
// lower by the exact paired test at 0.05, at both factors, by VMAF. The level a request gets where it names none is the
// one of those with the highest mean VMAF over both factors.
func TestVideoSuperResBeatsBicubicOnSourceClips(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_NVVFX") == "" {
		testskip.NotApplicable(t, "OVERGO_NVVFX names no NVIDIA Video Effects runtime here")
	}
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	clips, names := superResSources(t, roots, ffmpeg)
	scratch := t.TempDir()
	// vmaf[factor][level][clip] and psnr alike.
	vmaf, psnr := map[int]map[latentvideo.UpscaleLevel][]float64{}, map[int]map[latentvideo.UpscaleLevel][]float64{}
	runtime := testcheck.Must(nvvfx.Open())(t)
	worker := testcheck.Must(device.New(0))(t)
	t.Cleanup(func() { testcheck.NoError(t, worker.Close()) })
	// The denoise and deblur strengths are read on the same decoded clips, with the same worker and effects runtime.
	t.Run("repair-levels", func(t *testing.T) { readRepairLevels(t, ffmpeg, clips, worker, runtime) })
	for _, factor := range superResFactors {
		vmaf[factor], psnr[factor] = map[latentvideo.UpscaleLevel][]float64{}, map[latentvideo.UpscaleLevel][]float64{}
		for clip, frames := range clips {
			reference := filepath.Join(scratch, "reference.rgb")
			testcheck.NoError(t, writeRaw(reference, frames))
			small := make([][]float32, len(frames))
			for index, frame := range frames {
				small[index] = boxShrink(frame, factor)
			}
			for _, level := range superResLevels {
				upscaled := testcheck.Must(upscaleFrames(ctx, worker, runtime, superResQuality[level], small, factor))(t)
				distorted := filepath.Join(scratch, "distorted.rgb")
				testcheck.NoError(t, writeRaw(distorted, upscaled))
				score := testcheck.Must(readVMAF(ctx, ffmpeg, distorted, reference, scratch))(t)
				vmaf[factor][level] = append(vmaf[factor][level], score)
				psnr[factor][level] = append(psnr[factor][level], meanPSNR(upscaled, frames))
			}
			t.Logf("factor %d, clip %d of %d (%s) read", factor, clip+1, len(clips), names[clip])
		}
	}
	best, bestMean := latentvideo.UpscaleLevel(""), math.Inf(-1)
	for _, level := range superResLevels {
		beats := true
		var line []string
		var total float64
		for _, factor := range superResFactors {
			var wins, losses, ties uint64
			for clip := range clips {
				switch difference := vmaf[factor][level][clip] - vmaf[factor]["bicubic"][clip]; {
				case difference > 0:
					wins++
				case difference < 0:
					losses++
				default:
					ties++
				}
			}
			test := tensorstats.ExactPairedBinary(wins, losses, ties)
			beats = beats && level != "bicubic" && test.Tail <= superResPairedBar
			meanVMAF, meanDB := mean(vmaf[factor][level]), mean(psnr[factor][level])
			total += meanVMAF
			line = append(line, fmt.Sprintf("x%d VMAF %.2f PSNR %.2f dB, against bicubic %d-%d-%d (tail %.3g)", factor, meanVMAF, meanDB, wins, losses, ties, test.Tail))
		}
		t.Logf("%-12s %s; beats bicubic %t", level, strings.Join(line, "; "), beats)
		if beats && total > bestMean {
			best, bestMean = level, total
		}
	}
	if best == "" {
		t.Fatal("no upscaling level beats bicubic at both factors by the exact paired test at 0.05")
	}
	if got := latentvideo.UpscaleLevel("").Default(); got != string(best) {
		t.Fatalf("a request naming no level gets %s, where the level that beats bicubic with the highest mean VMAF is %s", got, best)
	}
}

// superResSources are the first superResClips VidGen clips VidGen_frames names that decode to superResFrames frames at
// 1280 x 720, each frame planar float32 RGB in [0, 1], and their names.
func superResSources(t *testing.T, roots dataroot.Roots, ffmpeg string) ([][][]float32, []string) {
	t.Helper()
	sources := filepath.Join(roots.Datasets, "VidGen")
	listed := testcheck.Must(os.ReadDir(filepath.Join(roots.Datasets, "VidGen_frames")))(t)
	var clips [][][]float32
	var names []string
	for _, entry := range listed {
		clip := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
		pictures, err := media.DecodeVideoFile(t.Context(), filepath.Join(sources, clip+clipFileExtension), ffmpeg, superResFPS, superResFrames)
		if err != nil || len(pictures) < superResFrames || pictures[0].Bounds().Dx() != superResWidth || pictures[0].Bounds().Dy() != superResHeight {
			continue
		}
		frames := make([][]float32, len(pictures))
		for index, picture := range pictures {
			frames[index] = planarUnit(picture.Bounds().Min.X, picture.Bounds().Min.Y, func(x, y int) (uint32, uint32, uint32) {
				r, g, b, _ := picture.At(x, y).RGBA()
				return r, g, b
			})
		}
		clips, names = append(clips, frames), append(names, clip)
		if len(clips) == superResClips {
			break
		}
	}
	if len(clips) < superResClips {
		t.Fatalf("VidGen holds %d decodable 1280 x 720 clips among VidGen_frames' names, not %d", len(clips), superResClips)
	}
	return clips, names
}

// planarUnit is a 1280 x 720 picture as planar float32 RGB in [0, 1], read by pixel.
func planarUnit(minX, minY int, at func(x, y int) (uint32, uint32, uint32)) []float32 {
	plane := superResWidth * superResHeight
	frame := make([]float32, media.RGBChannels*plane)
	for y := range superResHeight {
		for x := range superResWidth {
			r, g, b := at(minX+x, minY+y)
			for channel, value := range []uint32{r, g, b} {
				frame[channel*plane+y*superResWidth+x] = media.NormalizedRGBAChannel(value)
			}
		}
	}
	return frame
}

// boxShrink is a 1280 x 720 planar frame shrunk by factor, each pixel the mean of its factor by factor block.
func boxShrink(frame []float32, factor int) []float32 {
	width, height := superResWidth/factor, superResHeight/factor
	plane, small := superResWidth*superResHeight, make([]float32, media.RGBChannels*width*height)
	for channel := range media.RGBChannels {
		for y := range height {
			for x := range width {
				var sum float32
				for dy := range factor {
					for dx := range factor {
						sum += frame[channel*plane+(y*factor+dy)*superResWidth+x*factor+dx]
					}
				}
				small[channel*width*height+y*width+x] = sum / float32(factor*factor)
			}
		}
	}
	return small
}

// upscaleFrames runs each small frame through the effect at level back to 1280 x 720, on the device worker.
func upscaleFrames(ctx context.Context, worker *device.Worker, runtime *nvvfx.Runtime, level uint32, small [][]float32, factor int) ([][]float32, error) {
	upscaled := make([][]float32, len(small))
	err := worker.Do(ctx, func(state *device.State) (failed error) {
		source := nvvfx.Image{Width: superResWidth / factor, Height: superResHeight / factor}
		target := nvvfx.Image{Width: superResWidth, Height: superResHeight}
		for _, image := range []*nvvfx.Image{&source, &target} {
			pixels, err := state.Driver.MemAlloc(image.Bytes())
			if err != nil {
				return err
			}
			image.Pixels = pixels
			defer func() { failed = errors.Join(failed, state.Driver.MemFree(pixels)) }()
		}
		upscaler, err := runtime.NewUpscaler(state.Driver, state.Stream, level, source, target)
		if err != nil {
			return err
		}
		defer func() { failed = errors.Join(failed, upscaler.Close()) }()
		host, result := make([]byte, source.Bytes()), make([]byte, target.Bytes())
		for index, frame := range small {
			for at, value := range frame {
				binary.LittleEndian.PutUint32(host[at*float32Size:], math.Float32bits(value))
			}
			if err := errors.Join(state.Driver.MemcpyHtoD(source.Pixels, host), upscaler.Frame(source.Pixels, target.Pixels),
				state.Driver.StreamSynchronize(state.Stream), state.Driver.MemcpyDtoH(result, target.Pixels)); err != nil {
				return err
			}
			out := make([]float32, len(result)/float32Size)
			for at := range out {
				out[at] = math.Float32frombits(binary.LittleEndian.Uint32(result[at*float32Size:]))
			}
			upscaled[index] = out
		}
		return nil
	})
	return upscaled, err
}

// writeRaw writes planar frames as interleaved 8-bit RGB, the raw video FFmpeg reads.
func writeRaw(path string, frames [][]float32) error {
	plane := superResWidth * superResHeight
	data := make([]byte, 0, len(frames)*media.RGBChannels*plane)
	for _, frame := range frames {
		for at := range plane {
			for channel := range media.RGBChannels {
				data = append(data, byte(math.Round(float64(min(max(frame[channel*plane+at], 0), 1))*math.MaxUint8)))
			}
		}
	}
	return os.WriteFile(path, data, frameFileMode)
}

// readVMAF is the pooled mean VMAF of the distorted raw clip against the reference, by FFmpeg's libvmaf.
func readVMAF(ctx context.Context, ffmpeg, distorted, reference, scratch string) (float64, error) {
	// The log is named relative to the scratch directory FFmpeg runs in: a drive letter's colon ends a filter option.
	const logName = "vmaf.json"
	logPath := filepath.Join(scratch, logName)
	size := fmt.Sprintf("%dx%d", superResWidth, superResHeight)
	raw := []string{"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", fmt.Sprint(superResFPS)}
	args := append(append(append(append([]string{"-nostdin", "-hide_banner", "-loglevel", "error"}, raw...), "-i", distorted), raw...),
		"-i", reference, "-lavfi", "[0:v][1:v]libvmaf=log_fmt=json:log_path="+logName+":n_threads=8", "-f", "null", "-")
	var diagnostic strings.Builder
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: ffmpeg, Args: args, Stderr: &diagnostic, Dir: scratch})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("libvmaf exited %d: %s", receipt.ExitCode, diagnostic.String())
	}
	if err != nil {
		return 0, err
	}
	var log struct {
		Pooled struct {
			VMAF struct {
				Mean float64 `json:"mean"`
			} `json:"vmaf"`
		} `json:"pooled_metrics"`
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		return 0, err
	}
	if err := json.Unmarshal(data, &log); err != nil {
		return 0, err
	}
	return log.Pooled.VMAF.Mean, nil
}

// meanPSNR is the mean over frames of each frame's peak signal-to-noise ratio, in dB, in [0, 1].
func meanPSNR(upscaled, original [][]float32) float64 {
	var total float64
	for index, frame := range upscaled {
		var squared float64
		for at, value := range frame {
			difference := float64(min(max(value, 0), 1) - original[index][at])
			squared += difference * difference
		}
		total += 10 * math.Log10(float64(len(frame))/squared)
	}
	return total / float64(len(upscaled))
}

// mean is the mean of values.
func mean(values []float64) float64 {
	var total float64
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}
