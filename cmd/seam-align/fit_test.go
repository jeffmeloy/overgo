package main

import (
	"math"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/rae"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheCheapBaselineIsTheCapturesClosedFormFit writes a capture whose trained samples' captions are an exact affine
// map of their patch means and whose held-back source's are not: the cheap baseline is that map, the bridge giving
// each trained sample's caption from its patch mean, the held-back samples left out of the fit.
func TestTheCheapBaselineIsTheCapturesClosedFormFit(t *testing.T) {
	t.Parallel()
	const tolerance = 1e-4
	weights, offsets := [][]float64{{1, -2, 0.5}, {0.25, 3, -1}}, []float64{7, -4}
	// The trained sources come in opposite pairs, so their mean is zero and the captions' mean is the map's offset.
	trainedSources := [][]float64{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {1, 1, 1}, {-1, -1, -1}}
	capture := evaluation.CaptionTrainingCapture{EncoderWidth: len(weights[0]), TextWidth: len(weights)}
	var sources, targets []float32
	add := func(group string, source, target []float64) {
		index := strconv.Itoa(len(capture.Samples))
		capture.Samples = append(capture.Samples, evaluation.CapturedCaption{
			CaptionTrainingClip: evaluation.CaptionTrainingClip{Vid: index, Source: group}, Frame: testutil.ArtifactID(t, artifact.KindFile, index),
		})
		for _, value := range source {
			sources = append(sources, float32(value))
		}
		for _, value := range target {
			targets = append(targets, float32(value))
		}
	}
	add("held", []float64{5, 5, 5}, []float64{100, -100})
	for index, source := range trainedSources {
		target := make([]float64, len(weights))
		for out := range weights {
			target[out] = offsets[out]
			for in, value := range source {
				target[out] += weights[out][in] * value
			}
		}
		add("source-"+strconv.Itoa(index), source, target)
	}
	work := t.TempDir()
	representations := filepath.Join(work, "capture.safetensors")
	count := len(capture.Samples)
	testcheck.NoError(t, safetensors.Save(representations,
		map[string][]float32{capturePatchMean: sources, targetTensor: targets},
		map[string][]int{capturePatchMean: {count, len(weights[0])}, targetTensor: {count, len(weights)}}, nil))
	samples := representations + evaluation.CaptureSamplesSuffix
	testcheck.NoError(t, jsonfile.Write(samples, capture, 0o600))
	spec, output := filepath.Join(work, "fit.json"), filepath.Join(work, "baseline.safetensors")
	testcheck.NoError(t, jsonfile.Write(spec, fitSpec{Capture: samples, Source: "dino-patch-mean", Held: 1, Output: output}, 0o600))
	testcheck.NoError(t, fitBridge(spec))
	bridge := testcheck.Must(rae.LoadEncoderBridge(output))(t)
	for index, source := range trainedSources {
		row := make([]float32, len(source))
		for in, value := range source {
			row[in] = float32(value)
		}
		mapped := testcheck.Must(bridge.Apply(row))(t)
		for out, value := range mapped {
			if want := targets[(index+1)*len(weights)+out]; math.Abs(float64(value-want)) > tolerance {
				t.Fatalf("the baseline maps trained sample %d's patch mean to %v; want its caption %v", index, mapped, targets[(index+1)*len(weights):(index+2)*len(weights)])
			}
		}
	}
}
