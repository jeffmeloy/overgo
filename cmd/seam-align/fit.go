package main

import (
	"errors"
	"fmt"
	"strings"

	"overgo/internal/composition"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/rae"
	"overgo/internal/safetensors"
)

// fitSpec names a cheap baseline's fit: the capture's samples file, the source representation the fit maps, the source
// videos held back as the trained bridge's run held them, and the bridge file it writes.
type fitSpec struct {
	Capture string `json:"capture"`
	Source  string `json:"source"`
	Held    int    `json:"held"`
	Output  string `json:"output"`
}

// fitBridge writes a capture's closed-form seam adapter as a bridge: the least-squares map of the source
// representation of the samples a run trains on to their captions' mean embeddings, centered on those captions' mean,
// which the bridge's offsets carry. It is the cheap baseline a trained bridge is read against, fitted with no gradient
// through the text model; applied to each of a frame's tokens, it places rows whose mean is its estimate of the
// caption's.
func fitBridge(specPath string) (err error) {
	var spec fitSpec
	if err := jsonfile.DecodeStrict(specPath, &spec); err != nil {
		return err
	}
	var samples evaluation.CaptionTrainingCapture
	if err := jsonfile.DecodeStrict(spec.Capture, &samples); err != nil {
		return err
	}
	capture, err := safetensors.OpenFile(strings.TrimSuffix(spec.Capture, evaluation.CaptureSamplesSuffix))
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, capture.Close()) }()
	source, err := captureRows(capture, sourcePrefix+spec.Source)
	if err != nil {
		return err
	}
	target, err := captureRows(capture, targetTensor)
	if err != nil {
		return err
	}
	if len(source) != len(samples.Samples) || len(target) != len(samples.Samples) {
		return fmt.Errorf("seam-align: the capture holds %d sources and %d targets for %d samples", len(source), len(target), len(samples.Samples))
	}
	trainedSource, _ := evaluation.HeldBack(samples.Samples, source, spec.Held)
	trainedTarget, _ := evaluation.HeldBack(samples.Samples, target, spec.Held)
	if len(trainedTarget) == 0 {
		return errors.New("seam-align: every sample of the capture is held back")
	}
	inWidth, outWidth := len(source[0]), len(target[0])
	mean := make([]float64, outWidth)
	for _, row := range trainedTarget {
		for column, value := range row {
			mean[column] += value / float64(len(trainedTarget))
		}
	}
	centered := make([][]float64, len(trainedTarget))
	for index, row := range trainedTarget {
		centered[index] = make([]float64, outWidth)
		for column, value := range row {
			centered[index][column] = value - mean[column]
		}
	}
	adapter, _, err := composition.AlignSeamAdapter(trainedSource, centered)
	if err != nil {
		return err
	}
	bridge := rae.EncoderBridge{Weight: make([]float32, outWidth*inWidth), Bias: make([]float32, outWidth)}
	for out := range outWidth {
		bridge.Bias[out] = float32(mean[out])
		for in := range inWidth {
			bridge.Weight[out*inWidth+in] = float32(adapter[in][out])
		}
	}
	return rae.SaveBridgeWeights(spec.Output, bridge)
}
