//overgo:runtime-inputs caller

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/checked"
	"overgo/internal/dataset"
	"overgo/internal/media"
)

type frontendObservation struct {
	Admission             artifact.ID `json:"admission"`
	Config                artifact.ID `json:"config"`
	FeaturesSHA256        string      `json:"features_sha256"`
	Frames                int         `json:"frames"`
	Values                int         `json:"values"`
	ReconstructionSHA256  string      `json:"reconstruction_sha256,omitzero"`
	ReconstructionSamples int         `json:"reconstruction_samples,omitzero"`
	SourceMaximumError    *float64    `json:"source_maximum_error,omitzero"`
	FrameLimit            int         `json:"frame_limit,omitzero"`
	TraceFrames           int         `json:"trace_frames,omitzero"`
	TraceSHA256           string      `json:"trace_sha256,omitzero"`
}

func inspectFeatures(ctx context.Context, repository artifact.Repository, encoder *json.Encoder, frontend *audiodsp.Frontend, config artifact.Content,
	inspection dataset.AudioInspection, chunkSamples int, reconstruct bool, frameLimit, traceFrames int, workspace *audiodsp.Workspace) error {
	if inspection.Signal.Format.Channels != 1 {
		return errors.New("audio-inspect: frontend requires mono input; no implicit downmix")
	}
	sampleRate, ok := checked.Int(inspection.Signal.Format.SampleRate)
	if !ok {
		return errors.New("audio-inspect: sample rate exceeds native integer")
	}
	var chunks [][]float32
	for start := 0; start < len(inspection.Samples); {
		end := start + min(chunkSamples, len(inspection.Samples)-start)
		chunks = append(chunks, inspection.Samples[start:end])
		start = end
	}
	options := audiodsp.ProcessOptions{FrameLimit: frameLimit}
	digest := sha256.New()
	if traceFrames != 0 {
		traceEncoder := json.NewEncoder(digest)
		options.Observe = func(trace audiodsp.FrameTrace) error {
			if trace.Frame < traceFrames {
				return traceEncoder.Encode(trace)
			}
			return nil
		}
	}
	values, frames, err := frontend.Process(ctx, chunks, sampleRate, workspace, options)
	if err != nil {
		return err
	}
	if traceFrames > frames {
		return errors.New("audio-inspect: requested trace exceeds selected frames")
	}
	observation := frontendObservation{Admission: inspection.DecisionID, Config: config.Descriptor.ID, FeaturesSHA256: media.SamplesSHA256(values), Frames: frames, Values: len(values), FrameLimit: frameLimit}
	if traceFrames != 0 {
		observation.TraceFrames, observation.TraceSHA256 = traceFrames, hex.EncodeToString(digest.Sum(nil))
	}
	if reconstruct {
		waveform, err := frontend.Reconstruct(ctx, chunks, sampleRate, workspace)
		if err != nil {
			return err
		}
		observation.ReconstructionSHA256 = media.SamplesSHA256(waveform)
		observation.ReconstructionSamples = len(waveform)
		// Compare the original grid only when no sample-rate conversion occurred.
		var declaration audiodsp.FrontendConfig
		if err := json.Unmarshal(config.Data, &declaration); err != nil {
			return err
		}
		if declaration.SampleRate == sampleRate && len(waveform) == len(inspection.Samples) {
			var maximum float64
			for index, value := range waveform {
				maximum = max(maximum, math.Abs(float64(value)-float64(inspection.Samples[index])))
			}
			observation.SourceMaximumError = &maximum
		}
	}
	content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-frontend-observation/v1"), observation)
	if err != nil {
		return err
	}
	lineage := artifact.DependencyLineage(content.Descriptor.ID, inspection.DecisionID, config.Descriptor.ID)
	batch, err := artifact.NewDocumentBatch("audio/frontend/"+content.Descriptor.ID.DigestHex(), []artifact.Content{config, content}, lineage, nil)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return err
	}
	return encoder.Encode(observation)
}
