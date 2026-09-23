package speechactivity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

// segmentsContract names what a detection request produces: the active spans
// of the recording it carried, as sample intervals with their mean
// probability. The stored-recipe detector publishes the same spans bound to
// source artifacts; a request carries its recording inline, so it has none.
var segmentsContract = artifact.JSONContract(artifact.KindOutput, "overgo/activity-segments/v1")

// DetectionRequest is one whole recording and the bounds it runs within.
// Policy replaces the reviewed boundary policy field by field: a field the
// request names overrides the default, and every other field stays as the
// model's own tools ship it.
type DetectionRequest struct {
	Audio          []byte          `json:"audio"`
	MaximumSamples uint64          `json:"maximum_samples"`
	MemoryBytes    uint64          `json:"memory_bytes"`
	Policy         json.RawMessage `json:"policy,omitzero"`
}

// ValidateDetectionRequest refuses a request that carries no recording or no
// bound, before any model is loaded.
func ValidateDetectionRequest(request DetectionRequest) error {
	if len(request.Audio) == 0 || request.MaximumSamples == 0 || request.MemoryBytes == 0 {
		return errors.New("speech activity: detection requires encoded audio, a sample bound and a host memory ceiling")
	}
	return nil
}

// OfflineDetector runs whole recordings through a classifier loaded from a
// checkpoint and its reviewed declarations, under a default boundary policy
// that each request may override. It holds no store: the recording arrives
// with the request and the spans leave with the response.
type OfflineDetector struct {
	network     *Network
	frontend    *audiodsp.Frontend
	config      audiodsp.FrontendConfig
	policy      OfflineConfig
	memoryBytes uint64
}

// NewOfflineDetector binds a checkpoint to its declaration, frontend and
// default policy, and refuses a default that is not a usable operating point
// before any recording arrives.
func NewOfflineDetector(ctx context.Context, checkpoint string, declaration Declaration, frontend audiodsp.FrontendConfig, policy OfflineConfig, memoryBytes uint64) (*OfflineDetector, error) {
	network, err := loadClassifier(ctx, checkpoint, declaration, frontend, memoryBytes)
	if err != nil {
		return nil, err
	}
	processor, err := audiodsp.NewFrontend(frontend, memoryBytes)
	if err != nil {
		return nil, err
	}
	if _, err := OfflineDecisions(ctx, nil, policy, memoryBytes); err != nil {
		return nil, err
	}
	return &OfflineDetector{network: network, frontend: processor, config: frontend, policy: policy, memoryBytes: memoryBytes}, nil
}

// Detect decodes the request's recording and returns its active spans under
// the default policy with the request's overrides applied.
func (d *OfflineDetector) Detect(ctx context.Context, request DetectionRequest) ([]recipecontract.ActivitySegment, error) {
	if err := ValidateDetectionRequest(request); err != nil {
		return nil, err
	}
	policy := d.policy
	if len(request.Policy) != 0 {
		if err := strictjson.DecodeBytes(request.Policy, &policy); err != nil {
			return nil, fmt.Errorf("speech activity: policy override: %w", err)
		}
	}
	audio, _, err := media.DecodeAudio(ctx, request.Audio, request.MaximumSamples)
	if err != nil {
		return nil, err
	}
	if !frontendReads(audio, d.config, d.memoryBytes) {
		return nil, fmt.Errorf("speech activity: the recording is %d channels of %s at %d Hz; the frontend reads one channel of float samples at %d Hz within its memory ceiling",
			audio.Format.Channels, audio.Format.Encoding, audio.Format.SampleRate, d.config.SampleRate)
	}
	var w DetectionWorkspace
	return offlineSegments(ctx, d.network, d.frontend, d.config, audio.Samples, policy, d.memoryBytes, &w)
}

// RegisterRuntime binds a loaded detector to the activity stage of one
// model's program, so a capability execution reaches it by the module the
// audio contract declares and by nothing else.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, detector *OfflineDetector) error {
	if detector == nil {
		return errors.New("speech activity: incomplete runtime binding")
	}
	return workflowruntime.RegisterContextStage(runtime, modelrecipe.ModuleDetectActivity, modelID,
		func(ctx context.Context, request DetectionRequest) ([]recipecontract.ActivitySegment, error) {
			if err := context.Cause(ctx); err != nil {
				return nil, err
			}
			return detector.Detect(ctx, request)
		}, func(segments []recipecontract.ActivitySegment) (artifact.Content, error) {
			return artifact.JSONContent(segmentsContract, segments)
		})
}

// loadClassifier binds a checkpoint to its declaration and refuses one whose
// input is not the feature the frontend produces.
func loadClassifier(ctx context.Context, checkpoint string, declaration Declaration, frontend audiodsp.FrontendConfig, memoryBytes uint64) (*Network, error) {
	network, err := LoadNetwork(ctx, checkpoint, declaration, memoryBytes)
	if err != nil {
		return nil, err
	}
	if uint64(network.InputWidth()) != uint64(frontend.Geometry.FeatureBins) {
		return nil, errors.New("speech activity: frontend and classifier widths differ")
	}
	return network, nil
}

// offlineSegments runs a whole recording through the frontend, the
// classifier and the offline policy, and reads the chosen frames back as
// sample spans whose confidence is the mean raw probability across each span.
func offlineSegments(ctx context.Context, network *Network, frontend *audiodsp.Frontend, config audiodsp.FrontendConfig, samples []float32, policy OfflineConfig, memoryBytes uint64, w *DetectionWorkspace) ([]recipecontract.ActivitySegment, error) {
	features, frames, err := frontend.Process(ctx, [][]float32{samples}, config.SampleRate, &w.frontend, audiodsp.ProcessOptions{})
	if err != nil {
		return nil, err
	}
	probabilities, _, err := network.Evaluate(ctx, features, frames, nil, &w.network, nil)
	if err != nil {
		return nil, err
	}
	decisions, err := OfflineDecisions(ctx, probabilities, policy, memoryBytes)
	if err != nil {
		return nil, err
	}
	var segments []recipecontract.ActivitySegment
	hop := config.Geometry.HopSamples
	for start := 0; start < len(decisions); {
		if !decisions[start] {
			start++
			continue
		}
		end := start
		var sum float64
		for end < len(decisions) && decisions[end] {
			sum += float64(probabilities[end])
			end++
		}
		endSample := uint64(end) * hop
		if end == len(decisions) {
			// Native offline finalization includes the analysis window and clips
			// to the actual signal. No synthetic samples or frames are generated.
			endSample = min(uint64(len(samples)), endSample+config.Geometry.WindowSamples)
		}
		segments = append(segments, recipecontract.ActivitySegment{Span: recipecontract.SampleSpan{Start: uint64(start) * hop, End: endSample}, Confidence: sum / float64(end-start)})
		start = end
	}
	return segments, nil
}
