package audioparity

import (
	_ "embed"
	"fmt"

	"overgo/internal/audiodsp"
	"overgo/internal/speechactivity"
	"overgo/internal/strictjson"
)

var (
	//go:embed recipes/firered_vad_policy.json
	activityPolicyJSON []byte
	//go:embed recipes/firered_vad_execution.json
	activityExecutionJSON []byte
	//go:embed recipes/vad_frontend.json
	activityFrontendJSON []byte
)

// reviewedActivityPolicy is the boundary policy a detector applies to the
// probability its network produces per frame, and where that policy comes
// from. The numbers are the operating point the model's own command-line
// tools ship, not a property of its weights, so the document names the
// sources and their digests and this type keeps them together with the
// values they justify.
type reviewedActivityPolicy struct {
	Doc    string `json:"doc"`
	Source struct {
		Offline         string `json:"offline"`
		OfflineSHA256   string `json:"offline_sha256"`
		Streaming       string `json:"streaming"`
		StreamingSHA256 string `json:"streaming_sha256"`
		OfflineWeights  string `json:"offline_weights_sha256"`
	} `json:"source"`
	Offline   speechactivity.OfflineConfig  `json:"offline"`
	Streaming speechactivity.BoundaryConfig `json:"streaming"`
}

// ReviewedActivityDetector is what a whole-recording detector needs beside
// its weights, none of which the checkpoint carries: how its classifier's
// tensors are bound, how a waveform becomes the features it reads, and the
// boundary policy its own tools apply by default.
type ReviewedActivityDetector struct {
	Network  speechactivity.Declaration
	Frontend audiodsp.FrontendConfig
	Policy   speechactivity.OfflineConfig
}

// ActivityDetectorFor resolves the reviewed declarations for a checkpoint by
// the digest of its weights, since the checkpoint names no type of its own;
// known is false for weights no reviewed declaration answers for.
func ActivityDetectorFor(weightsSHA256 string) (ReviewedActivityDetector, bool, error) {
	policy, err := activityPolicy()
	if err != nil || weightsSHA256 != policy.Source.OfflineWeights {
		return ReviewedActivityDetector{}, false, err
	}
	reviewed := ReviewedActivityDetector{Policy: policy.Offline}
	if err := strictjson.DecodeBytes(activityExecutionJSON, &reviewed.Network); err != nil {
		return ReviewedActivityDetector{}, false, fmt.Errorf("audio parity: activity declaration: %w", err)
	}
	if err := strictjson.DecodeBytes(activityFrontendJSON, &reviewed.Frontend); err != nil {
		return ReviewedActivityDetector{}, false, fmt.Errorf("audio parity: activity frontend: %w", err)
	}
	return reviewed, true, nil
}

// activityPolicy decodes the reviewed document and refuses one whose values
// do not describe a usable operating point.
func activityPolicy() (reviewedActivityPolicy, error) {
	var policy reviewedActivityPolicy
	if err := strictjson.DecodeBytes(activityPolicyJSON, &policy); err != nil {
		return reviewedActivityPolicy{}, fmt.Errorf("audio parity: activity policy: %w", err)
	}
	if policy.Offline.Threshold <= 0 || policy.Offline.Threshold >= 1 ||
		policy.Streaming.Threshold <= 0 || policy.Streaming.Threshold >= 1 ||
		policy.Offline.Smoothing <= 0 || policy.Streaming.Smoothing <= 0 ||
		policy.Offline.MaxSpeech <= policy.Offline.MinSpeech ||
		policy.Streaming.MaxSpeech <= policy.Streaming.MinSpeech {
		return reviewedActivityPolicy{}, fmt.Errorf("audio parity: activity policy is not a usable operating point")
	}
	return policy, nil
}
