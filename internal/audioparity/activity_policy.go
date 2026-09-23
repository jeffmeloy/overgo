package audioparity

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/speechactivity"
	"overgo/internal/strictjson"
)

//go:embed recipes/firered_vad_policy.json
var activityPolicyJSON []byte

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
	} `json:"source"`
	Offline   speechactivity.OfflineConfig  `json:"offline"`
	Streaming speechactivity.BoundaryConfig `json:"streaming"`
}

// ActivityPolicy reports the reviewed default boundary policy and the source
// it was read from. A caller that wants another operating point states it
// rather than editing this one.
func ActivityPolicy() (speechactivity.OfflineConfig, speechactivity.BoundaryConfig, error) {
	policy, err := activityPolicy()
	if err != nil {
		return speechactivity.OfflineConfig{}, speechactivity.BoundaryConfig{}, err
	}
	return policy.Offline, policy.Streaming, nil
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

// ActivityPolicySourcesMatch reports whether the sources the policy cites
// still hash to the digests it names, given the directory the model's own
// repository sits in. A policy whose source has moved is reported rather
// than trusted, because the values are that source's opinion and nothing
// else records them.
func ActivityPolicySourcesMatch(repository string) error {
	policy, err := activityPolicy()
	if err != nil {
		return err
	}
	for path, digest := range map[string]string{
		policy.Source.Offline:   policy.Source.OfflineSHA256,
		policy.Source.Streaming: policy.Source.StreamingSHA256,
	} {
		data, readErr := os.ReadFile(filepath.Join(repository, filepath.FromSlash(path)))
		if readErr != nil {
			return readErr
		}
		sum := sha256.Sum256(data)
		if observed := hex.EncodeToString(sum[:]); observed != digest {
			return fmt.Errorf("audio parity: activity policy source %s is %s, not %s", path, observed, digest)
		}
	}
	return nil
}
