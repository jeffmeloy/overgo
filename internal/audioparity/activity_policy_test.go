package audioparity

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testutil"
)

// TestActivityPolicyIsTheModelsOwnOperatingPoint holds the reviewed default
// boundary policy to the values the model's own command-line tools ship, and
// holds the document to naming where it read them. The numbers are that
// tool's opinion rather than a property of the weights, so the check is that
// the cited source still says what the policy claims it says.
func TestActivityPolicyIsTheModelsOwnOperatingPoint(t *testing.T) {
	t.Parallel()
	policy, err := activityPolicy()
	if err != nil {
		t.Fatal(err)
	}
	offline, streaming := policy.Offline, policy.Streaming
	// Read from the upstream tools at the digests the policy cites.
	if offline.Smoothing != 5 || offline.Threshold != 0.4 || offline.MinSpeech != 20 ||
		offline.MaxSpeech != 2000 || offline.MinSilence != 20 ||
		offline.MergeSilence != 0 || offline.ExtendSpeech != 0 {
		t.Fatalf("offline policy is not the shipped operating point: %+v", offline)
	}
	if streaming.Smoothing != 5 || streaming.Threshold != 0.4 || streaming.PadStart != 5 ||
		streaming.MinSpeech != 8 || streaming.MaxSpeech != 2000 || streaming.MinSilence != 20 {
		t.Fatalf("streaming policy is not the shipped operating point: %+v", streaming)
	}
	// The values above are what this test asserts, and they are embedded, so
	// it never skips. The cited sources are the model's own repository, which
	// is not part of this one: where it is present the digests must still
	// match, and where it is absent there is nothing to compare.
	repository := filepath.Join(testutil.RepoRoot(t), "models", "FireRedVAD", "FireRedVAD")
	switch err := policySourcesMatch(policy, repository); {
	case errors.Is(err, fs.ErrNotExist):
		t.Log("policy sources are not present here; the values above are unchecked against them")
	case err != nil:
		t.Fatal(err)
	}
}

// policySourcesMatch reports whether the sources the policy cites still hash
// to the digests it names, given the directory the model's own repository
// sits in. A policy whose source has moved is reported rather than trusted,
// because the values are that source's opinion and nothing else records them.
func policySourcesMatch(policy reviewedActivityPolicy, repository string) error {
	for path, digest := range map[string]string{
		policy.Source.Offline:   policy.Source.OfflineSHA256,
		policy.Source.Streaming: policy.Source.StreamingSHA256,
	} {
		data, err := os.ReadFile(filepath.Join(repository, filepath.FromSlash(path)))
		if err != nil {
			return err
		}
		sum := sha256.Sum256(data)
		if observed := hex.EncodeToString(sum[:]); observed != digest {
			return fmt.Errorf("activity policy source %s is %s, not %s", path, observed, digest)
		}
	}
	return nil
}
