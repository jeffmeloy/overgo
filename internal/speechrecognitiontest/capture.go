package speechrecognitiontest

import (
	"context"
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// remadeCaptures are the speech captures whose bytes are gone, by their files' ids, and the row that remakes them into
// the data root: the audio reference store recorded the CPU ASR reference's tensors (asr-reference) and the
// transducer's two (transducer-reference-sdpa and its row1) only in a removed checkout's untracked tmp, so the eleven
// tests that read them failed every commit (speech-hold, the owner's review of 2026-11-20). speech-captures empties it.
var remadeCaptures = map[string]string{
	"file:sha256:fd71c122163677bd53aca47f2c90baf859a6b430f02fd8f0616701731158e363": "speech-captures",
	"file:sha256:f6fcb10383f93fa7146f8723e2109510880ae948f3a7e65f9ab27ca02355a1ee": "speech-captures",
	"file:sha256:4802e430d644b0a3ee458ef9b3fba5bf2aff36af31dcacde05f9fea5eefd2ec6": "speech-captures",
}

// capturePath is where the capture file source records as id lies, or, where none of its recorded locations exists
// and remade names it, the row remaking it. A capture with no location left that remade does not name, and a source
// that cannot say where a capture lies, are errors.
func capturePath(ctx context.Context, source artifact.Reader, id artifact.ID, remade map[string]string) (path, row string, err error) {
	if _, err := source.Locations(ctx, id); err != nil {
		return "", "", err
	}
	path, err = artifact.AvailablePath(ctx, source, id, artifact.LocationFile)
	if row, declared := remade[id.String()]; err != nil && declared {
		return "", row, nil
	}
	return path, "", err
}

// CaptureFile is where the speech capture file source records as id lies. A capture whose bytes are gone while a row
// remakes them (remadeCaptures) declares the test inapplicable, naming the row, so commits are not held on speech
// meanwhile (speech-hold); any other capture with no location left fails the test.
func CaptureFile(t testing.TB, source artifact.Reader, id artifact.ID) string {
	t.Helper()
	path, row, err := capturePath(t.Context(), source, id, remadeCaptures)
	if row != "" {
		testskip.NotApplicable(t, fmt.Sprintf("the capture %s has no bytes left while %s remakes it into the data root", id, row))
	}
	testcheck.NoError(t, err)
	return path
}
