package audioparity

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
)

func TestEncoderCapturePublication(t *testing.T) {
	t.Parallel()
	election := qualificationElection(t)
	encoded, err := os.ReadFile("testdata/encoder_capture.json")
	testcheck.NoError(t, err)
	var capture encoderCapture
	if err = strictjson.DecodeBytes(encoded, &capture); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	tensorPath := filepath.Join(dir, "tensors.safetensors")
	if err = safetensors.Save(tensorPath, map[string][]float32{"probe": {1}}, map[string][]int{"probe": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	data := testcheck.Must(os.ReadFile(tensorPath))(t)
	id := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, data))(t)
	capture.TensorsSHA256, capture.TensorCount = id.DigestHex(), 1
	source := []byte("# identity-only synthetic publication fixture\n")
	sourceID := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, source))(t)
	capture.CaptureSHA256 = sourceID.DigestHex()
	sourcePath := filepath.Join(dir, "source.py")
	if err = os.WriteFile(sourcePath, source, 0o600); err != nil {
		t.Fatal(err)
	}
	reportPath := filepath.Join(dir, "capture.json")
	writeReport := func() {
		t.Helper()
		data, err := json.Marshal(capture)
		testcheck.NoError(t, err)
		if err = os.WriteFile(reportPath, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeReport()
	store := overgodbtest.Open(t, t.TempDir())
	batch := testcheck.Must(election.Batch("capture-test/election"))(t)
	if _, err = artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	golden := testcheck.Must(PublishEncoderCapture(t.Context(), store, election, reportPath, sourcePath, tensorPath))(t)
	if _, err = artifact.RequireTypedContent(t.Context(), store, golden); err != nil {
		t.Fatal(err)
	}
	if _, err = artifact.AvailablePath(t.Context(), store, id, artifact.LocationFile); err != nil {
		t.Fatal(err)
	}
	again, err := PublishEncoderCapture(t.Context(), store, election, reportPath, sourcePath, tensorPath)
	if err != nil || again != golden {
		t.Fatalf("replay=%s %v", again, err)
	}
	installed := filepath.Join(t.TempDir(), "tensors.safetensors")
	if err = os.WriteFile(installed, data, 0o600); err != nil {
		t.Fatal(err)
	}
	again, err = PublishEncoderCapture(t.Context(), store, election, reportPath, sourcePath, installed)
	if err != nil || again != golden {
		t.Fatalf("same capture at another location=%s %v", again, err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err = PublishEncoderCapture(ctx, store, election, reportPath, sourcePath, tensorPath); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
	if _, err = PublishEncoderCapture(nil, store, election, reportPath, sourcePath, tensorPath); err == nil {
		t.Fatal("accepted nil context")
	}
	if _, err = PublishEncoderCapture(t.Context(), nil, election, reportPath, sourcePath, tensorPath); err == nil {
		t.Fatal("accepted nil repository")
	}
	capture.TensorCount++
	writeReport()
	if _, err = PublishEncoderCapture(t.Context(), store, election, reportPath, sourcePath, tensorPath); err == nil {
		t.Fatal("accepted changed tensor count")
	}
	capture.TensorCount--
	capture.TensorsSHA256 = sourceID.DigestHex()
	writeReport()
	if _, err = PublishEncoderCapture(t.Context(), store, election, reportPath, sourcePath, tensorPath); err == nil {
		t.Fatal("accepted changed tensor identity")
	}
	capture.TensorsSHA256 = id.DigestHex()
	writeReport()
	if err = os.WriteFile(sourcePath, []byte("changed"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = PublishEncoderCapture(t.Context(), store, election, reportPath, sourcePath, tensorPath); err == nil {
		t.Fatal("accepted changed capture source")
	}
}

func TestEncoderCaptureRefusesChangedBindings(t *testing.T) {
	t.Parallel()
	election := qualificationElection(t)
	encoded, err := os.ReadFile("testdata/encoder_capture.json")
	testcheck.NoError(t, err)
	for _, tc := range []struct {
		name   string
		mutate func(*encoderCapture)
	}{
		{"source", func(c *encoderCapture) { c.Source.Commit = "changed" }},
		{"corpus", func(c *encoderCapture) { c.Dataset.Bytes++ }},
		{"missing cases", func(c *encoderCapture) { c.Cases = nil }},
		{"runtime", func(c *encoderCapture) { c.Runtime.Device = "cuda" }},
		{"attention", func(c *encoderCapture) { c.Runtime.Attention = "sdpa" }},
		{"transcript", func(c *encoderCapture) { c.Cases[0].Text = "changed" }},
		{"empty frames", func(c *encoderCapture) { c.Cases[0].Frames[1] = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var capture encoderCapture
			if err = strictjson.DecodeBytes(encoded, &capture); err != nil {
				t.Fatal(err)
			}
			tc.mutate(&capture)
			if err = capture.validate(election); err == nil {
				t.Fatal("accepted changed binding")
			}
		})
	}
}
