package audioparity

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
)

func TestEncoderCapturePublication(t *testing.T) {
	t.Parallel()
	election := qualificationElection(t)
	encoded, err := os.ReadFile("testdata/encoder_capture.json")
	if err != nil {
		t.Fatal(err)
	}
	var capture encoderCapture
	if err = strictjson.DecodeBytes(encoded, &capture); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	tensorPath := filepath.Join(dir, "tensors.safetensors")
	if err = safetensors.Save(tensorPath, map[string][]float32{"probe": {1}}, map[string][]int{"probe": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(tensorPath)
	if err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		t.Fatal(err)
	}
	capture.TensorsSHA256, capture.TensorCount = id.DigestHex(), 1
	source := []byte("# identity-only synthetic publication fixture\n")
	sourceID, err := artifact.IdentifyBytes(artifact.KindFile, source)
	if err != nil {
		t.Fatal(err)
	}
	capture.CaptureSHA256 = sourceID.DigestHex()
	sourcePath := filepath.Join(dir, "source.py")
	if err = os.WriteFile(sourcePath, source, 0o600); err != nil {
		t.Fatal(err)
	}
	reportPath := filepath.Join(dir, "capture.json")
	writeReport := func() {
		t.Helper()
		data, err := json.Marshal(capture)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(reportPath, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeReport()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch, err := election.Batch("capture-test/election")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	golden, err := PublishEncoderCapture(t.Context(), store, election, reportPath, sourcePath, tensorPath)
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
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
