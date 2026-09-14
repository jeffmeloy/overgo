package audioparity

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
)

func TestTransducerCapturePublication(t *testing.T) {
	t.Parallel()
	// Synthetic publication evidence only: this test does not claim ASR parity.
	election := qualificationElection(t)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch, err := election.Batch("transducer-capture-test/model")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	tensorPath := filepath.Join(dir, "tensors.safetensors")
	if err := safetensors.Save(tensorPath, map[string][]float32{"probe": {1}}, map[string][]int{"probe": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	tensorBytes, err := os.ReadFile(tensorPath)
	if err != nil {
		t.Fatal(err)
	}
	tensorID, err := artifact.IdentifyBytes(artifact.KindFile, tensorBytes)
	if err != nil {
		t.Fatal(err)
	}
	source := []byte("# synthetic publication contract fixture\n")
	sourceID, err := artifact.IdentifyBytes(artifact.KindFile, source)
	if err != nil {
		t.Fatal(err)
	}
	sourcePath, reportPath := filepath.Join(dir, "source.py"), filepath.Join(dir, "capture.json")
	if err := os.WriteFile(sourcePath, source, 0o600); err != nil {
		t.Fatal(err)
	}
	var capture transducerCapture
	if err := json.Unmarshal([]byte(`{"version":1,"fixture":"synthetic","reference":"text","samples":1,"sample_rate":16000,"first_chunk_frames":1,"following_chunk_frames":1,"dataset":{"path":"clean/test/fixture.parquet"},"runtime":{"python":"fixture","torch":"fixture","transformers":"fixture","librosa":"fixture","tokenizers":"fixture","threads":1,"device":"cpu","attention":"sdpa"},"captures":{"offline":{"chunks":[{"frames":1}],"steps":[0],"wall_ns":1,"text":"text"},"streaming":{"chunks":[{"frames":1}],"steps":[0],"wall_ns":1,"text":"text"}}}`), &capture); err != nil {
		t.Fatal(err)
	}
	capture.Model = election.Model.ID
	capture.Revision, capture.License = election.ModelSource.Commit, election.License.SPDX
	capture.CaptureSHA256, capture.AudioSHA256, capture.Dataset.SHA256 = sourceID.DigestHex(), sourceID.DigestHex(), sourceID.DigestHex()
	capture.TensorsSHA256, capture.TensorCount = tensorID.DigestHex(), 1
	capture.Files = make(map[string]string)
	for _, component := range election.Model.Components {
		capture.Files[component.Name] = component.Artifact.DigestHex()
	}
	capture.Runtime.Sources = map[string]string{"reference.py": sourceID.DigestHex()}
	writeReport := func(c transducerCapture) {
		t.Helper()
		data, err := json.Marshal(c)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(reportPath, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeReport(capture)
	golden, err := PublishTransducerCapture(t.Context(), store, reportPath, sourcePath, tensorPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.RequireTypedContent(t.Context(), store, golden); err != nil {
		t.Fatal(err)
	}
	again, err := PublishTransducerCapture(t.Context(), store, reportPath, sourcePath, tensorPath)
	if err != nil || again != golden {
		t.Fatalf("replay=%s: %v", again, err)
	}
	for _, tc := range []struct {
		name   string
		change func(*transducerCapture)
	}{
		{"invalid revision", func(c *transducerCapture) { c.Revision = strings.Repeat("z", 40) }},
		{"parent traversal", func(c *transducerCapture) { c.Dataset.Path = "../fixture" }},
		{"drive path", func(c *transducerCapture) { c.Files = map[string]string{"C:/fixture": sourceID.DigestHex()} }},
		{"missing runtime", func(c *transducerCapture) { c.Runtime.Tokenizers = "" }},
		{"nonfinite", func(c *transducerCapture) { c.Nonfinite = map[string]int{"joint": 1} }},
		{"component mismatch", func(c *transducerCapture) { c.Files = map[string]string{"wrong": sourceID.DigestHex()} }},
		{"tensor count", func(c *transducerCapture) { c.TensorCount++ }},
		{"tensor identity", func(c *transducerCapture) { c.TensorsSHA256 = sourceID.DigestHex() }},
		{"source identity", func(c *transducerCapture) { c.CaptureSHA256 = tensorID.DigestHex() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			candidate := capture
			tc.change(&candidate)
			writeReport(candidate)
			if _, err := PublishTransducerCapture(t.Context(), store, reportPath, sourcePath, tensorPath); err == nil {
				t.Fatal("invalid capture admitted")
			}
		})
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err := PublishTransducerCapture(ctx, store, "absent", sourcePath, tensorPath); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled publication: %v", err)
	}
	if _, err := PublishTransducerCapture(nil, store, reportPath, sourcePath, tensorPath); err == nil {
		t.Fatal("nil context admitted")
	}
	if _, err := PublishTransducerCapture(t.Context(), nil, reportPath, sourcePath, tensorPath); err == nil {
		t.Fatal("nil repository admitted")
	}
}
