//go:build windows

package mediacapability

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/vqaserve"
)

// TestVQAActivation proves the page form end to end on the real RxBrain
// checkpoint: the directory resolves into the vqa recipe, the canonical
// demo image enters the store as an image artifact, and a request naming
// it and the canonical question runs through the catalog's executor to
// the answer the parity harness pins, published as a plain-text output.
func TestVQAActivation(t *testing.T) {
	cudatest.Require(t)
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(roots.Models, "Hy-Embodied-RxBrain-1.0")
	image, err := os.ReadFile(filepath.Join(model, "Hy-Embodied-RxBrain-1.0", "demo_cases", "bridgev2_move_toy", "input", "obs_1.jpg"))
	if err != nil {
		t.Skipf("UNAVAILABLE: RxBrain demo image: %v", err)
	}
	capability := vqaCapability()
	source, err := capability.Resolve(model)
	if err != nil {
		t.Fatal(err)
	}
	modelID := source.Inventory.Manifest.ID
	definition, _, err := source.Define(modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishArtifactExtent(t, store, modelID, model)
	if err := modelrecipe.EnsureRuntimePolicy(t.Context(), store, definition); err != nil {
		t.Fatal(err)
	}
	imageContent, err := artifact.DocumentContract{Kind: artifact.KindFile, MediaType: "image/jpeg", Schema: "overgo.image-input.v1"}.OwnedContentBytes(image)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/vqa-image", Contents: []artifact.Content{imageContent}}); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(vqaserve.Request{Image: imageContent.Descriptor.ID, Question: "What objects are on the stovetop, and where is the green toy?"})
	if err != nil {
		t.Fatal(err)
	}
	execution := candidateCapabilityExecution(t, store, program)
	output, err := capability.Execute(t.Context(), store, model, execution, string(raw))
	if err != nil {
		t.Fatal(err)
	}
	answer, ok := capabilityruntime.Unwrap(output).(string)
	if !ok {
		t.Fatalf("answer output type=%T", output)
	}
	const wantPrefix = "The stovetop holds a metal pot on the left burner"
	if !strings.HasPrefix(answer, wantPrefix) {
		t.Fatalf("answer %q does not start with %q", answer, wantPrefix)
	}
	content, err := OutputContent(answer)
	if err != nil || content.Descriptor.MediaType != textMediaType {
		t.Fatalf("published answer = %+v, %v", content.Descriptor, err)
	}
	t.Logf("VQA activation answer=%q phases=%d", answer, len(output.(capabilityruntime.Measured).Phases))
}
