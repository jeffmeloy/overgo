package mediacapability

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/testutil"
)

// TestClipFormRunRetainsItsRequest runs a clip-form LiveEdit request through
// the form resolver. The session director is handed the resolved request as
// the value it is, never as text, and the run keeps the clip form as its
// request: a verification of it then holds a request recipe measure can
// replay, which resolves to the same edit. A request that arrives resolved
// passes on as the text it arrived as.
func TestClipFormRunRetainsItsRequest(t *testing.T) {
	t.Parallel()
	const directory = "C:/models/LiveEdit"
	clip := latentvideo.ReferenceEditRequest{
		Prompt: "a red fox licking a vanilla ice cream cone on a snowy pine forest trail",
		Seed:   7, SourceArtifact: testutil.ArtifactID(t, artifact.KindFile, "source clip"),
	}
	raw, err := json.Marshal(clip)
	if err != nil {
		t.Fatal(err)
	}
	resolved := latentvideo.ReferenceEditRequest{
		Condition: latentvideo.ReferenceEditCondition{
			TextContext: []float32{0.5}, FramesPerChunk: 3, LocalAttention: 6,
			Timesteps: []int64{1000}, Sigmas: []float32{1}, Seed: clip.Seed,
		},
		Source: latentvideo.SourceVideo{Pixels: make([]float32, 3*21*16*32), Channels: 3, Frames: 21, Height: 16, Width: 32},
	}
	resolve := func(_ context.Context, _ artifact.Reader, path string, request latentvideo.ReferenceEditRequest) (latentvideo.ReferenceEditRequest, error) {
		if path != directory || !reflect.DeepEqual(request, clip) {
			return latentvideo.ReferenceEditRequest{}, fmt.Errorf("resolved %+v at %s, want the clip form at %s", request, path, directory)
		}
		return resolved, nil
	}
	var passed string
	text := func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		passed = input
		return "edited from text", nil
	}
	value := func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, request latentvideo.ReferenceEditRequest, kept artifact.Content) (any, error) {
		if !reflect.DeepEqual(request, resolved) {
			return nil, errors.New("the director was handed another request than the resolved one")
		}
		return capabilityruntime.Measured{Output: "edited clip", Input: kept}, nil
	}
	serve := resolveEditForm(resolve, text, value)
	output, err := serve(t.Context(), nil, directory, modelrecipe.CapabilityEvidenceSelection{}, string(raw))
	if err != nil {
		t.Fatal(err)
	}
	measured, ok := output.(capabilityruntime.Measured)
	if !ok || measured.Output != "edited clip" || passed != "" {
		t.Fatalf("output = %#v with %q passed on as text, want the director's measured edit of the value", output, passed)
	}
	request, retained := capabilityruntime.RetainedRequest(measured.Input)
	var kept latentvideo.ReferenceEditRequest
	if !retained || json.Unmarshal([]byte(request), &kept) != nil || !reflect.DeepEqual(kept, clip) {
		t.Fatalf("the run keeps %.120q (retained %t, schema %q), want the clip form %s",
			request, retained, measured.Input.Descriptor.Schema, raw)
	}
	whole, err := json.Marshal(resolved)
	if err != nil {
		t.Fatal(err)
	}
	if output, err := serve(t.Context(), nil, directory, modelrecipe.CapabilityEvidenceSelection{}, string(whole)); err != nil || output != "edited from text" || passed != string(whole) {
		t.Fatalf("a resolved request gave %v (%v), want it passed on as its text", output, err)
	}
}
