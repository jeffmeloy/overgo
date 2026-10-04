package mediacapability

import (
	"context"
	"encoding/json"
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
// the form resolver. The session director receives the resolved request,
// whose condition and pixels a run keeps only as a digest, and the run
// keeps the clip form as its request instead: a verification of it then
// holds a request recipe measure can replay, which resolves to the same edit.
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
	// The director's envelope around the resolved request, which it keeps
	// only as its digest: the production clip resolves to 11 MB.
	director := func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		var request latentvideo.ReferenceEditRequest
		if err := json.Unmarshal([]byte(input), &request); err != nil || !reflect.DeepEqual(request, resolved) {
			return nil, fmt.Errorf("the director received %.80s, not the resolved request (%v)", input, err)
		}
		digest, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.video-gen-input-digest.v1"),
			map[string]int{"request_bytes": len(input)})
		return capabilityruntime.Measured{Output: "edited clip", Input: digest}, err
	}
	output, err := resolveEditForm(resolve, director)(t.Context(), nil, directory, modelrecipe.CapabilityEvidenceSelection{}, string(raw))
	if err != nil {
		t.Fatal(err)
	}
	measured, ok := output.(capabilityruntime.Measured)
	if !ok || measured.Output != "edited clip" {
		t.Fatalf("output = %#v, want the director's measured edit", output)
	}
	request, retained := capabilityruntime.RetainedRequest(measured.Input)
	var kept latentvideo.ReferenceEditRequest
	if !retained || json.Unmarshal([]byte(request), &kept) != nil || !reflect.DeepEqual(kept, clip) {
		t.Fatalf("the run keeps %.120q (retained %t, schema %q), want the clip form %s",
			request, retained, measured.Input.Descriptor.Schema, raw)
	}
}
