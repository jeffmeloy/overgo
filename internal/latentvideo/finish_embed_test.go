package latentvideo

import (
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
)

// TestAnEditRequestKeepsItsConditionThroughJSON holds the requests that embed a clip's finish to their own fields
// (finish-embed/keep-condition): a LiveEdit request that asks for no finish keeps its condition through JSON, as a request
// handed over as text is decoded and recorded. From 2026-11-09 the finish's IsZero was the condition's too, and the
// request's omitzero left the condition out, so TestVideoProductionActivation's edit was refused as incomplete; the card
// lane that runs it ran for no landing until 2026-11-12, and this host test runs with every change to the package.
func TestAnEditRequestKeepsItsConditionThroughJSON(t *testing.T) {
	t.Parallel()
	request := ReferenceEditRequest{Condition: ReferenceEditCondition{
		TextContext: []float32{1, 2}, FramesPerChunk: 1, LocalAttention: 1, Timesteps: []int64{900}, Sigmas: []float32{0.9},
	}}
	var decoded ReferenceEditRequest
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(json.Marshal(request))(t), &decoded))
	if !slices.Equal(decoded.Condition.TextContext, request.Condition.TextContext) || !slices.Equal(decoded.Condition.Timesteps, request.Condition.Timesteps) ||
		decoded.Condition.FramesPerChunk != request.Condition.FramesPerChunk {
		t.Fatalf("the request came back from JSON with condition %+v, want %+v", decoded.Condition, request.Condition)
	}
}
