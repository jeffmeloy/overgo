package mediacapability

import (
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentvideo"
	"overgo/internal/vqaserve"
)

// TestArtifactControlsDeclareTheirSlot pins the slot an artifact field
// declares through its tags: the label a page shows and the media kind it
// takes; a field without the tags is a plain artifact control, and the
// shipped requests declare theirs.
func TestArtifactControlsDeclareTheirSlot(t *testing.T) {
	type request struct {
		First  artifact.ID `json:"first" label:"first frame" media:"image"`
		Plain  artifact.ID `json:"plain,omitzero"`
		Prompt string      `json:"prompt"`
	}
	controls, refusal := describe(reflect.TypeOf(request{}))
	if refusal != "" || len(controls) != 3 {
		t.Fatalf("controls = %+v (%s)", controls, refusal)
	}
	if label, media := controls[0].Slot(); label != "first frame" || media != "image" || !controls[0].Required {
		t.Fatalf("first = %+v", controls[0])
	}
	if label, media := controls[1].Slot(); label != "" || media != "" || controls[1].Required {
		t.Fatalf("plain = %+v", controls[1])
	}
	for _, shipped := range []struct {
		request     any
		name, media string
	}{{vqaserve.Request{}, "image", "image"}, {latentvideo.ReferenceEditRequest{}, "source_artifact", "video"}} {
		controls, refusal := describe(reflect.TypeOf(shipped.request))
		if refusal != "" {
			t.Fatal(refusal)
		}
		found := false
		for _, control := range controls {
			if _, media := control.Slot(); control.Name == shipped.name && media == shipped.media && control.Label != "" {
				found = true
			}
		}
		if !found {
			t.Fatalf("%s lacks its %s slot: %+v", shipped.name, shipped.media, controls)
		}
	}
}
