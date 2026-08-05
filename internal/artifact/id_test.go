package artifact

import (
	"bytes"
	"encoding/json"
	"testing"
)

const (
	fixtureModelPayload = "fixture-model"
	fixtureMediaType    = "application/vnd.llamacpp2go.model"
)

func TestIDRoundTrip(t *testing.T) {
	id, err := IdentifyBytes(KindModel, []byte(fixtureModelPayload))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseID(id.String())
	if err != nil {
		t.Fatal(err)
	}
	if parsed != id || parsed.Kind() != KindModel {
		t.Fatalf("round trip = %v, want %v", parsed, id)
	}
	encoded, err := json.Marshal(id)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ID
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != id {
		t.Fatalf("JSON round trip = %v, want %v", decoded, id)
	}
}

func TestIdentityIncludesKind(t *testing.T) {
	model, err := IdentifyBytes(KindModel, []byte(fixtureModelPayload))
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := IdentifyBytes(KindCheckpoint, []byte(fixtureModelPayload))
	if err != nil {
		t.Fatal(err)
	}
	if model == checkpoint || model.Digest() != checkpoint.Digest() {
		t.Fatalf("kind qualification failed: model=%s checkpoint=%s", model, checkpoint)
	}
}

func TestIdentifyReaderReportsSize(t *testing.T) {
	payload := []byte(fixtureModelPayload)
	id, size, err := Identify(KindModel, bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	want, _ := IdentifyBytes(KindModel, payload)
	if id != want || size != uint64(len(payload)) {
		t.Fatalf("identify = (%s, %d), want (%s, %d)", id, size, want, len(payload))
	}
}

func TestDescriptorValidation(t *testing.T) {
	id, _ := IdentifyBytes(KindModel, []byte(fixtureModelPayload))
	valid := Descriptor{ID: id, Size: uint64(len(fixtureModelPayload)), MediaType: fixtureMediaType}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	invalid := valid
	invalid.MediaType = " text/plain "
	if err := invalid.Validate(); err == nil {
		t.Fatal("accepted padded media type")
	}
}

func TestRelationJSONRoundTrip(t *testing.T) {
	encoded, err := json.Marshal(RelationQuantizedFrom)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Relation
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != RelationQuantizedFrom {
		t.Fatalf("relation = %v, want %v", decoded, RelationQuantizedFrom)
	}
}
