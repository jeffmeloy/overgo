package artifact

import (
	"bytes"
	"encoding/json"
	"testing"
)

const (
	fixtureModelPayload = "fixture-model"
	fixtureMediaType    = "application/vnd.overgo.model"
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

// TestIDDecodesInPlace holds an identity's JSON decode to its own bytes: a
// store open decodes every identity it holds, so a plain quoted identity
// parses without a nested JSON decode, and an escaped or malformed one still
// decodes or refuses as JSON does.
func TestIDDecodesInPlace(t *testing.T) {
	id, err := IdentifyBytes(KindModel, []byte(fixtureModelPayload))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(id)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ID
	if allocations := testing.AllocsPerRun(100, func() {
		if err := decoded.UnmarshalJSON(encoded); err != nil {
			t.Fatal(err)
		}
	}); allocations > 2 || decoded != id {
		t.Fatalf("decoding an identity allocated %.0f times, decoded %v", allocations, decoded)
	}
	escaped := bytes.Replace(encoded, []byte(":"), []byte(`:`), 1)
	if err := decoded.UnmarshalJSON(escaped); err != nil || decoded != id {
		t.Fatalf("escaped identity = %v, %v", decoded, err)
	}
	for _, malformed := range []string{`"model:sha256:"`, `"model:sha256:` + id.DigestHex() + `:x"`, `"` + id.String()} {
		if err := decoded.UnmarshalJSON([]byte(malformed)); err == nil {
			t.Fatalf("%s decoded", malformed)
		}
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
	if model == checkpoint || model.digest != checkpoint.digest {
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
