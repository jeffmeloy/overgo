package server

import (
	"encoding/json"
	"testing"
)

func TestPointerPresenceSemantics(t *testing.T) {
	absent, err := json.Marshal(responsesStreamEvent{Type: "response.test"})
	switch err {
	case nil:
	default:
		t.Fatal(err)
	}
	present, err := json.Marshal(responsesStreamEvent{
		OutputIndex: new(firstEventIndex), Text: new(""), Type: "response.test",
	})
	switch err {
	case nil:
	default:
		t.Fatal(err)
	}
	if string(absent) != `{"type":"response.test"}` {
		t.Fatalf("nil pointers changed omission semantics: %s", absent)
	}
	if string(present) != `{"output_index":0,"text":"","type":"response.test"}` {
		t.Fatalf("zero-value pointers changed presence semantics: %s", present)
	}
}
