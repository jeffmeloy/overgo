package strictjson

import (
	"errors"
	"strings"
	"testing"
)

func TestDecodeRejectsUnknownAndTrailingValues(t *testing.T) {
	var value struct {
		Name string `json:"name"`
	}
	if err := Decode(strings.NewReader(`{"name":"ok"}`), &value); err != nil || value.Name != "ok" {
		t.Fatalf("value = %+v, error = %v", value, err)
	}
	if err := Decode(strings.NewReader(`{"extra":1}`), &value); err == nil {
		t.Fatal("unknown field accepted")
	}
	if err := Decode(strings.NewReader(`{} {}`), &value); !errors.Is(err, ErrTrailingValue) {
		t.Fatalf("error = %v", err)
	}
}

func TestDecodeBoundedRejectsWhitespaceBeyondLimit(t *testing.T) {
	var value map[string]any
	if err := DecodeBounded(strings.NewReader("{}   "), 4, &value); !errors.Is(err, ErrLimit) {
		t.Fatalf("error = %v", err)
	}
}

func TestHasValue(t *testing.T) {
	if HasValue(nil) || HasValue([]byte("null")) || !HasValue([]byte(`{"value":1}`)) {
		t.Fatal("optional JSON classification failed")
	}
}
