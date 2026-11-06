package sampling

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestOrderedJSONPreservesSchemaPropertyAndConstantOrder(t *testing.T) {
	source := []byte(
		`{"properties":{"z":{"const":{"b":1,"a":2}},"a":{"type":"string"}},` +
			`"required":["z"]}`,
	)
	value := testcheck.Must(parseOrderedJSON(source))(t)
	encoded := testcheck.Must(marshalOrderedJSON(value))(t)
	if encoded != string(source) {
		t.Fatalf("round trip = %s, want %s", encoded, source)
	}
	object, ok := value.(orderedJSONObject)
	if !ok {
		t.Fatalf("root type = %T", value)
	}
	if keys := object.keys(); len(keys) != 2 ||
		keys[0] != "properties" ||
		keys[1] != "required" {
		t.Fatalf("root keys = %v", keys)
	}
}

func TestOrderedJSONRejectsDuplicateKeysTrailingDataAndExcessDepth(t *testing.T) {
	cases := [][]byte{
		[]byte(`{"a":1,"a":2}`),
		[]byte(`{} true`),
	}
	const excessDepth = maxJSONSchemaDepth + 2
	deep := make([]byte, 0, 2*excessDepth)
	for range excessDepth {
		deep = append(deep, '[')
	}
	for range excessDepth {
		deep = append(deep, ']')
	}
	cases = append(cases, deep)
	for _, source := range cases {
		if _, err := parseOrderedJSON(source); err == nil {
			t.Fatalf("accepted invalid JSON %q", source)
		}
	}
}

func TestOrderedJSONAcceptsUTF8BOM(t *testing.T) {
	input := append([]byte{0xef, 0xbb, 0xbf}, []byte(`{"type":"boolean"}`)...)
	value := testcheck.Must(parseOrderedJSON(input))(t)
	if _, ok := value.(orderedJSONObject); !ok {
		t.Fatalf("value type = %T", value)
	}
}
