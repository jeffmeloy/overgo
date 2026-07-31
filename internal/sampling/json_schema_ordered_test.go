package sampling

import "testing"

func TestOrderedJSONPreservesSchemaPropertyAndConstantOrder(t *testing.T) {
	source := []byte(
		`{"properties":{"z":{"const":{"b":1,"a":2}},"a":{"type":"string"}},` +
			`"required":["z"]}`,
	)
	value, err := parseOrderedJSON(source)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := marshalOrderedJSON(value)
	if err != nil {
		t.Fatal(err)
	}
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
	deep := make([]byte, 0, 600)
	for range 258 {
		deep = append(deep, '[')
	}
	for range 258 {
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
	value, err := parseOrderedJSON(input)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := value.(orderedJSONObject); !ok {
		t.Fatalf("value type = %T", value)
	}
}
