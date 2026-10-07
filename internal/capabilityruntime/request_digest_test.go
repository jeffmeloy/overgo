package capabilityruntime

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

// TestOversizedRequestDigestsInline pins the store diet for requests:
// a request over the inline bound is stored as its digest document --
// full-request hash for identity, abbreviated form for the reader --
// while a settings-shaped request stays verbatim.
func TestOversizedRequestDigestsInline(t *testing.T) {
	type request struct {
		Prompt  string    `json:"prompt"`
		Values  []float32 `json:"values"`
		Steps   int       `json:"steps"`
		Invalid bool      `json:"-"`
	}
	program := scalarProgram(t)
	modelID := program.Definition().Model
	small := `{"prompt":"a fox","values":[1,2,3],"steps":4}`
	_, content, err := decodeJSONInput[request]("digest-test", func(request) error { return nil }, modelID, program, small)
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.Schema != "overgo.digest-test-input.v1" {
		t.Fatalf("small request schema = %q", content.Descriptor.Schema)
	}
	var builder strings.Builder
	builder.WriteString(`{"prompt":"a fox","steps":4,"values":[`)
	for index := range 400_000 {
		if index > 0 {
			builder.WriteString(",")
		}
		builder.WriteString("0.5")
	}
	builder.WriteString(`]}`)
	_, digested, err := decodeJSONInput[request]("digest-test", func(request) error { return nil }, modelID, program, builder.String())
	if err != nil {
		t.Fatal(err)
	}
	if digested.Descriptor.Schema != "overgo.digest-test-input-digest.v1" {
		t.Fatalf("oversized request schema = %q", digested.Descriptor.Schema)
	}
	if len(digested.Data) > inlineRequestBytes {
		t.Fatalf("digest document is itself oversized: %d bytes", len(digested.Data))
	}
	var body struct {
		RequestSHA256 string         `json:"request_sha256"`
		RequestBytes  int            `json:"request_bytes"`
		Abbreviated   map[string]any `json:"abbreviated"`
	}
	if err := strictjson.DecodeBytes(digested.Data, &body); err != nil {
		t.Fatal(err)
	}
	if body.RequestBytes <= inlineRequestBytes || body.RequestSHA256 == "" {
		t.Fatalf("digest body = %+v", body)
	}
	if body.Abbreviated["prompt"] != "a fox" || body.Abbreviated["values"] != "[400000 values]" {
		t.Fatalf("abbreviated = %v", body.Abbreviated)
	}
	id, err := artifact.ParseID(body.RequestSHA256)
	if err != nil || id.Kind() != artifact.KindFile {
		t.Fatalf("request identity = %q, %v", body.RequestSHA256, err)
	}
	// The verbatim request is retained to run again; its digest is not.
	if retained, ok := RetainedRequest(content); !ok || retained != string(content.Data) {
		t.Fatalf("retained request = %q, %v", retained, ok)
	}
	if retained, ok := RetainedRequest(digested); ok {
		t.Fatalf("digest retained as a request: %q", retained)
	}
	// A request over the bound of a stored document is digested like any
	// oversized one: a clip's decoded pixels pass that bound at an ordinary
	// size, and the digest never needed the request as a document.
	huge := `{"prompt":"` + strings.Repeat("a", artifact.MaxContentBytes) + `","values":[1],"steps":4}`
	_, beyond, err := decodeJSONInput[request]("digest-test", func(request) error { return nil }, modelID, program, huge)
	if err != nil {
		t.Fatalf("a request over the document bound: %v", err)
	}
	if err := strictjson.DecodeBytes(beyond.Data, &body); err != nil {
		t.Fatal(err)
	}
	if beyond.Descriptor.Schema != "overgo.digest-test-input-digest.v1" || body.RequestBytes <= artifact.MaxContentBytes || len(beyond.Data) > inlineRequestBytes {
		t.Fatalf("request over the document bound = schema %q, %d request bytes, %d digest bytes", beyond.Descriptor.Schema, body.RequestBytes, len(beyond.Data))
	}
	// An input a request maps to a recipe's port is recorded the same way:
	// whole when it is small, by its digest under the schema's digest name
	// when it is bulk, so a stage's record of a decoded clip stays small.
	whole, err := InputContent("overgo.digest-test-port.v1", []float32{1, 2, 3})
	if err != nil || whole.Descriptor.Schema != "overgo.digest-test-port.v1" || string(whole.Data) != "[1,2,3]" {
		t.Fatalf("small port input = schema %q data %q, %v", whole.Descriptor.Schema, whole.Data, err)
	}
	bulk, err := InputContent("overgo.digest-test-port.v1", make([]float32, inlineRequestBytes))
	if err != nil || bulk.Descriptor.Schema != "overgo.digest-test-port-digest.v1" || len(bulk.Data) > inlineRequestBytes {
		t.Fatalf("bulk port input = schema %q, %d bytes, %v", bulk.Descriptor.Schema, len(bulk.Data), err)
	}
}

// scalarProgram reuses the package capability fixture for
// input-decode tests.
func scalarProgram(t *testing.T) recipe.Program {
	t.Helper()
	_, _, program := capabilityFixture(t, "request-digest")
	return program
}
