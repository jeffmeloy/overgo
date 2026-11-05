package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

func requireMediaTestReceipt(t testing.TB, name string, raw []byte, required []string, assertions map[string]map[string][]string) {
	t.Helper()
	var names []string
	for _, tests := range assertions[name] {
		names = append(names, tests...)
	}
	slices.Sort(names)
	names = slices.Compact(names)
	required = slices.Clone(required)
	slices.Sort(required)
	if !slices.Equal(names, required) {
		t.Fatalf("%s: package binding changed the retained assertion set", name)
	}
	testcheck.NoError(t, checkMediaTestReceipt(name, raw, assertions[name]))
}

func TestMediaValidationRetainsFailures(t *testing.T) {
	t.Parallel()
	prefix := `{"Action":"start","Package":"fixture"}` + "\n" + `{"Action":"run","Package":"fixture","Test":"TestGeneration"}` + "\n"
	for _, action := range []string{"pass", "fail", "skip"} {
		data := prefix + `{"Action":"` + action + `","Package":"fixture","Test":"TestGeneration"}` + "\n" + `{"Action":"` + action + `","Package":"fixture"}` + "\n"
		passed, got := mediaValidationVerdict([]byte(data), map[string][]string{"fixture": {"TestGeneration"}})
		if passed != (action == "pass") {
			t.Fatalf("%s: %s", action, got)
		}
	}
	for _, data := range []string{"", "malformed", prefix} {
		if passed, _ := mediaValidationVerdict([]byte(data), map[string][]string{"fixture": {"TestGeneration"}}); passed {
			t.Fatalf("incomplete evidence passed: %q", data)
		}
	}
}

func TestMediaValidationProjection(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := overgodb.Create(filepath.Join(root, "store"))
	testcheck.NoError(t, err)
	defer store.Close()
	contract := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/x-ndjson", Schema: "overgo/go-test-output/v1"}
	data := []byte("{\"Action\":\"start\",\"Package\":\"fixture\"}\n{\"Action\":\"run\",\"Package\":\"fixture\",\"Test\":\"TestGeneration\"}\n{\"Action\":\"fail\",\"Package\":\"fixture\",\"Test\":\"TestGeneration\"}\n{\"Action\":\"fail\",\"Package\":\"fixture\"}\n")
	content, err := contract.ContentBytes(data)
	testcheck.NoError(t, err)
	passed, err := contract.ContentBytes(bytes.ReplaceAll(data, []byte(`"fail"`), []byte(`"pass"`)))
	testcheck.NoError(t, err)
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/media-validation", Contents: []artifact.Content{content, passed}}); err != nil {
		t.Fatal(err)
	}
	missing, err := contract.Identify([]byte("absent"))
	testcheck.NoError(t, err)
	index := mediaValidationIndex{Version: 1, Source: strings.Repeat("1", 40), Environment: "fixture", Checks: []mediaValidationCheck{
		{Name: "Failed generation", Evidence: content.Descriptor.ID, Command: "go test -json", Scope: "Fixture failure"},
		{Name: "Missing acquisition", Evidence: missing, Command: "go test -json", Scope: "Fixture missing content"},
		{Name: "Missing overlay", Evidence: passed.Descriptor.ID, Command: "go test -overlay=fixture.json -json", Scope: "Passing output with an unresolved producer overlay", Harness: map[string]artifact.ID{"historical/fixture_test.go": missing}},
	}}
	index.Notes = []string{"[Retained protocol](docs/image_video_protocol.json); source path `docs/original.go`."}
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "docs"), 0o755))
	encoded, err := json.Marshal(index)
	testcheck.NoError(t, err)
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, mediaValidationPath), encoded, 0o600))
	bindings, err := json.Marshal(map[string]map[string][]string{"Failed generation": {"fixture": {"TestGeneration"}}, "Missing acquisition": {"fixture": {"TestGeneration"}}, "Missing overlay": {"fixture": {"TestGeneration"}}})
	testcheck.NoError(t, err)
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, mediaAssertionsPath), bindings, 0o600))
	var output bytes.Buffer
	projection, err := projectMediaValidation(t.Context(), root, store)
	testcheck.NoError(t, err)
	writeMediaValidation(&output, projection)
	for _, want := range []string{"Not accepted:", "Unavailable: evidence content absent", "retained overlay sources:", "### Acquisition notes (historical)", "](image_video_protocol.json)", "`docs/original.go`", "](image_video_validation.json)", "](image_video_assertions.json)", content.Descriptor.ID.String(), missing.String()} {
		if !strings.Contains(output.String(), want) {
			t.Fatalf("missing %q in report: %s", want, output.String())
		}
	}
	if strings.Contains(output.String(), "Passed declared assertions") {
		t.Fatal("failed or absent generation presented as passing")
	}
}
