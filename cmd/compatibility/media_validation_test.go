package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestMediaValidationRetainsFailures(t *testing.T) {
	prefix := `{"Action":"start","Package":"fixture"}` + "\n" + `{"Action":"run","Package":"fixture","Test":"TestGeneration"}` + "\n"
	for _, action := range []string{"pass", "fail", "skip"} {
		data := prefix + `{"Action":"` + action + `","Package":"fixture","Test":"TestGeneration"}` + "\n" + `{"Action":"` + action + `","Package":"fixture"}` + "\n"
		got := mediaValidationVerdict([]byte(data))
		if strings.HasPrefix(got, "Passed") != (action == "pass") {
			t.Fatalf("%s: %s", action, got)
		}
	}
	for _, data := range []string{"", "malformed", prefix} {
		if strings.HasPrefix(mediaValidationVerdict([]byte(data)), "Passed") {
			t.Fatalf("incomplete evidence passed: %q", data)
		}
	}
}

func TestMediaValidationProjection(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	contract := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/x-ndjson", Schema: "overgo/go-test-output/v1"}
	data := []byte("{\"Action\":\"start\",\"Package\":\"fixture\"}\n{\"Action\":\"run\",\"Package\":\"fixture\",\"Test\":\"TestGeneration\"}\n{\"Action\":\"fail\",\"Package\":\"fixture\",\"Test\":\"TestGeneration\"}\n{\"Action\":\"fail\",\"Package\":\"fixture\"}\n")
	content, err := contract.ContentBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "test/media-validation", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	missing, err := contract.Identify([]byte("absent"))
	if err != nil {
		t.Fatal(err)
	}
	index := mediaValidationIndex{Version: 1, Source: strings.Repeat("1", 40), Environment: "fixture", Checks: []mediaValidationCheck{
		{Name: "Failed generation", Evidence: content.Descriptor.ID, Command: "go test -json", Scope: "Fixture failure"},
		{Name: "Missing acquisition", Evidence: missing, Command: "go test -json", Scope: "Fixture missing content"},
	}}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, mediaValidationPath), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := writeMediaValidation(t.Context(), &output, root, store); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Not passed:", "Unavailable: evidence content absent", content.Descriptor.ID.String(), missing.String()} {
		if !strings.Contains(output.String(), want) {
			t.Fatalf("missing %q in report: %s", want, output.String())
		}
	}
	if strings.Contains(output.String(), "Passed selected tests") {
		t.Fatal("failed or absent generation presented as passing")
	}
}
