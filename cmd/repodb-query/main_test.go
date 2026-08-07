package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

func TestRunQueriesSeededStore(t *testing.T) {
	root := t.TempDir()
	store, err := repodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := artifact.IdentifyBytes(artifact.KindDataset, []byte("dataset"))
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/query-cli",
		Artifacts: []artifact.Descriptor{{ID: id, Size: 7}},
		Aliases:   []artifact.AliasBinding{{Name: "dataset/current", Target: id}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var text bytes.Buffer
	if err := run([]string{"-repo", root, "-kind", "dataset"}, &text); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text.String(), id.String()) || !strings.Contains(text.String(), "dataset/current") {
		t.Fatalf("text output = %q", text.String())
	}
	var encoded bytes.Buffer
	if err := run([]string{"-repo", root, "-alias", "dataset/current", "-json"}, &encoded); err != nil {
		t.Fatal(err)
	}
	var result repodb.QueryResult
	if err := json.Unmarshal(encoded.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Artifacts) != 1 || result.Artifacts[0].ID != id {
		t.Fatalf("JSON result = %+v", result)
	}
}

func TestRunRejectsUnboundedAndInvalidFilters(t *testing.T) {
	if err := run(nil, &bytes.Buffer{}); err == nil {
		t.Fatal("missing repository accepted")
	}
	if err := run([]string{"-repo", "x", "-limit", "0"}, &bytes.Buffer{}); err == nil {
		t.Fatal("zero result bound accepted")
	}
	if _, err := parseFollow("sideways"); err == nil {
		t.Fatal("invalid follow direction accepted")
	}
}
