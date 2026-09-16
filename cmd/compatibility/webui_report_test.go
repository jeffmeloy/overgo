package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/jsonfile"
)

// TestMeasureTreeReadsThisRepository measures the checked-out client: the
// front page is one shell, every tab module counts, and the routes the
// client names are a subset of the manifest's routes.
func TestMeasureTreeReadsThisRepository(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	census, err := measureWebUITree(root, "head")
	if err != nil {
		t.Fatal(err)
	}
	if census.Shells != 1 || census.Modules == 0 || census.JavaScriptLines == 0 {
		t.Fatalf("census does not describe the client: %+v", census)
	}
	if census.ClientRoutes == 0 || census.ClientRoutes > census.Routes || census.BearerRoutes > census.Routes {
		t.Fatalf("route counts are inconsistent: %+v", census)
	}
	if census.StreamReaderSites != 1 {
		t.Fatalf("stream reader sites = %d, the ratchet holds one", census.StreamReaderSites)
	}
}

func TestWebUIReport(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "comparison.json")
	spec := map[string]any{"before": map[string]string{"root": root, "label": "same"}, "after": map[string]string{"root": root, "label": "same"}}
	if err := jsonfile.Write(path, spec, 0600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := writeWebUIReport(path, &output); err != nil {
		t.Fatal(err)
	}
	var result struct{ Before, After webuiCensus }
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Before, result.After) || result.Before.Tree != "same" || result.Before.ClientRoutes == 0 {
		t.Fatalf("inconsistent comparison: %+v", result)
	}
	first := output.String()
	output.Reset()
	if err := writeWebUIReport(path, &output); err != nil || output.String() != first {
		t.Fatalf("non-deterministic output: %v", err)
	}
	closed, err := os.CreateTemp(t.TempDir(), "closed")
	if err != nil {
		t.Fatal(err)
	}
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	if err := writeWebUIReport(path, closed); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("output failure lost: %v", err)
	}
	for _, invalid := range []string{`{}`, `{"unknown":true}`, `{"before":{"root":"absent"},"after":{"root":"absent"}}`, `{} {}`} {
		if err := os.WriteFile(path, []byte(invalid), 0600); err != nil {
			t.Fatal(err)
		}
		output.Reset()
		if err := writeWebUIReport(path, &output); err == nil || output.Len() != 0 {
			t.Fatalf("invalid spec published output: %s error=%v", invalid, err)
		}
	}
}
