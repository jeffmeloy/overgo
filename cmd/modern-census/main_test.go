package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/repoanalysis"
)

func TestModernCensusCommand(t *testing.T) {
	var output bytes.Buffer
	if err := run([]string{"-go-version", "1.26", "-prior"}, &output); err != nil {
		t.Fatal(err)
	}
	text := output.String()
	for _, required := range []string{
		"new_expression go1.26:",
		"time_since go1.0:",
		"catalog=54 applicable=48 excluded=6 target=go1.26",
		"source=40781f167719913666fe2a7dc1c77ea6f256df0a",
		"plugin=v0.1.1",
		"prior 313a14483e4bc43eee727215a2d99259aafc4fbe reimplement:",
		"prior 0c224d095050374431f3e4563a03d07c9beb49b7 replay-candidate:",
	} {
		if !strings.Contains(text, required) {
			t.Errorf("catalog output omits %q\n%s", required, text)
		}
	}
	for _, excluded := range []string{"generic_methods go1.27:", "json_v2 go1.27:"} {
		if strings.Contains(text, excluded) {
			t.Errorf("Go 1.26 output includes %q", excluded)
		}
	}
}

func TestModernGoWorkSelectionReportsCoverage(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "internal", "sample")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	source := []byte("package sample\n\nfunc copyMap(dst, src map[string]int) { for key, value := range src { dst[key] = value } }\n")
	if err := os.WriteFile(filepath.Join(directory, "sample.go"), source, 0o644); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-work", "-root", root, "-rules", "maps_copy", "-limit", "1"}, &output); err != nil {
		t.Fatal(err)
	}
	text := output.String()
	for _, required := range []string{
		"maps_copy risk=mechanical selected=1",
		"verify-direct go test example/internal/sample",
		"modern-work: inspected=1 matched=1 selected=1 excluded=0 unmeasured=0",
	} {
		if !strings.Contains(text, required) {
			t.Errorf("work output omits %q\n%s", required, text)
		}
	}
}

func TestModernCensusExtremaFixReportsAudit(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "internal", "sample")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	source := []byte("package sample\n\nimport \"math\"\n\nfunc maximum(left, right float64) float64 { return math.Max(left, right) }\n")
	if err := os.WriteFile(filepath.Join(directory, "sample.go"), source, 0o644); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-fix-extrema", "-root", root}, &output); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(output.String()); got != "modern-census: extrema rewritten=1 files=1 retained=0 source=typed-ordered-extrema-policy" {
		t.Fatalf("first extrema audit line = %q", got)
	}
	output.Reset()
	if err := run([]string{"-fix-extrema", "-root", root}, &output); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(output.String()); got != "modern-census: extrema rewritten=0 files=0 retained=0 source=typed-ordered-extrema-policy" {
		t.Fatalf("second extrema audit line = %q", got)
	}
}

func TestModernCensusNumericRangeFixReportsAudit(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "internal", "hostmath")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	source := []byte("package hostmath\n\nfunc fill(values []int) { for index := 0; index < len(values); index++ { values[index] = index } }\n")
	if err := os.WriteFile(filepath.Join(directory, "sample.go"), source, 0o644); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-fix-numeric-range", "-root", root}, &output); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(output.String()); got != "modern-census: numeric-range rewritten=1 files=1 retained=0 source=typed-stable-integer-bounds" {
		t.Fatalf("first numeric range audit line = %q", got)
	}
	output.Reset()
	if err := run([]string{"-fix-numeric-range", "-root", root}, &output); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(output.String()); got != "modern-census: numeric-range rewritten=0 files=0 retained=0 source=typed-stable-integer-bounds" {
		t.Fatalf("second numeric range audit line = %q", got)
	}
}

func TestModernCensusCheckRequiresCompleteRatchetAdmission(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "internal", "sample")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "sample.go")
	if err := os.WriteFile(path, []byte("package sample\n\nfunc identity(value string) string { return value }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-write-baseline", "-root", root}, io.Discard); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-publish-census", "-root", root}, io.Discard); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-check", "-root", root}, &output); err != nil {
		t.Fatal(err)
	}
	if text := output.String(); !strings.Contains(text, "check=pass measured=48/48") || !strings.Contains(text, "baseline=docs/modern_go_baseline.json") {
		t.Fatalf("check audit line=%q", text)
	}
	legacy := []byte("package sample\n\nfunc fallback(value, other string) string { if value == \"\" { value = other }; return value }\n")
	if err := os.WriteFile(path, legacy, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-check", "-root", root}, io.Discard); err == nil || !strings.Contains(err.Error(), "debt increased") {
		t.Fatalf("new debt check=%v", err)
	}
}

func TestModernCensusManualIdiomFixReportsAudit(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "internal", "sample")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	source := []byte("package sample\n\nfunc fallback(value, other string) string { if value == \"\" { value = other }; return value }\n")
	if err := os.WriteFile(filepath.Join(directory, "sample.go"), source, 0o644); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-fix-manual-idioms", "-root", root}, &output); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(output.String()); got != "modern-census: manual-idioms fallback=1 ticker=0 slice-clone=0 typed-sort=0 files=1 retained=0 source=typed-eager-safe-equivalence" {
		t.Fatalf("first manual audit line=%q", got)
	}
	output.Reset()
	if err := run([]string{"-fix-manual-idioms", "-root", root}, &output); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(output.String()); got != "modern-census: manual-idioms fallback=0 ticker=0 slice-clone=0 typed-sort=0 files=0 retained=0 source=typed-eager-safe-equivalence" {
		t.Fatalf("second manual audit line=%q", got)
	}
}

func TestModernCensusExceptionClosureIsDeterministic(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "internal", "sample")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	source := []byte("package sample\n\nfunc fallback(value string) string { if value == \"\" { value = derive() }; return value }\nfunc derive() string { return \"fallback\" }\n")
	if err := os.WriteFile(filepath.Join(directory, "sample.go"), source, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-write-baseline", "-root", root}, io.Discard); err != nil {
		t.Fatal(err)
	}
	expires := time.Now().UTC().AddDate(1, 0, 0).Format(time.DateOnly)
	args := []string{"-close-exceptions", "-expires", expires, "-root", root}
	var output bytes.Buffer
	if err := run(args, &output); err != nil {
		t.Fatal(err)
	}
	if text := output.String(); !strings.Contains(text, "exceptions=closed groups=1 candidates=1") || !strings.Contains(text, "authority=") {
		t.Fatalf("exception closure audit line=%q", text)
	}
	name := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoBaselineFile))
	first, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	output.Reset()
	if err := run(args, &output); err != nil {
		t.Fatal(err)
	}
	second, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("second exception closure changed the authority")
	}
	output.Reset()
	if err := run([]string{"-publish-census", "-root", root}, &output); err != nil {
		t.Fatal(err)
	}
	if text := output.String(); !strings.Contains(text, "unresolved=0") || !strings.Contains(text, "excluded=6") {
		t.Fatalf("published census audit line=%q", text)
	}
	publishedName := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoPublishedCensusFile))
	firstPublished, err := os.ReadFile(publishedName)
	if err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-publish-census", "-root", root}, io.Discard); err != nil {
		t.Fatal(err)
	}
	secondPublished, err := os.ReadFile(publishedName)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstPublished, secondPublished) {
		t.Fatal("second census publication changed the evidence")
	}
	if err := run([]string{"-check", "-root", root}, io.Discard); err != nil {
		t.Fatal(err)
	}
}

func TestModernCensusCommandRejectsInvalidVersion(t *testing.T) {
	if err := run([]string{"-go-version", "future"}, io.Discard); err == nil {
		t.Fatal("invalid Go version accepted")
	}
}
