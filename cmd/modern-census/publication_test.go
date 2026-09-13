package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/repoanalysis"
)

func TestCombinedCensusPublication(t *testing.T) {
	root := t.TempDir()
	write := func(path string, data []byte) {
		t.Helper()
		name := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	read := func(path string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	const source = "internal/sample/sample.go"
	write("go.mod", []byte("module example\n\ngo 1.26\n"))
	write(source, []byte("package sample\nfunc identity(value string) string { return value }\n"))
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-write-baseline", "-root", root}, io.Discard); err != nil {
		t.Fatal(err)
	}
	initial := read(repoanalysis.ModernGoBaselineFile)
	write(source, append(read(source), []byte("// A source change with no new modernization debt.\n")...))
	cleanSource := read(source)
	var sequential, combined bytes.Buffer
	for _, mode := range []string{"-publish-census", "-lower-baseline"} {
		if err := run([]string{mode, "-root", root}, &sequential); err != nil {
			t.Fatal(err)
		}
	}
	wantBaseline := read(repoanalysis.ModernGoBaselineFile)
	wantCensus := read(repoanalysis.ModernGoPublishedCensusFile)
	write(repoanalysis.ModernGoBaselineFile, initial)
	write(repoanalysis.ModernGoPublishedCensusFile, []byte("previous publication\n"))
	args := []string{"-publish-census", "-lower-baseline", "-root", root}
	if err := run(args, &combined); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wantBaseline, read(repoanalysis.ModernGoBaselineFile)) || !bytes.Equal(wantCensus, read(repoanalysis.ModernGoPublishedCensusFile)) || sequential.String() != combined.String() {
		t.Fatal("combined publication changed the sequential output contract")
	}
	if err := run(args, io.Discard); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wantBaseline, read(repoanalysis.ModernGoBaselineFile)) || !bytes.Equal(wantCensus, read(repoanalysis.ModernGoPublishedCensusFile)) {
		t.Fatal("retry changed an identical publication")
	}
	for _, extra := range []string{"-check", "-census", "-write-baseline"} {
		if err := run(append(slices.Clone(args), extra), io.Discard); err == nil {
			t.Fatalf("combined publication admitted conflicting mode %s", extra)
		}
	}
	// Refusals must preserve both existing outputs.
	write(source, []byte("package sample\nfunc fallback(value, other string) string { if value == \"\" { value = other }; return value }\n"))
	if err := run(args, io.Discard); err == nil || !strings.Contains(err.Error(), "debt increased") {
		t.Fatalf("new debt admitted: %v", err)
	}
	if !bytes.Equal(wantBaseline, read(repoanalysis.ModernGoBaselineFile)) || !bytes.Equal(wantCensus, read(repoanalysis.ModernGoPublishedCensusFile)) {
		t.Fatal("refused debt changed publication outputs")
	}
	// Establish a real exception, then expire its exact identity.
	if err := run([]string{"-write-baseline", "-root", root}, io.Discard); err != nil {
		t.Fatal(err)
	}
	expires := time.Now().UTC().AddDate(1, 0, 0).Format(time.DateOnly)
	if err := run([]string{"-close-exceptions", "-expires", expires, "-root", root}, io.Discard); err != nil {
		t.Fatal(err)
	}
	baselinePath := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoBaselineFile))
	baseline, err := repoanalysis.LoadModernGoBaseline(baselinePath)
	if err != nil {
		t.Fatal(err)
	}
	if len(baseline.Exceptions) == 0 {
		t.Fatal("expiry fixture has no exception")
	}
	for index := range baseline.Exceptions {
		baseline.Exceptions[index].Expires = time.Now().UTC().AddDate(0, 0, -1).Format(time.DateOnly)
	}
	baseline.ExceptionSHA256, err = repoanalysis.ModernGoExceptionIdentity(baseline.Exceptions)
	if err != nil {
		t.Fatal(err)
	}
	if err := jsonfile.Write(baselinePath, baseline, 0o644); err != nil {
		t.Fatal(err)
	}
	expired := read(repoanalysis.ModernGoBaselineFile)
	if err := run(args, io.Discard); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("expired exception admitted: %v", err)
	}
	if !bytes.Equal(expired, read(repoanalysis.ModernGoBaselineFile)) || !bytes.Equal(wantCensus, read(repoanalysis.ModernGoPublishedCensusFile)) {
		t.Fatal("expiry refusal changed publication outputs")
	}
	write(repoanalysis.ModernGoBaselineFile, []byte("invalid baseline\n"))
	if err := run(args, io.Discard); err == nil {
		t.Fatal("invalid baseline admitted")
	}
	if string(read(repoanalysis.ModernGoBaselineFile)) != "invalid baseline\n" || !bytes.Equal(wantCensus, read(repoanalysis.ModernGoPublishedCensusFile)) {
		t.Fatal("invalid baseline refusal changed outputs")
	}
	write(source, cleanSource)
	write(repoanalysis.ModernGoBaselineFile, initial)
	publishedPath := filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoPublishedCensusFile))
	if err := os.Remove(publishedPath); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(publishedPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := run(args, io.Discard); err == nil {
		t.Fatal("publication write failure ignored")
	}
	if !bytes.Equal(initial, read(repoanalysis.ModernGoBaselineFile)) {
		t.Fatal("baseline lowered before failed publication")
	}
	if err := os.Remove(publishedPath); err != nil {
		t.Fatal(err)
	}
	if err := run(args, io.Discard); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wantBaseline, read(repoanalysis.ModernGoBaselineFile)) || !bytes.Equal(wantCensus, read(repoanalysis.ModernGoPublishedCensusFile)) {
		t.Fatal("retry after write failure changed output contract")
	}
}
