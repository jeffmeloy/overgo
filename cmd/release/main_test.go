package main

import (
	"archive/zip"
	"bytes"
	"slices"
	"testing"
)

func TestReleaseCommandsIncludeDiffusion(t *testing.T) {
	if len(releaseCommands) != 18 || !slices.Contains(releaseCommands, "repodb-query") {
		t.Fatalf("release commands = %v", releaseCommands)
	}
}

func TestCreateArchiveIsDeterministic(t *testing.T) {
	entries := []archiveEntry{
		{Name: "tool.exe", Data: []byte("binary"), Mode: 0o755},
		{Name: "README.md", Data: []byte("readme"), Mode: 0o644},
	}
	first, err := createArchive(entries)
	if err != nil {
		t.Fatal(err)
	}
	second, err := createArchive(entries)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("archives differ")
	}
	reader, err := zip.NewReader(bytes.NewReader(first), int64(len(first)))
	if err != nil {
		t.Fatal(err)
	}
	if len(reader.File) != 2 || reader.File[0].Modified.Year() != 1980 {
		t.Fatalf("archive files = %+v", reader.File)
	}
}

func TestCreateArchiveRejectsUnsafePath(t *testing.T) {
	for _, name := range []string{`..\bad`, "../bad", "/absolute"} {
		if _, err := createArchive([]archiveEntry{{Name: name, Data: []byte("x")}}); err == nil {
			t.Fatalf("unsafe archive path %q was accepted", name)
		}
	}
}
