package main

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestReleaseCommandsIncludeDiffusion(t *testing.T) {
	if len(releaseCommands) != 19 || !slices.Contains(releaseCommands, "overgodb-import") ||
		!slices.Contains(releaseCommands, "overgodb-query") {
		t.Fatalf("release commands = %v", releaseCommands)
	}
}

func TestReleaseDocumentsExist(t *testing.T) {
	root := filepath.Join("..", "..")
	for _, name := range releaseDocuments {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(name))); err != nil {
			t.Errorf("release document %s: %v", name, err)
		}
	}
}

func TestReleaseVersionAndArchiveLayout(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, versionFile), []byte("0.1.1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	version, err := readReleaseVersion(root)
	if err != nil {
		t.Fatal(err)
	}
	if version != "0.1.1" || releaseArchiveName(version) != "overgo-v0.1.1-windows-amd64.zip" {
		t.Fatalf("version=%q archive=%q", version, releaseArchiveName(version))
	}
	if err := os.WriteFile(filepath.Join(root, "bin", "server.exe"), []byte("binary"), 0o600); err != nil {
		t.Fatal(err)
	}
	outside, err := executableOutsideBin(root)
	if err != nil {
		t.Fatal(err)
	}
	if outside != "" {
		t.Fatalf("bin executable reported outside bin: %s", outside)
	}
	if err := os.WriteFile(filepath.Join(root, "stray.exe"), []byte("stray"), 0o600); err != nil {
		t.Fatal(err)
	}
	outside, err = executableOutsideBin(root)
	if err != nil {
		t.Fatal(err)
	}
	if outside != "stray.exe" {
		t.Fatalf("outside executable = %q", outside)
	}
}

// TestBinHygiene pins the bin/ ownership contract: executables are the
// only admitted residents, and any other working file fails the
// release before an archive exists.
func TestBinHygiene(t *testing.T) {
	root := t.TempDir()
	foreign, err := foreignFileInBin(root)
	if err != nil || foreign != "" {
		t.Fatalf("absent bin = (%q, %v)", foreign, err)
	}
	if err := os.Mkdir(filepath.Join(root, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "bin", "server.exe"), []byte("binary"), 0o600); err != nil {
		t.Fatal(err)
	}
	foreign, err = foreignFileInBin(root)
	if err != nil || foreign != "" {
		t.Fatalf("executable-only bin = (%q, %v)", foreign, err)
	}
	if err := os.WriteFile(filepath.Join(root, "bin", "triage.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	foreign, err = foreignFileInBin(root)
	if err != nil || foreign != "bin/triage.json" {
		t.Fatalf("foreign file = (%q, %v)", foreign, err)
	}
}

func TestReleaseIntegrityContract(t *testing.T) {
	for _, historical := range []string{
		"docs/IMPLEMENTATION_LOG.md", "docs/adaptive_new_parity_report.md", "docs/training_plan.md",
	} {
		if slices.Contains(releaseDocuments, historical) {
			t.Errorf("chronological assessment shipped in release: %s", historical)
		}
	}
	for _, durable := range []string{"docs/COMPATIBILITY.md", "docs/TRAINING_COMPATIBILITY.md", "docs/OVERGODB_IMPORT.md", "SBOM.cdx.json"} {
		if !slices.Contains(releaseDocuments, durable) {
			t.Errorf("durable release document missing: %s", durable)
		}
	}
}

func TestReleaseManifestAudit(t *testing.T) {
	command := releaseManifestAuditCommand("repository")
	joined := strings.Join(command.Args, " ")
	if command.Dir != "repository" || !strings.Contains(joined, "-inspect-plan") ||
		!strings.Contains(joined, "-paths cmd,internal") || strings.Contains(joined, "cache") {
		t.Fatalf("release manifest audit = dir %q args %q", command.Dir, joined)
	}
}

func TestCreateArchiveIsDeterministic(t *testing.T) {
	entries := []archiveEntry{
		{Name: "bin/tool.exe", Data: []byte("binary"), Mode: 0o755},
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
