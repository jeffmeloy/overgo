package main

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
)

var releaseCommands = []string{
	"benchmark",
	"block-check",
	"cuda-info",
	"cuda-smoke",
	"diffusion",
	"embedding",
	"generate",
	"gguf-hash",
	"gguf-merge",
	"gguf-quantize",
	"gguf-split",
	"inspect-gguf",
	"json-schema-grammar",
	"model-info",
	"perplexity",
	"overgodb-import",
	"overgodb-query",
	"server",
	"tokenize",
}

var releaseDocuments = []string{
	"VERSION",
	"README.md",
	"docs/COMPATIBILITY.md",
	"docs/TRAINING_COMPATIBILITY.md",
	"docs/OVERGODB_IMPORT.md",
	"compatibility.json",
	"media_policy.json",
	"resource_policy.json",
	"SBOM.cdx.json",
	"kernels/manifest.json",
}

const (
	versionFile          = "VERSION"
	releaseDirectoryMode = os.FileMode(0o755)
)

var versionPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)

type archiveEntry struct {
	Name string
	Data []byte
	Mode os.FileMode
}

func main() {
	out := flag.String("out", "dist", "release output directory")
	verify := flag.Bool("verify-reproducible", false, "build twice and require byte-identical archives")
	flag.Parse()
	if flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: release [-out directory] [-verify-reproducible]")
		os.Exit(1)
	}
	if err := buildRelease(".", *out, *verify); err != nil {
		fmt.Fprintln(os.Stderr, runrecord.LaneError(runrecord.LaneFailed, err.Error()))
		os.Exit(1)
	}
}

func buildRelease(root, output string, verify bool) error {
	if output == "" {
		return errors.New("release: output directory is empty")
	}
	version, err := readReleaseVersion(root)
	if err != nil {
		return err
	}
	outside, err := executableOutsideBin(root)
	if err != nil {
		return err
	}
	if outside != "" {
		return fmt.Errorf("release: executable outside bin: %s", outside)
	}
	foreign, err := foreignFileInBin(root)
	if err != nil {
		return err
	}
	if foreign != "" {
		return fmt.Errorf("release: non-executable file in bin: %s", foreign)
	}
	kernelCheck := exec.Command("go", "run", "./cmd/kernel-manifest")
	kernelCheck.Dir = root
	kernelCheck.Env = releaseEnvironment()
	if output, err := kernelCheck.CombinedOutput(); err != nil {
		return fmt.Errorf("release: kernel manifest verification failed: %w\n%s", err, output)
	}
	sbomCheck := exec.Command("go", "run", "./cmd/sbom", "-check")
	sbomCheck.Dir = root
	sbomCheck.Env = releaseEnvironment()
	if output, err := sbomCheck.CombinedOutput(); err != nil {
		return fmt.Errorf("release: SBOM verification failed: %w\n%s", err, output)
	}
	compatibilityCheck := exec.Command("go", "run", "./cmd/compatibility", "-check")
	compatibilityCheck.Dir = root
	compatibilityCheck.Env = releaseEnvironment()
	if output, err := compatibilityCheck.CombinedOutput(); err != nil {
		return fmt.Errorf("release: compatibility verification failed: %w\n%s", err, output)
	}
	manifestAudit := releaseManifestAuditCommand(root)
	if output, err := manifestAudit.CombinedOutput(); err != nil {
		return fmt.Errorf("release: uncached manifest audit failed: %w\n%s", err, output)
	}
	first, err := buildArchive(root)
	if err != nil {
		return err
	}
	if verify {
		second, secondErr := buildArchive(root)
		if secondErr != nil {
			return secondErr
		}
		if !bytes.Equal(first, second) {
			return errors.New("release: repeated builds are not byte-identical")
		}
	}
	if err := os.MkdirAll(output, releaseDirectoryMode); err != nil {
		return err
	}
	archive := releaseArchiveName(version)
	archivePath := filepath.Join(output, archive)
	if err := os.WriteFile(archivePath, first, clioptions.OutputFileMode); err != nil {
		return err
	}
	sum := sha256.Sum256(first)
	checksum := hex.EncodeToString(sum[:]) + "  " + archive + "\n"
	return os.WriteFile(archivePath+".sha256", []byte(checksum), clioptions.OutputFileMode)
}

func releaseManifestAuditCommand(root string) *exec.Cmd {
	command := exec.Command("go", "run", "./cmd/gate", "-inspect-plan", "-paths", "cmd,internal")
	command.Dir = root
	command.Env = releaseEnvironment()
	return command
}

func buildArchive(root string) ([]byte, error) {
	binaryDirectory := filepath.Join(root, "bin")
	if err := os.MkdirAll(binaryDirectory, releaseDirectoryMode); err != nil {
		return nil, err
	}
	entries := make([]archiveEntry, 0, len(releaseCommands)+len(releaseDocuments)+1)
	for _, name := range releaseCommands {
		output := filepath.Join(binaryDirectory, name+".exe")
		command := exec.Command(
			"go", "build",
			"-trimpath",
			"-buildvcs=false",
			"-ldflags=-s -w -buildid=",
			"-o", output,
			"./cmd/"+name,
		)
		command.Dir = root
		command.Env = releaseEnvironment()
		if output, buildErr := command.CombinedOutput(); buildErr != nil {
			return nil, fmt.Errorf("release: build %s: %w\n%s", name, buildErr, output)
		}
		data, readErr := os.ReadFile(output)
		if readErr != nil {
			return nil, readErr
		}
		entries = append(entries, archiveEntry{Name: path.Join("bin", name+".exe"), Data: data, Mode: 0o755})
	}
	for _, name := range releaseDocuments {
		data, readErr := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
		if readErr != nil {
			return nil, readErr
		}
		entries = append(entries, archiveEntry{Name: name, Data: data, Mode: clioptions.OutputFileMode})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	var checksums strings.Builder
	for _, entry := range entries {
		sum := sha256.Sum256(entry.Data)
		fmt.Fprintf(&checksums, "%s  %s\n", hex.EncodeToString(sum[:]), entry.Name)
	}
	entries = append(entries, archiveEntry{
		Name: "SHA256SUMS",
		Data: []byte(checksums.String()),
		Mode: clioptions.OutputFileMode,
	})
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	return createArchive(entries)
}

func readReleaseVersion(root string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(root, versionFile))
	if err != nil {
		return "", fmt.Errorf("release: read %s: %w", versionFile, err)
	}
	version := strings.TrimSpace(string(raw))
	if !versionPattern.MatchString(version) {
		return "", fmt.Errorf("release: invalid version %q", version)
	}
	return version, nil
}

func releaseArchiveName(version string) string {
	return "overgo-v" + version + "-windows-amd64.zip"
}

func executableOutsideBin(root string) (string, error) {
	raw, err := clioptions.CombinedOutputIn(
		root, os.Environ(), "git", "ls-files", "-co", "--exclude-standard", "-z", "--", ".",
	)
	if err != nil {
		return "", fmt.Errorf("release: enumerate eligible repository files: %w", err)
	}
	var candidates []string
	for name := range strings.SplitSeq(raw, "\x00") {
		name = filepath.ToSlash(name)
		if name == "" || strings.HasPrefix(name, "bin/") || !strings.EqualFold(path.Ext(name), ".exe") {
			continue
		}
		candidates = append(candidates, name)
	}
	slices.Sort(candidates)
	if len(candidates) == 0 {
		return "", nil
	}
	return candidates[0], nil
}

// foreignFileInBin reports the first non-executable file under bin/.
// The directory is the sole executable destination, so anything else
// in it is stray working state a release must not silently absorb.
func foreignFileInBin(root string) (string, error) {
	binRoot := filepath.Join(root, "bin")
	entries, err := os.ReadDir(binRoot)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.EqualFold(filepath.Ext(entry.Name()), ".exe") {
			return path.Join("bin", entry.Name()), nil
		}
	}
	return "", nil
}

func createArchive(entries []archiveEntry) ([]byte, error) {
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	fixed := time.Date(1980, 1, 1, 0, 0, 0, 0, time.UTC)
	for _, entry := range entries {
		if entry.Name == "" ||
			strings.Contains(entry.Name, "\\") ||
			strings.HasPrefix(entry.Name, "/") ||
			path.Clean(entry.Name) != entry.Name ||
			strings.HasPrefix(entry.Name, "../") {
			return nil, fmt.Errorf("release: invalid archive path %q", entry.Name)
		}
		header := &zip.FileHeader{Name: entry.Name, Method: zip.Store}
		header.SetModTime(fixed)
		header.SetMode(entry.Mode)
		file, err := writer.CreateHeader(header)
		if err != nil {
			return nil, err
		}
		if _, err := io.Copy(file, bytes.NewReader(entry.Data)); err != nil {
			return nil, err
		}
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}

func releaseEnvironment() []string {
	result := make([]string, 0, len(os.Environ())+3)
	for _, value := range os.Environ() {
		key, _, _ := strings.Cut(value, "=")
		switch strings.ToUpper(key) {
		case "CGO_ENABLED", "GOOS", "GOARCH":
			continue
		}
		result = append(result, value)
	}
	return append(result, "CGO_ENABLED=0", "GOOS=windows", "GOARCH=amd64")
}
