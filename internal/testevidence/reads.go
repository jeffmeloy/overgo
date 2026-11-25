package testevidence

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/envsetting"
	"overgo/internal/processcontrol"
)

// Reads are what one package's test binary read while it ran, from the binary's own test log (-test.testlogfile, the
// log go test keeps a pass by): each file it opened, by the digest of its content as the run left it, each path it
// looked at, with whether it was there, and each environment variable it read, with its value. What a process the tests
// start reads (a git, a go build) is in no log, as go test's cache does not see it either (observed-inputs/measure).
type Reads struct {
	// Opened is each file the binary opened, by its absolute path, to the SHA-256 of its content; a directory, or a
	// file gone by the run's end, is the empty digest.
	Opened map[string]string
	// Looked is each path the binary looked at (stat), absolute, to whether it was there when the run ended.
	Looked map[string]bool
	// Environment is each variable the binary read, to its value in the run.
	Environment map[string]string
}

// ReadsOf runs the tests of the package pkg in the module at directory, with the go test arguments given (-run, tags),
// once and with no kept pass, and returns what its test binary read (Reads). A failed run returns its output as the
// error, what it read beside it.
func ReadsOf(ctx context.Context, directory, pkg string, arguments ...string) (Reads, error) {
	var listed bytes.Buffer
	if _, err := processcontrol.Run(ctx, processcontrol.Command{Path: "go", Args: []string{"list", "-f", "{{.Dir}}", pkg}, Dir: directory, Stdout: &listed}); err != nil {
		return Reads{}, fmt.Errorf("testevidence: list %s: %w", pkg, err)
	}
	packageDir := strings.TrimSpace(listed.String())
	logged, err := os.CreateTemp("", "overgo-test-reads-*.log")
	if err != nil {
		return Reads{}, err
	}
	logPath := logged.Name()
	defer os.Remove(logPath)
	if err := logged.Close(); err != nil {
		return Reads{}, err
	}
	var output bytes.Buffer
	args := append(append([]string{"test", pkg, "-count=1"}, arguments...), "-args", "-test.testlogfile="+logPath)
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: "go", Args: args, Dir: directory, Stdout: &output, Stderr: &output})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("testevidence: the tests of %s failed: %s", pkg, strings.TrimSpace(output.String()))
	}
	reads, readErr := readTestLog(logPath, packageDir, func(name string) string {
		return envsetting.Neutral(name, "a test binary's variable is reported with the value the run had")
	})
	return reads, errors.Join(err, readErr)
}

// readTestLog reads a test binary's log at path (lines "open <path>", "stat <path>", "getenv <name>", "chdir <path>"),
// its relative paths taken from where the binary ran, packageDir, or the directory it changed to; value gives an
// environment variable's value.
func readTestLog(path, packageDir string, value func(string) string) (Reads, error) {
	file, err := os.Open(path)
	if err != nil {
		return Reads{}, err
	}
	defer file.Close()
	reads := Reads{Opened: map[string]string{}, Looked: map[string]bool{}, Environment: map[string]string{}}
	working := packageDir
	lines := bufio.NewScanner(file)
	lines.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for lines.Scan() {
		kind, argument, found := strings.Cut(lines.Text(), " ")
		if !found {
			continue
		}
		absolute := argument
		if !filepath.IsAbs(absolute) {
			absolute = filepath.Join(working, argument)
		}
		switch kind {
		case "getenv":
			reads.Environment[argument] = value(argument)
		case "chdir":
			working = absolute
		case "stat":
			_, statErr := os.Stat(absolute)
			reads.Looked[absolute] = statErr == nil
		case "open":
			if _, held := reads.Opened[absolute]; !held {
				reads.Opened[absolute] = contentDigest(absolute)
			}
		}
	}
	return reads, lines.Err()
}

// contentDigest is the SHA-256 of the file's content, empty for a directory or a file that cannot be read.
func contentDigest(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
