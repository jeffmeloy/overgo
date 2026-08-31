package gitauthority

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"overgo/internal/processcontrol"
)

const (
	minimumWriterGitMajor = 2
	minimumWriterGitMinor = 36
	minimumWriterGitPatch = 0

	writerGitVersionPrefix         = "git version "
	writerGitWindowsComponent      = "windows"
	writerGitAppleDecorationPrefix = "(Apple Git-"
)

const (
	writerGitMajorIndex = iota
	writerGitMinorIndex
	writerGitPatchIndex
	writerGitNumericComponentCount
)

const (
	writerGitWindowsLabelIndex     = writerGitNumericComponentCount
	writerGitWindowsBuildIndex     = writerGitWindowsLabelIndex + 1
	writerGitWindowsComponentCount = writerGitWindowsBuildIndex + 1
)

// WriterArguments returns Git command arguments that harden every repository
// component written by an authority transaction. Command-line configuration
// intentionally overrides weaker repository and user configuration for this
// one child process.
func WriterArguments(arguments ...string) []string {
	configured := make([]string, 0, 5+len(arguments))
	configured = append(
		configured,
		"--no-replace-objects",
		"-c", "core.fsync=all",
		"-c", "core.fsyncMethod=fsync",
	)
	return append(configured, arguments...)
}

// RequireWriterSupport proves that the selected Git implements core.fsync,
// core.fsyncMethod, and reference hardening. Git 2.36.0 introduced those
// capabilities; older Git versions silently accept unknown -c keys, so merely
// supplying WriterArguments is not a capability proof.
func RequireWriterSupport(ctx context.Context, repository string) error {
	if ctx == nil || strings.TrimSpace(repository) == "" {
		return errors.New("git authority: writer support requires a repository and context")
	}
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "git", Args: []string{"--version"}, Dir: filepath.Clean(repository),
		Env: RepositoryEnvironment(), Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return fmt.Errorf("git authority: inspect writer capability: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return fmt.Errorf(
			"git authority: inspect writer capability: exit=%d: %s",
			receipt.ExitCode,
			strings.TrimSpace(stderr.String()),
		)
	}
	return requireWriterVersion(stdout.String())
}

func requireWriterVersion(raw string) error {
	line := strings.TrimSpace(raw)
	if !strings.HasPrefix(line, writerGitVersionPrefix) || strings.ContainsAny(line, "\r\n") {
		return errors.New("git authority: Git writer version output is malformed")
	}
	versionAndDecoration := strings.TrimPrefix(line, writerGitVersionPrefix)
	version, decoration, _ := strings.Cut(versionAndDecoration, " ")
	if decoration != "" && (!strings.HasPrefix(decoration, writerGitAppleDecorationPrefix) ||
		!strings.HasSuffix(decoration, ")") || len(decoration) == len(writerGitAppleDecorationPrefix)+1) {
		return errors.New("git authority: Git writer version output has an unsupported decoration")
	}
	parts := strings.Split(version, ".")
	if len(parts) != writerGitNumericComponentCount &&
		(len(parts) != writerGitWindowsComponentCount || parts[writerGitWindowsLabelIndex] != writerGitWindowsComponent) {
		return errors.New("git authority: Git writer version is malformed or prerelease")
	}
	values := make([]int, writerGitNumericComponentCount)
	for index := range values {
		value, err := parseVersionNumber(parts[index])
		if err != nil {
			return errors.New("git authority: Git writer version is malformed")
		}
		values[index] = value
	}
	if len(parts) == writerGitWindowsComponentCount {
		if _, err := parseVersionNumber(parts[writerGitWindowsBuildIndex]); err != nil {
			return errors.New("git authority: Git for Windows writer version is malformed")
		}
	}
	if !writerVersionSupported(values) {
		return fmt.Errorf(
			"git authority: Git writer %d.%d.%d lacks durable object/index/reference configuration; need >= %d.%d.%d",
			values[writerGitMajorIndex], values[writerGitMinorIndex], values[writerGitPatchIndex],
			minimumWriterGitMajor, minimumWriterGitMinor, minimumWriterGitPatch,
		)
	}
	return nil
}

func writerVersionSupported(version []int) bool {
	if version[writerGitMajorIndex] != minimumWriterGitMajor {
		return version[writerGitMajorIndex] > minimumWriterGitMajor
	}
	if version[writerGitMinorIndex] != minimumWriterGitMinor {
		return version[writerGitMinorIndex] > minimumWriterGitMinor
	}
	return version[writerGitPatchIndex] >= minimumWriterGitPatch
}

func parseVersionNumber(raw string) (int, error) {
	if raw == "" {
		return 0, errors.New("empty version number")
	}
	for _, character := range raw {
		if character < '0' || character > '9' {
			return 0, errors.New("non-decimal version number")
		}
	}
	return strconv.Atoi(raw)
}
