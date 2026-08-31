// Package gitauthority owns Git history conditions required by durable
// authority readers and recovery writers.
package gitauthority

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/processcontrol"
)

// RequireCompleteHistory rejects local mechanisms that can hide or rewrite
// ancestry from an authority decision.
func RequireCompleteHistory(ctx context.Context, repository string) error {
	if ctx == nil || strings.TrimSpace(repository) == "" {
		return errors.New("git authority: repository and context are required")
	}
	if replacementBase := strings.TrimSpace(os.Getenv("GIT_REPLACE_REF_BASE")); replacementBase != "" {
		return errors.New("git authority: custom replacement-ref namespace cannot prove immutable Git objects")
	}
	pathOutput, err := output(ctx, repository, "rev-parse", "--path-format=absolute", "--git-path", "info/grafts")
	if err != nil {
		return fmt.Errorf("git authority: locate legacy Git grafts: %w", err)
	}
	graftsPath := filepath.Clean(strings.TrimSpace(string(pathOutput)))
	if graftsPath == "" || !filepath.IsAbs(graftsPath) {
		return errors.New("git authority: legacy Git grafts path is not absolute")
	}
	grafts, err := os.ReadFile(graftsPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("git authority: inspect legacy Git grafts: %w", err)
	}
	if len(grafts) != 0 {
		return errors.New("git authority: nonempty legacy Git grafts cannot prove completion history")
	}
	replacements, err := output(ctx, repository, "for-each-ref", "--format=%(refname)", "refs/replace/")
	if err != nil {
		return fmt.Errorf("git authority: inspect replacement refs: %w", err)
	}
	if strings.TrimSpace(string(replacements)) != "" {
		return errors.New("git authority: replacement refs cannot prove immutable Git objects")
	}
	shallowOutput, err := output(ctx, repository, "rev-parse", "--is-shallow-repository")
	if err != nil {
		return fmt.Errorf("git authority: inspect completion history depth: %w", err)
	}
	if strings.TrimSpace(string(shallowOutput)) != "false" {
		return errors.New("git authority: shallow repository cannot prove complete completion history")
	}
	return nil
}

// RepositoryRoot returns the absolute non-bare worktree root only when the
// requested directory is that exact root. Discovery ignores ambient Git
// overrides and rejects ambiguous output.
func RepositoryRoot(ctx context.Context, repository string) (string, error) {
	if ctx == nil || strings.TrimSpace(repository) == "" {
		return "", errors.New("git authority: repository and context are required")
	}
	repository, err := filepath.Abs(filepath.Clean(repository))
	if err != nil {
		return "", fmt.Errorf("git authority: resolve requested repository: %w", err)
	}
	rootOutput, err := output(ctx, repository, "rev-parse", "--show-toplevel")
	if err != nil {
		return "", fmt.Errorf("git authority: discover repository root: %w", err)
	}
	rawRoot := strings.TrimSpace(string(rootOutput))
	if rawRoot == "" || strings.ContainsAny(rawRoot, "\r\n") || !filepath.IsAbs(rawRoot) {
		return "", errors.New("git authority: discovered repository root is empty, multiline, or non-absolute")
	}
	root, err := filepath.Abs(filepath.Clean(rawRoot))
	if err != nil {
		return "", fmt.Errorf("git authority: resolve discovered repository root: %w", err)
	}
	repositoryInfo, err := os.Stat(repository)
	if err != nil {
		return "", fmt.Errorf("git authority: inspect requested repository: %w", err)
	}
	rootInfo, err := os.Stat(root)
	if err != nil {
		return "", fmt.Errorf("git authority: inspect discovered repository root: %w", err)
	}
	if !repositoryInfo.IsDir() || !rootInfo.IsDir() || !os.SameFile(repositoryInfo, rootInfo) {
		return "", fmt.Errorf("git authority: operation must run at exact repository root %s", root)
	}
	return root, nil
}

// RequireRepositoryRoot proves that repository names the exact worktree root.
func RequireRepositoryRoot(ctx context.Context, repository string) error {
	_, err := RepositoryRoot(ctx, repository)
	return err
}

// RepositoryEnvironment returns the current process environment without Git
// variables. Authority operations must derive repository identity from their
// explicit working directory rather than ambient repository, object, index,
// namespace, or configuration overrides. Git variable names are compared
// case-insensitively because Windows environment names are case-insensitive.
func RepositoryEnvironment() []string {
	inherited := os.Environ()
	environment := make([]string, 0, len(inherited))
	for _, entry := range inherited {
		name, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(strings.ToUpper(name), "GIT_") {
			continue
		}
		environment = append(environment, entry)
	}
	return environment
}

// ReaderEnvironment makes repository-scoped Git inspection non-mutating.
// Several nominally read-only Git commands opportunistically refresh and lock
// the index unless optional locks are disabled. Authority readers must never
// change the state whose identity they are establishing.
func ReaderEnvironment() []string {
	return append(RepositoryEnvironment(), "GIT_OPTIONAL_LOCKS=0")
}

func output(ctx context.Context, repository string, arguments ...string) ([]byte, error) {
	gitArguments := append([]string{"--no-replace-objects"}, arguments...)
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "git", Args: gitArguments, Dir: filepath.Clean(repository), Env: ReaderEnvironment(),
		Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(arguments, " "), err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("git %s: exit=%d: %s", strings.Join(arguments, " "), receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}
