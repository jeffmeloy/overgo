package plan

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/processcontrol"
)

// resolveRevision names the repository root and the exact commit revision
// resolves to there.
func resolveRevision(ctx context.Context, repository, revision string) (string, string, error) {
	root, err := gitauthority.RepositoryRoot(ctx, repository)
	if err != nil {
		return "", "", fmt.Errorf("plan: resolve repository: %w", err)
	}
	output, err := gitCommand(ctx, root, gitRevisionCommand, "--verify", "--end-of-options", revision+"^{commit}")
	if err != nil {
		return "", "", fmt.Errorf("plan: resolve revision: %w", err)
	}
	commits := strings.Fields(string(output))
	if len(commits) != 1 {
		return "", "", fmt.Errorf("plan: revision resolved to %d commits", len(commits))
	}
	return root, commits[0], nil
}

// gitCommand runs one git command at repository and returns its output.
func gitCommand(ctx context.Context, repository string, arguments ...string) ([]byte, error) {
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "git", Args: append([]string{"--no-replace-objects"}, arguments...), Dir: filepath.Clean(repository),
		Env: gitauthority.RepositoryEnvironment(), Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, fmt.Errorf("plan: git %s: %w: %s", strings.Join(arguments, " "), err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("plan: git %s: exit=%d: %s", strings.Join(arguments, " "), receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}
