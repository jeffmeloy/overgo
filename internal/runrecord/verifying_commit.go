package runrecord

import (
	"errors"
	"fmt"
	"os/exec"
	"strings"
)

// HeadCommit returns the repository HEAD revision with no cleanliness
// requirement; callers that must bind to committed source use
// VerifyingCommit instead.
func HeadCommit(root string) (string, error) {
	head, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("resolve HEAD commit: %w", err)
	}
	return strings.TrimSpace(string(head)), nil
}

// VerifyingCommit returns HEAD only when tracked and untracked worktree
// state is empty, so a claim cannot identify a commit that differs from
// executed source or fixture bytes. Every capability claim carries this
// commit; the discipline lives beside the claims it protects.
func VerifyingCommit(root string) (string, error) {
	status, err := exec.Command("git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all").Output()
	if err != nil {
		return "", fmt.Errorf("inspect verifying worktree: %w", err)
	}
	if len(status) != 0 {
		return "", errors.New("verifying worktree is dirty; commit or remove every change before recording a claim")
	}
	head, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("resolve verifying commit: %w", err)
	}
	commit := strings.TrimSpace(string(head))
	if !validGitCommit(commit) {
		return "", errors.New("verifying HEAD is not a full Git commit identity")
	}
	return commit, nil
}

func validGitCommit(value string) bool {
	if len(value) != 40 && len(value) != 64 {
		return false
	}
	for _, char := range value {
		if char < '0' || char > '9' && char < 'a' || char > 'f' {
			return false
		}
	}
	return true
}
