package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestSourceOvergoDBPorcelainPreservesWorktreeSpaces(t *testing.T) {
	const target = "0123456789abcdef0123456789abcdef01234567"
	raw := []byte(
		"worktree C:/Users/Jeff/first worktree\x00" +
			"HEAD 89abcdef0123456789abcdef0123456789abcdef\x00" +
			"branch refs/heads/first\x00\x00" +
			"worktree C:/Users/Jeff/target worktree\x00" +
			"HEAD " + target + "\x00" +
			"detached\x00\x00",
	)
	got, err := sourceOvergoDBFromPorcelain(raw, target)
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(filepath.FromSlash("C:/Users/Jeff/target worktree"), "overgodb-store")
	if got != want {
		t.Fatalf("source store = %q, want %q", got, want)
	}
}

func TestSourceOvergoDBPorcelainRejectsAmbiguousHead(t *testing.T) {
	const target = "0123456789abcdef0123456789abcdef01234567"
	raw := []byte(
		"worktree C:/one\x00HEAD " + target + "\x00\x00" +
			"worktree C:/two\x00HEAD " + target + "\x00\x00",
	)
	if _, err := sourceOvergoDBFromPorcelain(raw, target); err == nil ||
		!strings.Contains(err.Error(), "multiple OvergoDB worktrees") {
		t.Fatalf("ambiguous worktree error = %v", err)
	}
}
