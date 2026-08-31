package gitauthority

import (
	"path/filepath"
	"runtime"
	"testing"
)

func TestParseRegisteredWorktreesIsExact(t *testing.T) {
	first := filepath.Join(filepath.VolumeName(filepath.Dir(t.TempDir()))+string(filepath.Separator), "work", "one")
	second := filepath.Join(filepath.VolumeName(filepath.Dir(t.TempDir()))+string(filepath.Separator), "work", "two")
	if runtime.GOOS != "windows" {
		first, second = "/work/one", "/work/two"
	}
	raw := []byte("worktree " + first + "\x00HEAD 1111111111111111111111111111111111111111\x00branch refs/heads/one\x00\x00" +
		"worktree " + second + "\x00HEAD 2222222222222222222222222222222222222222\x00detached\x00\x00")
	worktrees, err := parseRegisteredWorktrees(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(worktrees) != 2 || worktrees[0].root != filepath.Clean(first) ||
		worktrees[0].head != "1111111111111111111111111111111111111111" ||
		worktrees[1].root != filepath.Clean(second) ||
		worktrees[1].head != "2222222222222222222222222222222222222222" {
		t.Fatalf("registered worktrees = %+v", worktrees)
	}
	for _, malformed := range [][]byte{
		[]byte("worktree " + first + "\x00\x00"),
		[]byte("HEAD 1111111111111111111111111111111111111111\x00\x00"),
		[]byte("worktree relative\x00HEAD 1111111111111111111111111111111111111111\x00\x00"),
		[]byte("worktree " + first + "\x00HEAD short\x00\x00"),
	} {
		if _, err := parseRegisteredWorktrees(malformed); err == nil {
			t.Fatalf("malformed listing accepted: %q", malformed)
		}
	}
}
