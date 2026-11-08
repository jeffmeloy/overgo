package guard

import (
	"cmp"
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unicode"

	"overgo/internal/gitauthority"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
)

// heldBy is the session the hold of the checkout at root names (WorktreeHolder), when that is another session than
// session and its process still runs, or was not recorded: a hold whose process has exited names a session that has
// ended, which the next implementing session's turn clears.
func heldBy(root, session string) string {
	raw, err := os.ReadFile(filepath.Join(root, processlock.StateDirectory, WorktreeHolder))
	if err != nil {
		return ""
	}
	held, process := ReadHold(raw)
	if held == session || process > 0 && !processcontrol.ProcessAlive(process) {
		return ""
	}
	return held
}

// trunkCheckout is the trunk's checkout of the repository root is a worktree of: the main worktree, whose git directory
// every worktree shares. Outside a repository it is none.
func trunkCheckout(root string) string {
	common, err := gitauthority.Query(context.Background(), root, "rev-parse", "--path-format=absolute", "--git-common-dir")
	if err != nil {
		return ""
	}
	return filepath.Dir(filepath.FromSlash(strings.TrimSpace(string(common))))
}

// within is path relative to directory, and whether it lies inside it.
func within(directory, path string) (string, bool) {
	relative, err := filepath.Rel(directory, path)
	return relative, directory != "" && err == nil && relative != "." && filepath.IsLocal(relative)
}

// trackedWrite is the refusal of who's write of the path relative to checkout, which the repository tracks or would
// track; a path it ignores is scratch, and nothing.
func trackedWrite(checkout, relative, who, instead string) string {
	if _, ignored := gitauthority.Query(context.Background(), checkout, "check-ignore", "-q", relative); ignored == nil {
		return ""
	}
	return who + " writes no file the repository tracks or would track (" + filepath.ToSlash(relative) + "); " + instead + ", or write to a scratch path"
}

// written are the files a call writes, each as an absolute path: the one a Write or Edit names, and those a shell
// command writes (writes), a relative one read from the directory the call works in. A flag, or a path the shell
// expands, names no file the guard can read.
func written(call ToolCall, root string) []string {
	named := []string{call.ToolInput.Path, call.ToolInput.Notebook}
	for _, payload := range payloads(call.ToolInput.Command, 0) {
		for _, run := range readShell(payload).programs() {
			named = append(named, writes(run)...)
		}
	}
	var paths []string
	for _, name := range named {
		if name == "" || harmlessSink(name) || strings.HasPrefix(name, "-") || strings.ContainsAny(name, "$`*?") {
			continue
		}
		if name = drivePath(name); !filepath.IsAbs(name) {
			name = filepath.Join(cmp.Or(call.Directory, root), name)
		}
		paths = append(paths, filepath.Clean(name))
	}
	return paths
}

// drivePath reads a POSIX shell's spelling of a Windows drive, /c/Users, as the drive's own, c:/Users.
func drivePath(name string) string {
	rest, rooted := strings.CutPrefix(name, "/")
	drive, after, nested := strings.Cut(rest, "/")
	if runtime.GOOS != "windows" || !rooted || !nested || len(drive) != 1 || !unicode.IsLetter(rune(drive[0])) {
		return name
	}
	return drive + ":/" + after
}
