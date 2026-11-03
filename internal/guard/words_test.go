package guard

import "testing"

// TestTheGuardReadsACommandsWords: a rule asks what a command runs and on what, so a word that only looks like a verb or a
// flag is left alone, a heredoc's body is data, and a command behind a prefix, a keyword or a substitution is still that
// command. Each case here is one the regexes it replaced answered the other way.
func TestTheGuardReadsACommandsWords(t *testing.T) {
	t.Parallel()
	for command, refused := range map[string]bool{
		"sudo rm -rf build":                           true,
		"FOO=1 rm -rf build":                          true,
		"for f in a b; do rm -f $f; done":             true,
		"if true; then rm -rf build; fi":              true,
		"/bin/rm -rf build":                           true,
		"env -u OVERGO_X git commit -m wip":           true,
		"git -C ../other clean -fdx":                  true,
		"git -c user.name=a commit -m wip":            true,
		"echo a | xargs -I {} rm -rf {}":              true,
		"out=$(go build ./cmd/plan 2>&1)":             true,
		`rm -r "$S/enforce-merge"`:                    false,
		"rm my-rf-notes.txt":                          false,
		"cat > notes.md <<'EOF'\ngit commit\nEOF":     false,
		"cat > notes.txt <<'EOF'\nrm -rf a\nEOF":      false,
		"cat > a.go <<'EOF'\nplain\nEOF\necho 'a\\b'": false,
	} {
		call := ToolCall{ToolName: "Bash"}
		call.ToolInput.Command = command
		if message, _ := Verdict(call, t.TempDir()); (message != "") != refused {
			t.Errorf("%q is answered %q, want refused %t", command, message, refused)
		}
	}
	if Landing(`echo "go run ./cmd/gate"`) || !Landing(`bash -c "go run ./cmd/gate -plan row"`) {
		t.Error("a landing is what a command runs, not what it says")
	}
}
