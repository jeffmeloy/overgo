package guard

import "testing"

// verdicts runs each shell command through the guard and holds it to refused or allowed.
func verdicts(t *testing.T, commands map[string]bool) {
	t.Helper()
	for command, refused := range commands {
		call := ToolCall{ToolName: "Bash"}
		call.ToolInput.Command = command
		if message, _ := Verdict(call, t.TempDir()); (message != "") != refused {
			t.Errorf("%s: the guard said %q, want refused=%v", command, message, refused)
		}
	}
}

// TestTheTrunkMarkIsTheGatesAlone refuses a command that sets the mark the landing moves the trunk under, or that
// points git at hooks other than the trunk's, however the shell spells the assignment; reading git's settings passes.
func TestTheTrunkMarkIsTheGatesAlone(t *testing.T) {
	t.Parallel()
	verdicts(t, map[string]bool{
		"OVERGO_TRUNK_MOVE=1 git update-ref refs/heads/master HEAD~1": true,
		"env OVERGO_TRUNK_MOVE=1 git merge --ff-only side":            true,
		"export OVERGO_TRUNK_MOVE=1":                                  true,
		"git -c core.hooksPath=/dev/null merge --ff-only side":        true,
		"git -c core.hookspath= reset --hard HEAD~1":                  true,
		"git config --get core.autocrlf":                              false,
		"git log --oneline -1":                                        false,
	})
}

// TestALandingGoesThroughTheLoop refuses the gate run to land, which skips the loop's confirmation from git and its
// refusal's class, and allows the gate's preflight and the loop's landing.
func TestALandingGoesThroughTheLoop(t *testing.T) {
	t.Parallel()
	verdicts(t, map[string]bool{
		"go run ./cmd/gate -plan row/do -message-file msg.txt":               true,
		"go run ./cmd/gate -plan row/do -message-file msg.txt -preflight":    false,
		"go run ./cmd/loop -land row/do -message-file msg.txt":               false,
		"go run ./cmd/gate -plan row/do -message-file - -preflight < msg.md": false,
	})
}

// TestALandingIsReadUnpiped refuses a landing piped into another command, whose exit code the shell reports in the
// landing's place, and allows one run alone and followed by a read of the commit it made.
func TestALandingIsReadUnpiped(t *testing.T) {
	t.Parallel()
	verdicts(t, map[string]bool{
		"go run ./cmd/loop -land row/do -message-file msg.txt | tail -3":                 true,
		"go run ./cmd/loop -land row/do -message-file msg.txt 2>&1 | grep landed":        true,
		"go run ./cmd/loop -land row/do -message-file msg.txt; git log --oneline -1":     false,
		"go run ./cmd/loop -land row/do -message-file msg.txt && git log --oneline -1":   false,
		"go run ./cmd/loop -leftovers | tail -5":                                         false,
		"go run ./cmd/gate -plan row/do -message-file msg.txt -preflight 2>&1 | tail -4": false,
	})
}
