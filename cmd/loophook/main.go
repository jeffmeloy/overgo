// Command loophook is the overgo loop-control automation, in Go (the repo owns
// its automation in Go; bash is only a one-line hook adapter). It implements the
// three hooks that programmatically hold a session on the plan loop:
//
//	loophook stop         Stop hook.      exit 2 blocks the turn-end, 0 allows.
//	loophook post-commit  PostToolUse.    arms/clears docs/.dispatch_pending.
//	loophook doctrine     SessionStart.   emits the turn contract + dispatch.
//
// The stop gate refuses a turn-end that ORPHANS work -- uncommitted .go, or an
// armed dispatch marker (a commit landed this turn but the next step was not
// dispatched: the milestone-stop). A bare commit is NOT a clean exit. Valves
// keep it from ever wedging: a stop_hook_active retry passes, a live background
// gate steps aside, a fresh recorded stop at HEAD passes, plan-complete passes.
package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"overgo/internal/plan"
)

const markerPath = "docs/.dispatch_pending"

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: loophook <stop|post-commit|doctrine>")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "stop":
		os.Exit(runStop(readStdin()))
	case "post-commit":
		runPostCommit(readStdin())
	case "doctrine":
		runDoctrine()
	default:
		fmt.Fprintf(os.Stderr, "loophook: unknown subcommand %q\n", os.Args[1])
		os.Exit(2)
	}
}

func readStdin() string {
	b, _ := io.ReadAll(os.Stdin)
	return string(b)
}

// stopDecision is the pure verdict the Stop hook renders. block==true means the
// turn-end is refused. Allow (block=false) whenever a valve fires; otherwise
// block iff work is orphaned (uncommitted .go or an armed dispatch marker).
func stopDecision(stopHookActive, freshStop, gateRunning, planComplete, dirtyGo, markerArmed bool) (block bool) {
	if stopHookActive || freshStop || gateRunning || planComplete {
		return false
	}
	return dirtyGo || markerArmed
}

func runStop(hookJSON string) int {
	// Re-attempt valve.
	if strings.Contains(hookJSON, `"stop_hook_active":true`) || strings.Contains(hookJSON, `"stop_hook_active": true`) {
		return 0
	}
	head := gitHead()
	freshStop := freshStopAtHead(head)
	if freshStop {
		_ = os.Remove(markerPath)
	}
	gateRunning := gateProcessRunning()
	next, complete := nextAction()
	if complete {
		_ = os.Remove(markerPath)
	}
	dirtyGo := hasDirtyGo()
	markerArmed := fileExists(markerPath)

	if !stopDecision(false, freshStop, gateRunning, complete, dirtyGo, markerArmed) {
		return 0
	}

	var b strings.Builder
	b.WriteString("overgo loop gate: this turn orphans work -- do not end it.\n")
	if dirtyGo {
		b.WriteString("  * uncommitted .go: commit via 'go run ./cmd/gate -plan <item>/<step> ...' or revert.\n")
	}
	if markerArmed {
		b.WriteString("  * a commit landed but the next step is NOT dispatched (the milestone-stop). Run 'go run ./cmd/plan -next' and take its first real action THIS turn.\n")
	}
	if len(next) > 150 {
		next = next[:150]
	}
	b.WriteString("  Dispatched now: " + next + "\n")
	b.WriteString("  A summary/checkpoint/'should I continue?' is NOT a valid end. Only:\n")
	b.WriteString("    go run ./cmd/plan -stop <user-stop|irreversible|external-prereq>: <detail>\n")
	fmt.Fprint(os.Stderr, b.String())
	return 2
}

// runPostCommit owns the dispatch-marker lifecycle. A plan-bound gate call is the
// commit boundary: arm the marker so the Stop gate refuses a milestone-stop until
// the next step is dispatched. A later non-gate command that has left new .go
// work in the tree consumes the boundary (the next step is underway): clear it.
func runPostCommit(hookJSON string) {
	if strings.Contains(hookJSON, "cmd/gate") {
		_ = os.WriteFile(markerPath, []byte(gitHead()+"\n"), 0o644)
		return
	}
	if fileExists(markerPath) && hasDirtyGo() {
		_ = os.Remove(markerPath)
	}
}

func runDoctrine() {
	// A turn never spans sessions -- any pending boundary is stale.
	_ = os.Remove(markerPath)
	fmt.Println(doctrineText)
	fmt.Println()
	fmt.Println("Dispatched now (go run ./cmd/plan -next):")
	next, _ := nextAction()
	fmt.Println(next)
}

// --- IO helpers (git + plan + marker), thin wrappers around the pure verdict ---

func gitHead() string {
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "none"
	}
	return strings.TrimSpace(string(out))
}

func hasDirtyGo() bool {
	out, err := exec.Command("git", "status", "--porcelain", "--", "*.go").Output()
	if err != nil {
		return false
	}
	return strings.TrimSpace(string(out)) != ""
}

func freshStopAtHead(head string) bool {
	b, err := os.ReadFile("docs/plan_stop.json")
	if err != nil {
		return false
	}
	return strings.Contains(string(b), `"head":"`+head+`"`)
}

func gateProcessRunning() bool {
	// go run ./cmd/gate builds a temp exe named gate.exe; match it.
	if out, err := exec.Command("tasklist").Output(); err == nil {
		return strings.Contains(strings.ToLower(string(out)), "gate.exe")
	}
	if out, err := exec.Command("pgrep", "-f", "gate.exe").Output(); err == nil {
		return strings.TrimSpace(string(out)) != ""
	}
	return false
}

// nextAction returns the one-line dispatched step and whether the plan is done.
func nextAction() (string, bool) {
	doc, err := plan.Load("")
	if err != nil {
		return "plan: " + err.Error(), false
	}
	it, st, ok := plan.Current(doc)
	if !ok {
		return "plan complete: every item is done", true
	}
	title := st.Title
	if title == "" {
		title = it.Title
	}
	return fmt.Sprintf("%s / %s: %s", it.ID, st.ID, title), false
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
