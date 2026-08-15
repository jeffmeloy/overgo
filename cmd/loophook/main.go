// Command loophook is the overgo loop-control automation, in Go (the repo owns
// its automation in Go; bash is only a one-line hook adapter). It implements the
// three hooks that programmatically hold a session on the plan loop:
//
//	loophook stop         Stop hook.      exit 2 blocks the turn-end, 0 allows.
//	loophook post-commit  PostToolUse.    arms/clears docs/.dispatch_pending.
//	loophook doctrine     SessionStart.   emits the turn contract + dispatch.
//	loophook prompt       UserPromptSubmit. Classifies bounded requests before
//	                      dispatching campaign work.
//
// The stop gate refuses a turn-end that ORPHANS work -- uncommitted repository
// work, or an armed dispatch marker (a commit landed this turn but the next step was not
// dispatched: the milestone-stop). A bare commit is NOT a clean exit. Valves
// keep it from ever wedging: a stop_hook_active retry passes, a live background
// gate steps aside, a fresh recorded stop at HEAD passes, plan-complete passes.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
)

const (
	markerPath  = "docs/.dispatch_pending"
	boundedPath = "docs/.bounded_request"
)

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
	case "prompt":
		runPrompt(readStdin())
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
// block iff work is orphaned (a meaningful dirty path or an armed dispatch marker).
func stopDecision(stopHookActive, boundedRequest, freshStop, gateRunning, planComplete, dirtyWork, markerArmed bool) (block bool) {
	if stopHookActive || boundedRequest || freshStop || gateRunning || planComplete {
		return false
	}
	return dirtyWork || markerArmed
}

func runStop(hookJSON string) int {
	// Re-attempt valve.
	if strings.Contains(hookJSON, `"stop_hook_active":true`) || strings.Contains(hookJSON, `"stop_hook_active": true`) {
		return 0
	}
	boundedRequest := consumeBoundedRequest(boundedPath)
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
	dirtyWork := hasDirtyPlannedScope()
	markerArmed := fileExists(markerPath)

	if !stopDecision(false, boundedRequest, freshStop, gateRunning, complete, dirtyWork, markerArmed) {
		return 0
	}

	var b strings.Builder
	b.WriteString("overgo loop gate: this turn orphans work -- do not end it.\n")
	if dirtyWork {
		b.WriteString("  * uncommitted repository work: commit via 'go run ./cmd/gate -plan <item>/<step> ...' or revert.\n")
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
// the next step is dispatched. A later non-gate command that has left new
// repository work in the tree consumes the boundary (the next step is underway): clear it.
func runPostCommit(hookJSON string) {
	if strings.Contains(hookJSON, "cmd/gate") {
		_ = os.WriteFile(markerPath, []byte(gitHead()+"\n"), 0o644)
		return
	}
	if fileExists(markerPath) && hasDirtyPlannedScope() {
		_ = os.Remove(markerPath)
	}
}

func runDoctrine() {
	// A turn never spans sessions -- any pending boundary is stale.
	_ = os.Remove(markerPath)
	_ = os.Remove(boundedPath)
	fmt.Println(doctrineText)
	fmt.Println()
	fmt.Println("Dispatched now (go run ./cmd/plan -next):")
	next, _ := nextAction()
	fmt.Println(next)
}

func runPrompt(hookJSON string) {
	var input struct {
		Prompt string `json:"prompt"`
	}
	if json.Unmarshal([]byte(hookJSON), &input) == nil && boundedRequestText(input.Prompt) {
		_ = os.WriteFile(boundedPath, []byte("bounded\n"), 0o644)
		fmt.Println("OVERGO BOUNDED REQUEST: answer only this request; it is complete when answered. Do not record a stop or dispatch campaign work.")
		return
	}
	_ = os.Remove(boundedPath)
	command := exec.Command("go", "run", "./cmd/plan", "-prompt")
	command.Stdout, command.Stderr = os.Stdout, os.Stderr
	_ = command.Run()
}

func boundedRequestText(prompt string) bool {
	prompt = strings.ToLower(strings.TrimSpace(prompt))
	if prompt == "" {
		return false
	}
	for _, prefix := range []string{"review ", "summarize ", "explain ", "describe ", "assess ", "evaluate ", "report ", "status", "what ", "why ", "how ", "should ", "would ", "is ", "are ", "does ", "do you think"} {
		if strings.HasPrefix(prompt, prefix) {
			return true
		}
	}
	for _, action := range []string{
		" implement", " build", " create", " add", " remove", " delete", " fix", " change", " update", " refactor", " migrate", " merge", " proceed", " continue", " press on", " keep working", " until done",
	} {
		if strings.Contains(" "+prompt, action) {
			return false
		}
	}
	if strings.HasSuffix(prompt, "?") {
		return true
	}
	return false
}

func consumeBoundedRequest(path string) bool {
	if !fileExists(path) {
		return false
	}
	_ = os.Remove(path)
	return true
}

// --- IO helpers (git + plan + marker), thin wrappers around the pure verdict ---

func gitHead() string {
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "none"
	}
	return strings.TrimSpace(string(out))
}

func hasDirtyPlannedScope() bool {
	out, err := exec.Command("git", "status", "--porcelain=v1", "-z", "--untracked-files=all").Output()
	if err != nil {
		return false
	}
	dirty, err := dirtyPlannedScope(out)
	return err == nil && dirty
}

func dirtyPlannedScope(status []byte) (bool, error) {
	paths, err := repoanalysis.ParseDirtyStatus(status)
	if err != nil {
		return false, err
	}
	return len(paths) > 0, nil
}

func freshStopAtHead(head string) bool {
	b, err := os.ReadFile("docs/plan_stop.json")
	if err != nil {
		return false
	}
	return strings.Contains(string(b), `"head":"`+head+`"`)
}

// gateProcessRunning reports whether a plan-bound gate is committing right now,
// so the Stop gate steps aside instead of false-blocking a turn-end that is
// merely waiting on a background gate.
//
// ACCEPTED RESIDUAL (loop-hardening-7): tasklist is the sole signal, so during
// the ~2s `go run ./cmd/gate` COMPILE phase -- before the temp gate.exe exists --
// this returns false. Ending a turn in that window false-blocks on dirty .go.
// That block is SAFE (it errs toward "commit your work", never toward orphaning)
// and SELF-HEALING: the very next attempt carries stop_hook_active, whose valve
// allows (TestStopDecision "retry valve"). No wedge was ever observed.
// Every candidate fix is a NET LOSS: gate_status.json records only the TERMINAL
// outcome (the early "running" write was deliberately removed after a killed gate
// left a stale one -- cmd/gate doc), and a launch marker armed pre-compile
// re-introduces exactly that hazard -- a killed gate would leave the marker set,
// making this return true while work IS orphaned, so the Stop gate would allow
// ending with uncommitted .go. Trading a safe self-healing transient for an
// unsafe stale-marker residual is the wrong trade; the falsifiable guard is the
// retry valve staying intact (reopen if a compile-window block ever wedges).
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
