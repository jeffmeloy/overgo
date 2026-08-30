// Command loophook is the overgo loop-control automation, in Go (the repo owns
// its automation in Go; bash is only a one-line hook adapter). It implements the
// three hooks that programmatically hold a session on the plan loop:
//
//	loophook stop         Stop hook.      exit 2 blocks the turn-end, 0 allows.
//	loophook post-commit  PostToolUse.    re-baselines turn dirt at gate commits.
//	loophook doctrine     SessionStart.   emits the turn contract + dispatch.
//	loophook prompt       UserPromptSubmit. Classifies bounded requests before
//	                      dispatching campaign work.
//
// The stop gate protects WORK and the LOOP: it refuses a turn-end that would
// orphan turn-created uncommitted work, and it refuses a turn-end while the
// plan holds ANY open row -- committed progress changes the message, never
// the verdict, so the loop runs until the plan is empty. The one sanctioned
// pause is a recorded user stop. Valves keep it from ever wedging: a
// stop_hook_active retry passes, a live background gate steps aside, a fresh
// recorded stop at HEAD passes, plan-complete passes.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
)

const (
	boundedPath = "docs/.bounded_request"
	// turnBasePath snapshots dirt at UserPromptSubmit so the Stop gate
	// blocks only on TURN-CREATED dirt. Pre-existing dirt is another lane's
	// parked in-flight work.
	turnBasePath = "docs/.loop_state"
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

// Stop verdict reasons. Allow whenever a valve fires; otherwise a turn-end is
// refused for one of two reasons: it would orphan turn-created work, or it
// would pause the loop while the plan still holds an open row (owner rule
// 2026-08-20: "the need to continue the plan doesn't go away unless I say
// stop"). Committed progress is necessary, never sufficient: a turn under an
// open plan row continues until the plan is EMPTY, and the only sanctioned
// pause is a recorded user stop (go run ./cmd/plan -stop), which arms the
// freshStop valve.
const (
	stopAllow    = ""
	stopOrphan   = "orphan"
	stopContinue = "continue"
)

// stopDecision is the pure verdict the Stop hook renders. planComplete is the
// only plan-derived allow; a landed commit changes the message, never the
// verdict.
func stopDecision(stopHookActive, freshStop, gateRunning, planComplete, dirtyWork bool) string {
	if stopHookActive || freshStop || gateRunning || planComplete {
		return stopAllow
	}
	if dirtyWork {
		return stopOrphan
	}
	return stopContinue
}

func runStop(hookJSON string) int {
	// Re-attempt valve.
	if strings.Contains(hookJSON, `"stop_hook_active":true`) || strings.Contains(hookJSON, `"stop_hook_active": true`) {
		return 0
	}
	// The bounded marker scopes the ANSWER, never the loop; consume it so it
	// cannot go stale, but it renders no verdict.
	consumeBoundedRequest(boundedPath)
	freshStop := freshStopAtHead(gitHead())
	gateRunning := gateProcessRunning()
	next, complete := nextAction()
	dirtyWork := hasTurnCreatedDirt()
	progressed := headAdvancedSinceTurnStart()

	verdict := stopDecision(false, freshStop, gateRunning, complete, dirtyWork)
	if verdict == stopAllow {
		return 0
	}

	if len(next) > 150 {
		next = next[:150]
	}
	var b strings.Builder
	switch verdict {
	case stopOrphan:
		b.WriteString("overgo loop gate: this turn would orphan uncommitted work.\n")
		b.WriteString("  * commit via 'go run ./cmd/gate -plan <item>/<step> ...' or revert it.\n")
	case stopContinue:
		if progressed {
			b.WriteString("overgo loop gate: the row landed, but the plan still holds open rows.\n")
			b.WriteString("  * continue with the next dispatched row now; a turn ends only when the\n")
			b.WriteString("    plan is EMPTY. To pause: go run ./cmd/plan -stop user-stop:<detail>.\n")
		} else {
			b.WriteString("overgo loop gate: a prompt answer does not pause the loop (owner rule).\n")
			b.WriteString("  * continue the dispatched row now; a turn ends only when the plan is\n")
			b.WriteString("    EMPTY. To pause: go run ./cmd/plan -stop user-stop:<detail>.\n")
		}
	}
	b.WriteString("  Dispatched now: " + next + "\n")
	fmt.Fprint(os.Stderr, b.String())
	return 2
}

// runPostCommit re-baselines the turn-dirt snapshot at a gate commit boundary
// so the orphan check measures only work created after the commit. The old
// dispatch-marker lifecycle is retired: cmd/loop owns continuity.
func runPostCommit(hookJSON string) {
	if strings.Contains(hookJSON, "cmd/gate") {
		writeTurnBase()
	}
}

func runDoctrine() {
	// A turn never spans sessions -- any pending boundary is stale.
	_ = os.Remove(boundedPath)
	_ = os.Remove(turnBasePath)
	fmt.Println(doctrineText)
	fmt.Println()
	fmt.Println("Dispatched now (go run ./cmd/plan -next):")
	next, _ := nextAction()
	fmt.Println(next)
}

func runPrompt(hookJSON string) {
	// Every turn starts by snapshotting the dirty set: the Stop gate then owes a
	// block only to dirt this turn creates, not to another lane's parked work.
	writeTurnBase()
	var input struct {
		Prompt string `json:"prompt"`
	}
	if json.Unmarshal([]byte(hookJSON), &input) == nil && boundedRequestText(input.Prompt) {
		_ = os.WriteFile(boundedPath, []byte("bounded\n"), 0o644)
		fmt.Println("OVERGO BOUNDED REQUEST: answer this request concisely, THEN continue the dispatched plan row in this same turn -- a prompt never pauses the loop (owner rule). Only an explicit user stop pauses; record it with go run ./cmd/plan -stop user-stop:<detail>.")
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

type dirtyFact struct {
	Path           string `json:"path"`
	OriginalPath   string `json:"original_path,omitempty"`
	IndexStatus    string `json:"index_status"`
	WorktreeStatus string `json:"worktree_status"`
	IndexIdentity  string `json:"index_identity"`
	WorkIdentity   string `json:"work_identity"`
}

// currentDirtyFacts binds path, index state, and worktree bytes.
func currentDirtyFacts() ([]dirtyFact, error) {
	out, err := exec.Command("git", "status", "--porcelain=v1", "-z", "--untracked-files=all").Output()
	if err != nil {
		return nil, err
	}
	paths, err := repoanalysis.ParseDirtyStatus(out)
	if err != nil {
		return nil, err
	}
	facts := make([]dirtyFact, 0, len(paths))
	for _, path := range paths {
		index, _ := exec.Command("git", "ls-files", "--stage", "--", path.Path).Output()
		facts = append(facts, dirtyFact{
			Path: path.Path, OriginalPath: path.OriginalPath,
			IndexStatus: path.IndexStatus, WorktreeStatus: path.WorktreeStatus,
			IndexIdentity: strings.TrimSpace(string(index)), WorkIdentity: workIdentity(path.Path),
		})
	}
	return facts, nil
}

// turnBase is the turn-start snapshot: the dirty set (orphan check) and the
// HEAD commit (loop-progress check).
type turnBase struct {
	Head  string      `json:"head"`
	Facts []dirtyFact `json:"facts"`
}

// writeTurnBase snapshots dirt and HEAD at a prompt or commit boundary.
func writeTurnBase() {
	facts, err := currentDirtyFacts()
	if err != nil {
		return
	}
	data, err := json.Marshal(turnBase{Head: gitHead(), Facts: facts})
	if err == nil {
		_ = os.WriteFile(turnBasePath, append(data, '\n'), 0o600)
	}
}

// readTurnBase parses the snapshot; ok=false on a missing or legacy file.
func readTurnBase() (turnBase, bool) {
	raw, err := os.ReadFile(turnBasePath)
	if err != nil {
		return turnBase{}, false
	}
	return readTurnBaseFrom(raw)
}

// readTurnBaseFrom parses snapshot bytes; ok=false for the legacy headless
// (facts-array) format.
func readTurnBaseFrom(raw []byte) (turnBase, bool) {
	var base turnBase
	if json.Unmarshal(raw, &base) != nil || base.Head == "" {
		return turnBase{}, false
	}
	return base, true
}

// headAdvancedSinceTurnStart reports whether a gate commit landed this turn.
// A missing or legacy snapshot fails toward ALLOW (progressed=true): without a
// baseline the gate cannot prove the turn made no progress, and the harassment
// direction is the wrong error.
func headAdvancedSinceTurnStart() bool {
	base, ok := readTurnBase()
	if !ok {
		return true
	}
	return gitHead() != base.Head
}

// hasTurnCreatedDirt reports whether dirt exists beyond the turn-start snapshot.
// A missing or unreadable snapshot FAILS SAFE toward the old behavior -- all
// dirt blocks -- so a lost marker errs toward "commit your work", never toward
// orphaning it.
func hasTurnCreatedDirt() bool {
	current, currentErr := currentDirtyFacts()
	if currentErr != nil {
		return true
	}
	base, ok := readTurnBase()
	if !ok {
		return len(current) > 0
	}
	return len(turnCreatedDirt(base.Facts, current)) > 0
}

// turnCreatedDirt returns new or further-modified dirty facts.
func turnCreatedDirt(base, current []dirtyFact) []dirtyFact {
	parked := make(map[dirtyFact]bool, len(base))
	for _, fact := range base {
		parked[fact] = true
	}
	var created []dirtyFact
	for _, fact := range current {
		if !parked[fact] {
			created = append(created, fact)
		}
	}
	return created
}

func workIdentity(path string) string {
	file, err := os.Open(filepath.FromSlash(path))
	if err != nil {
		return "missing"
	}
	hash := sha256.New()
	_, copyErr := io.Copy(hash, file)
	closeErr := file.Close()
	if copyErr != nil || closeErr != nil {
		return "unreadable"
	}
	return fmt.Sprintf("%x", hash.Sum(nil))
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
	role, err := plan.AutomationRole("")
	if err != nil {
		return "", false
	}
	store, err := overgodb.OpenReadOnly("overgodb-store")
	if err != nil {
		return "plan: " + err.Error(), false
	}
	defer store.Close()
	authority, err := plan.ResolveCompletionAuthority(context.Background(), ".", "HEAD", doc, store)
	if err != nil {
		return "plan: " + err.Error(), false
	}
	it, st, ok := plan.Current(doc, role, authority)
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
