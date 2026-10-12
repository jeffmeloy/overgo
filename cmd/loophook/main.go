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
// stop_hook_active retry passes, a live background gate steps aside, an active
// scoped operator stop passes, plan-complete passes.
package main

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/guard"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/repoanalysis"
)

// turnBaseFile is the process-state file that snapshots dirt at
// UserPromptSubmit so the Stop gate blocks only on TURN-CREATED dirt.
// Pre-existing dirt is another lane's parked in-flight work. Each session has
// its own, named by the session the hook is called for: sessions that overlap
// in one checkout each take a snapshot when their turn begins, and one file
// for the checkout had every prompt overwrite the other session's, so a turn
// was judged against a snapshot it did not take. Empty when the state
// directory cannot be made: no snapshot, so the gate reads none.
func turnBaseFile(session string) string {
	path, _ := processlock.StateFile(".", guard.SessionState(session), "docs/.loop_state")
	return path
}

// hookInput is what the harness hands a hook, as far as the loop reads it:
// the session the hook is called for, and the prompt of a prompt hook.
type hookInput struct {
	Session string `json:"session_id"`
	Prompt  string `json:"prompt"`
}

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
		runDoctrine(readStdin())
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

// parseHook reads the harness's input; text that is not the input it sends
// names no session, and its hooks share the one snapshot sessions had.
func parseHook(hookJSON string) (input hookInput) {
	_ = json.Unmarshal([]byte(hookJSON), &input)
	return input
}

// Stop verdict reasons. Allow whenever a valve fires; otherwise a turn-end is
// refused for one of two reasons: it would orphan turn-created work, or it
// would pause the loop while the plan still holds an open row (owner rule
// 2026-08-20: "the need to continue the plan doesn't go away unless I say
// stop"). Committed progress is necessary, never sufficient: a turn under an
// open plan row continues until the plan is EMPTY, and the only sanctioned
// pause is a recorded user stop (go run ./cmd/plan -stop), which arms the
// operator-stop decision.
const (
	stopAllow    = ""
	stopOrphan   = "orphan"
	stopContinue = "continue"
)

// stopDecision is the pure verdict the Stop hook renders. planComplete is the
// only plan-derived allow; a landed commit changes the message, never the
// verdict.
func stopDecision(stopHookActive, operatorStopped, gateRunning, planComplete, dirtyWork bool) string {
	if stopHookActive || operatorStopped || gateRunning || planComplete {
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
	session := parseHook(hookJSON).Session
	// A review session owns no row and edits nothing tracked (the guard
	// refuses it), so it is held to no row and charged with no file, and
	// reads no dispatch. Neither does a session in a worktree another
	// implementing session holds.
	if base, ok := readTurnBase(session); ok && (base.Review || base.Holder != "") {
		return 0
	}
	dispatch := nextDispatch()
	operatorStopped := dispatch.Stop != nil && dispatch.Stop.Blocked
	dirtyWork := hasTurnCreatedDirt(session)
	verdict := stopDecision(false, operatorStopped, gateCommitting(), dispatch.Complete, dirtyWork)
	// A turn that ends with nothing left for the session to do here gives the
	// worktree back to the next implementing session; a turn a valve lets end,
	// or one that leaves work uncommitted, keeps it.
	rested := (operatorStopped || dispatch.Complete || dispatch.Waiting != "") && !dirtyWork
	// A session whose role has no row to work is not held to the loop, but it
	// is still held to the work its own turn left uncommitted: it can write code.
	if dispatch.Waiting != "" {
		fmt.Fprintln(os.Stderr, dispatch.Line)
		if verdict != stopOrphan {
			if rested {
				releaseWorktree(session)
			}
			return 0
		}
	}
	if verdict == stopAllow {
		if rested {
			releaseWorktree(session)
		}
		return 0
	}
	fmt.Fprint(os.Stderr, stopMessage(verdict, headAdvancedSinceTurnStart(session), dispatch))
	return 2
}

// gateCommitting reports a gate committing now; a test of the stop gate, which
// the gate itself runs, says none is.
var gateCommitting = gateProcessRunning

// stopMessage renders the refusal for a verdict, naming the dispatched row
// from the dispatch's own fields.
func stopMessage(verdict string, progressed bool, dispatch plan.Dispatch) string {
	next := dispatch.Line
	if len(next) > 150 {
		next = next[:150]
	}
	var b strings.Builder
	switch verdict {
	case stopOrphan:
		b.WriteString("overgo loop gate: this turn would orphan uncommitted work.\n")
		b.WriteString("  * land it via 'go run ./cmd/loop -land <item>/<step> -message-file -' or revert it.\n")
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
	return b.String()
}

// runPostCommit re-baselines the turn-dirt snapshot at a gate commit boundary
// so the orphan check measures only work created after the commit. The old
// dispatch-marker lifecycle is retired: cmd/loop owns continuity.
func runPostCommit(hookJSON string) {
	if strings.Contains(hookJSON, "cmd/gate") {
		session := parseHook(hookJSON).Session
		base, _ := readTurnBase(session)
		writeTurnBase(session, base.Review, base.Holder)
	}
}

// takeoverDeclaration opens the message of a session that takes a worktree
// from the session that holds it, when that session is gone or its work here
// is this one's to continue.
const takeoverDeclaration = "take over this worktree"

// stateFileMode is the mode of a file the hooks keep in a worktree's process
// state: the session's own, and its owner's to read.
const stateFileMode = 0o600

// holdWorktree takes the worktree for an implementing session as its turn
// begins, and returns the session that holds it when that is another. The
// first session to begin a turn here holds it, a session that holds it keeps
// it, and one that declares a takeover takes it. A session the harness does
// not name takes none.
func holdWorktree(session string, takeover bool) (holder string) {
	path, err := filepath.Join(processlock.StateDirectory, guard.WorktreeHolder), processlock.EnsureStateDirectory(".")
	if session == "" || err != nil {
		return ""
	}
	if takeover {
		_ = os.WriteFile(path, []byte(session), stateFileMode)
		return ""
	}
	if file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, stateFileMode); err == nil {
		_, _ = file.WriteString(session)
		_ = file.Close()
		return ""
	}
	held, _ := os.ReadFile(path)
	// A file another session has created and not yet written names it no less.
	if holder = cmp.Or(strings.TrimSpace(string(held)), "that began as this one did"); holder == session {
		return ""
	}
	return holder
}

// releaseWorktree gives the worktree back when the session that holds it
// ends a turn with nothing left to do here.
func releaseWorktree(session string) {
	path, err := filepath.Join(processlock.StateDirectory, guard.WorktreeHolder), processlock.EnsureStateDirectory(".")
	if held, readErr := os.ReadFile(path); err == nil && readErr == nil && session != "" && strings.TrimSpace(string(held)) == session {
		_ = os.Remove(path)
	}
}

// heldNotice is what a session is told, in place of the lane's dispatched
// row, in a worktree another implementing session holds.
func heldNotice(holder string) string {
	return "OVERGO WORKTREE HELD. Session " + holder + " holds this worktree. A commit takes the checkout's whole tree, so two implementing sessions here would commit each other's files. " +
		"This session lands nothing here and edits no file the repository tracks (the guard refuses both); it is dispatched no row, held to none, and asked again at its next message. " +
		"Work in a worktree of its own: git worktree add --detach <directory> master, and open the session there with the data roots in its environment, which a worktree does not carry. " +
		"Its commits take the same command and land on the trunk through the repository's one queue: each waits its turn, is brought up to the trunk and is checked on it. " +
		"If that session is gone and its work here is this one's to continue, begin a message with \"" + takeoverDeclaration + "\"."
}

// reviewDeclaration opens the first message of a session that only reviews.
const reviewDeclaration = "review session"

// reviewNotice is what a review session is told when it starts and at each
// prompt, in place of the lane's dispatched row.
const reviewNotice = "OVERGO REVIEW SESSION. This session reviews: it lands nothing and edits no file the repository tracks (the guard refuses both), it is dispatched no row, and the stop gate holds it to none. Report what it finds with go run ./cmd/finding; scratch belongs in a path the repository ignores."

func runDoctrine(hookJSON string) {
	session := parseHook(hookJSON).Session
	// A session that declared itself a review session stays one when it is
	// resumed or compacted, and is told so in place of the turn contract.
	base, ok := readTurnBase(session)
	if ok && base.Review {
		fmt.Println(reviewNotice)
		return
	}
	// A session sent away from a worktree stays sent away until its next
	// message asks again.
	if ok && base.Holder != "" {
		fmt.Println(heldNotice(base.Holder))
		return
	}
	// A turn never spans sessions -- any pending boundary is stale.
	_ = os.Remove(turnBaseFile(session))
	fmt.Println(doctrineText)
	fmt.Println()
	fmt.Println("Dispatched now (go run ./cmd/plan -next):")
	fmt.Println(nextDispatch().Line)
	if root, err := os.Getwd(); err == nil {
		texts := memoryTexts(root)
		texts["doctrine"] = doctrineText
		if dead := deadReferences(root, texts); len(dead) != 0 {
			fmt.Println()
			fmt.Println("Dead references (the prose names what the repository no longer holds; fix the prose):")
			for _, reference := range dead {
				fmt.Println("  " + reference)
			}
		}
	}
}

// beginTurn snapshots the dirty set as a turn starts, for the session the
// hook is called for: the Stop gate then owes a block only to dirt this turn
// creates, not to another session's or another lane's parked work. It reports
// whether the session is a review session: one is declared by the first
// message of a session the harness names, opening with reviewDeclaration,
// and stays one. Any other session is an implementing session and takes the
// worktree, or is recorded as sent away by the session that holds it.
func beginTurn(input hookInput) (review bool) {
	base, known := readTurnBase(input.Session)
	opening := strings.ToLower(strings.TrimSpace(input.Prompt))
	review = base.Review || !known && input.Session != "" && strings.HasPrefix(opening, reviewDeclaration)
	holder := ""
	if !review {
		holder = holdWorktree(input.Session, strings.HasPrefix(opening, takeoverDeclaration))
	}
	writeTurnBase(input.Session, review, holder)
	return review
}

func runPrompt(hookJSON string) {
	input := parseHook(hookJSON)
	if beginTurn(input) {
		fmt.Println(reviewNotice)
		return
	}
	if base, _ := readTurnBase(input.Session); base.Holder != "" {
		fmt.Println(heldNotice(base.Holder))
		return
	}
	if boundedRequestText(input.Prompt) {
		fmt.Println("OVERGO BOUNDED REQUEST: answer this request concisely, THEN continue the dispatched plan row in this same turn -- a prompt never pauses the loop (owner rule). Only an explicit user stop pauses; record it with go run ./cmd/plan -stop user-stop:<detail>.")
		return
	}
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
	OriginalPath   string `json:"original_path,omitzero"`
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

// turnBase is a session's turn-start snapshot: the dirty set (orphan check)
// and the HEAD commit (loop-progress check), and what the guard reads too:
// whether the session declared itself a review session, and the session that
// holds this worktree when another implementing session does.
type turnBase struct {
	Head   string      `json:"head"`
	Facts  []dirtyFact `json:"facts"`
	Review bool        `json:"review,omitzero"`
	Holder string      `json:"holder,omitzero"`
}

// writeTurnBase snapshots dirt and HEAD for the session at a prompt or commit
// boundary, and keeps whether it is a review session and who holds the
// worktree in its place.
func writeTurnBase(session string, review bool, holder string) {
	facts, err := currentDirtyFacts()
	if err != nil {
		return
	}
	data, err := json.Marshal(turnBase{Head: gitHead(), Facts: facts, Review: review, Holder: holder})
	if err == nil {
		_ = os.WriteFile(turnBaseFile(session), append(data, '\n'), stateFileMode)
	}
}

// readTurnBase parses the session's snapshot; ok=false on a missing or legacy file.
func readTurnBase(session string) (turnBase, bool) {
	raw, err := os.ReadFile(turnBaseFile(session))
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
func headAdvancedSinceTurnStart(session string) bool {
	base, ok := readTurnBase(session)
	if !ok {
		return true
	}
	return gitHead() != base.Head
}

// hasTurnCreatedDirt reports whether dirt exists beyond the turn-start snapshot.
// A missing or unreadable snapshot FAILS SAFE toward the old behavior -- all
// dirt blocks -- so a lost marker errs toward "commit your work", never toward
// orphaning it.
func hasTurnCreatedDirt(session string) bool {
	current, currentErr := currentDirtyFacts()
	if currentErr != nil {
		return true
	}
	base, ok := readTurnBase(session)
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

// nextDispatch resolves the dispatched row through the plan's own resolver,
// which reuses the dispatch cached for this HEAD, plan and store head; a
// resolver failure is reported in the line and never ends the loop.
func nextDispatch() plan.Dispatch {
	dispatch, err := plan.ResolveDispatch(context.Background(), ".", plan.DispatchRequest{Mode: plan.ExecutionInteractive})
	if err != nil {
		revision, sourcesChanged := buildRevision()
		return plan.Dispatch{Line: cmp.Or(staleBuildNotice(revision, sourcesChanged, err), "plan: "+err.Error())}
	}
	return dispatch
}

// buildRevision is the commit this binary was built at, and whether the
// tree's Go sources have changed since; an unstamped build reports neither.
func buildRevision() (string, bool) {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "", false
	}
	for _, setting := range info.Settings {
		if setting.Key == "vcs.revision" {
			_, err := gitauthority.Query(context.Background(), ".", "diff", "--quiet", setting.Value, "--", "internal", "cmd/loophook", "go.mod")
			changed := err != nil
			return setting.Value, changed
		}
	}
	return "", false
}

// staleBuildNotice names the rebuild when the plan could not be read by a
// hook built before the tree's sources changed: a stale hook mis-reads newer
// plan records, so its failure is its own, not the plan's.
func staleBuildNotice(revision string, sourcesChanged bool, err error) string {
	if revision == "" || !sourcesChanged {
		return ""
	}
	return fmt.Sprintf("loophook: this build (%.12s) predates the tree's sources and cannot read the current plan (%v); rebuild it: go build -o bin/loophook.exe ./cmd/loophook", revision, err)
}
