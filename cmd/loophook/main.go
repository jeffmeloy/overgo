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
// pause is a recorded stop. Valves keep it from ever wedging: a
// stop_hook_active retry with nothing changed since its block passes, a live
// background gate steps aside, an active scoped recorded stop passes,
// plan-complete passes.
package main

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/guard"
	"overgo/internal/landing"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
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
	// ToolInput is a tool hook's call, as far as the loop reads it: the command a Bash call ran.
	ToolInput struct {
		Command string `json:"command"`
	} `json:"tool_input"`
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
// would pause the loop while the plan still holds an open row (the system's
// doctrine: the plan's work continues until a recorded stop). Committed
// progress is necessary, never sufficient: a turn under an
// open plan row continues until the plan is EMPTY, and the only sanctioned
// pause is a recorded stop (go run ./cmd/plan -stop), which arms the
// recorded-stop decision.
const (
	stopAllow    = ""
	stopOrphan   = "orphan"
	stopContinue = "continue"
)

// stopDecision is the pure verdict the Stop hook renders. planComplete is the
// only plan-derived allow; a landed commit changes the message, never the
// verdict.
func stopDecision(operatorStopped, gateRunning, planComplete, dirtyWork, typedTurn bool) string {
	if operatorStopped || gateRunning || planComplete {
		return stopAllow
	}
	if dirtyWork {
		return stopOrphan
	}
	// A turn a typed message began ends once the session has answered it: whoever typed it is present and decides
	// what follows, and the plan is not paused by it.
	if typedTurn {
		return stopAllow
	}
	return stopContinue
}

func runStop(hookJSON string) int {
	session := parseHook(hookJSON).Session
	// Re-attempt valve: a retry passes while the session is as its block found it (stuckSinceBlock).
	retry := strings.Contains(hookJSON, `"stop_hook_active":true`) || strings.Contains(hookJSON, `"stop_hook_active": true`)
	if retry && stuckSinceBlock(session) {
		// The valve keeps the gate from wedging a session; the block it passed is kept for the next prompt to report.
		if err := recordPassedBlock(session); err != nil {
			fmt.Fprintf(os.Stderr, "stop: the block this stop passed could not be kept for the next prompt: %v\n", err)
		}
		return 0
	}
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
	base, _ := readTurnBase(session)
	verdict := stopDecision(operatorStopped, gateCommitting(), dispatch.Complete, dirtyWork, base.Typed)
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
	if run := measurementRun(); run != "" {
		fmt.Fprintln(os.Stderr, "  "+run)
	}
	// A block no retry can read back would hold the session for good: it is let through instead.
	if err := recordStopBlock(session, verdict+": "+dispatch.Line); err != nil {
		fmt.Fprintf(os.Stderr, "stop: this block could not be kept, so the stop passes: %v\n", err)
		return 0
	}
	return 2
}

// stopBlock is what the Stop gate records when it blocks a session's turn-end: the HEAD and the dirty set it found. The
// harness marks every stop of the continuation a block began stop_hook_active, so a valve that passed each such retry
// let one block buy an unchecked stop at any later point: on 2026-10-30 a stop 34 minutes and two landings after a block
// passed without a word, and the row sat idle 1h44m. A retry passes only while both are as recorded, the session
// stopping again with nothing done since, stuck rather than resting; work since the block is judged again.
type stopBlock struct {
	Head  string      `json:"head"`
	Facts []dirtyFact `json:"facts"`
	// Reason is the verdict and the dispatched row the block held the session to, which a stop the valve passes reports.
	Reason string `json:"reason"`
}

// stopBlockFile is the session's block record, beside its turn-start snapshot; empty when the state directory cannot
// be made.
func stopBlockFile(session string) string {
	if processlock.EnsureStateDirectory(".") != nil {
		return ""
	}
	return filepath.Join(processlock.StateDirectory, guard.SessionState(session)+"-blocked")
}

// recordStopBlock records the HEAD and the dirty set a block found, with its reason, and says when it could not.
func recordStopBlock(session, reason string) error {
	facts, err := currentDirtyFacts()
	if err != nil {
		return err
	}
	path := stopBlockFile(session)
	if path == "" {
		return errors.New("no state directory")
	}
	data, err := json.Marshal(stopBlock{Head: gitHead(), Facts: facts, Reason: reason})
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), stateFileMode)
}

// stuckSinceBlock reports whether a retry finds the session as its last block did. A session whose state cannot be
// read or kept is let through, so the gate never wedges it; a retry with no block recorded is judged, and its block is
// recorded for the next.
func stuckSinceBlock(session string) bool {
	path := stopBlockFile(session)
	facts, err := currentDirtyFacts()
	if path == "" || err != nil {
		return true
	}
	raw, err := os.ReadFile(path)
	var block stopBlock
	if err != nil || json.Unmarshal(raw, &block) != nil {
		return false
	}
	return block.Head == gitHead() && slices.Equal(block.Facts, facts)
}

// passedBlockFile is where a stop the valve passed keeps the block it passed, for the session's next prompt.
func passedBlockFile(session string) string { return stopBlockFile(session) + "-passed" }

// recordPassedBlock keeps the block a stop passed through the valve for the next prompt to report: the turn ended with
// it unresolved. A later stop finds no block standing and is judged again.
func recordPassedBlock(session string) error {
	return os.Rename(stopBlockFile(session), passedBlockFile(session))
}

// takePassedBlock returns, once, the reason of a block an earlier stop of the session passed through the valve.
func takePassedBlock(session string) string {
	path := passedBlockFile(session)
	raw, err := os.ReadFile(path)
	var block stopBlock
	if err != nil || os.Remove(path) != nil || json.Unmarshal(raw, &block) != nil {
		return ""
	}
	return block.Reason
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
			b.WriteString("overgo loop gate: the row landed, and the plan holds open rows.\n")
		} else {
			b.WriteString("overgo loop gate: this turn began with no typed message, and the plan holds an open row.\n")
		}
		b.WriteString("  * continue with the dispatched row.\n")
	}
	b.WriteString("  Dispatched now: " + next + "\n")
	return b.String()
}

// runPostCommit re-baselines the turn-dirt snapshot at a gate commit boundary
// so the orphan check measures only work created after the commit. The old
// dispatch-marker lifecycle is retired: cmd/loop owns continuity.
func runPostCommit(hookJSON string) {
	input := parseHook(hookJSON)
	// A session that lands here implements: it is recorded so, and takes the worktree's hold when it is free.
	if guard.Landing(input.ToolInput.Command) {
		recordImplementer(input.Session)
		if base, known := readTurnBase(input.Session); known && !base.Review {
			base.Holder = holdWorktree(input.Session, false)
			writeTurnBase(input.Session, base)
		}
	}
	if strings.Contains(hookJSON, "cmd/gate") {
		base, _ := readTurnBase(input.Session)
		writeTurnBase(input.Session, base)
	}
}

// implementersFile lists, in the worktree's process state, the sessions that have landed here, one a line: the
// sessions whose turns take the worktree's hold.
const implementersFile = "implementers"

// implementer reports whether the session has landed here.
func implementer(session string) bool {
	listed, err := os.ReadFile(filepath.Join(processlock.StateDirectory, implementersFile))
	return err == nil && session != "" && slices.Contains(strings.Fields(string(listed)), session)
}

// recordImplementer records the session as one that has landed here.
func recordImplementer(session string) {
	if session == "" || strings.ContainsAny(session, " \t\r\n") || implementer(session) || processlock.EnsureStateDirectory(".") != nil {
		return
	}
	file, err := os.OpenFile(filepath.Join(processlock.StateDirectory, implementersFile), os.O_WRONLY|os.O_CREATE|os.O_APPEND, stateFileMode)
	if err == nil {
		_, _ = file.WriteString(session + "\n")
		_ = file.Close()
	}
}

// takeoverDeclaration opens the message of a session that takes a worktree from the session that holds it.
const takeoverDeclaration = guard.TakeoverDeclaration

// stateFileMode is the mode of a file the hooks keep in a worktree's process
// state: the session's own, readable by the account that runs it alone.
const stateFileMode = 0o600

// holdWorktree takes the worktree for an implementing session as its turn
// begins, and returns the session that holds it when that is another. A
// session that has landed here (implementer) takes it when it is free, a
// session that holds it keeps it, and one that declares a takeover takes it.
// Any other session takes none: it works beside no holder, or is told who
// holds it. A review session that had not said so took the hold twice on
// 2026-10-30 while the lead was between turns. A session the harness does not
// name takes none.
func holdWorktree(session string, takeover bool) (holder string) {
	path, err := filepath.Join(processlock.StateDirectory, guard.WorktreeHolder), processlock.EnsureStateDirectory(".")
	if session == "" || err != nil {
		return ""
	}
	own := guard.HoldText(session, sessionProcess())
	if takeover {
		_ = os.WriteFile(path, []byte(own), stateFileMode)
		return ""
	}
	// A hold whose session's process has exited names a session that has ended, a worker the driver ran or a session
	// closed or cut off, and is free: on 2026-11-07 two such holds refused the lead's every edit until it was cleared.
	if raw, err := os.ReadFile(path); err == nil {
		if held, process := guard.ReadHold(raw); held != session && process > 0 && !processcontrol.ProcessAlive(process) {
			_ = os.Remove(path)
		}
	}
	if !implementer(session) {
		if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
			return ""
		}
	} else if file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, stateFileMode); err == nil {
		_, _ = file.WriteString(own)
		_ = file.Close()
		return ""
	}
	raw, _ := os.ReadFile(path)
	held, process := guard.ReadHold(raw)
	// A file another session has created and not yet written names it no less.
	if holder = cmp.Or(held, "that began as this one did"); holder == session {
		// The session that holds it keeps its process current, across a restart of its Claude Code.
		if process != sessionProcess() {
			_ = os.WriteFile(path, []byte(own), stateFileMode)
		}
		return ""
	}
	return holder
}

// claudeProcessVariable names the process Claude Code runs as, which it sets for the hooks it starts.
const claudeProcessVariable = "CLAUDE_PID"

// sessionProcess is the process the hook's Claude Code session runs as, zero where none is named.
func sessionProcess() int {
	process, _ := strconv.Atoi(os.Getenv(claudeProcessVariable))
	return process
}

// releaseWorktree gives the worktree back when the session that holds it
// ends a turn with nothing left to do here.
func releaseWorktree(session string) {
	path, err := filepath.Join(processlock.StateDirectory, guard.WorktreeHolder), processlock.EnsureStateDirectory(".")
	if raw, readErr := os.ReadFile(path); err == nil && readErr == nil && session != "" {
		if held, _ := guard.ReadHold(raw); held != session {
			return
		}
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
	// The day's delivery, read from Git and the gate's run records, so the work's cost and progress are in sight without
	// being asked for (review-fixes/status-measures); a checkout it cannot be read in says nothing of it.
	if delivery, err := landing.DayDelivery(context.Background(), ".", time.Now(), landing.ReportFiling); err == nil {
		fmt.Println(delivery)
	}
	if run := measurementRun(); run != "" {
		fmt.Println()
		fmt.Println(run)
	}
	if root, err := os.Getwd(); err == nil {
		texts := memoryTexts(root)
		if beyond := repoanalysis.MemoryBeyondPointer(texts); len(beyond) != 0 {
			fmt.Println()
			fmt.Println("Session memory beyond its pointer (rules belong in skill.md or code; sort these there):")
			for _, entry := range beyond {
				fmt.Println("  " + entry)
			}
		}
		texts["doctrine"] = doctrineText
		if dead := repoanalysis.DeadReferences(root, texts); len(dead) != 0 {
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
	writeTurnBase(input.Session, turnBase{Review: review, Holder: holder, Typed: typedPrompt(input.Prompt)})
	return review
}

// measurementRun is the newest measurement run's record as the hooks print it,
// or nothing where none is kept: a run's end and times are read from the record
// the run keeps of itself (plan.MeasurementRun), never reckoned.
func measurementRun() string {
	var lines []string
	if run, kept := plan.ReadMeasurementRun("."); kept {
		lines = append(lines, "OVERGO MEASUREMENT RUN: "+run.Report()+".")
	}
	// A session's own run holding the card is why dispatch passes over the steps that need it (plan.CardHold).
	if hold, held := plan.ReadCardHold("."); held && hold.HoldsTheCard() {
		lines = append(lines, fmt.Sprintf("OVERGO CARD HELD: process %d holds the card for %s, since %s; dispatch passes over card steps while it lives.",
			hold.Process, hold.Purpose, hold.Started.Format(time.DateTime)))
	}
	return strings.Join(lines, "\n")
}

func runPrompt(hookJSON string) {
	input := parseHook(hookJSON)
	// A requested stop rests on the latest prompt asking for one (plan.RequireStopRequest).
	if err := plan.NoteStopRequest(".", input.Prompt); err != nil {
		fmt.Printf("OVERGO: this prompt's request to stop could not be kept: %v\n", err)
	}
	if reason := takePassedBlock(input.Session); reason != "" {
		fmt.Printf("OVERGO STOP PASSED: the last turn ended through the Stop gate's retry valve with its block unresolved (%s); continue that work.\n", reason)
	}
	if beginTurn(input) {
		fmt.Println(reviewNotice)
		return
	}
	// A session sent away is told so by the guard's refusal of its work, not at every message.
	if base, _ := readTurnBase(input.Session); base.Holder != "" {
		return
	}
	if run := measurementRun(); run != "" {
		fmt.Println(run)
	}
	command := exec.Command("go", "run", "./cmd/plan", "-prompt")
	command.Stdout, command.Stderr = os.Stdout, os.Stderr
	_ = command.Run()
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
	// Typed says the turn began with a typed message, not a task's notification or a scheduled wake.
	Typed bool `json:"typed,omitzero"`
}

// The openings of a prompt no person typed: a background task's notification and a scheduled wake of the loop.
var machineOpenings = []string{"<task-notification>", "<<autonomous-loop"}

// typedPrompt says a prompt is a typed message: it holds words and opens as no notification or wake does.
func typedPrompt(prompt string) bool {
	opening := strings.TrimSpace(prompt)
	return opening != "" && !slices.ContainsFunc(machineOpenings, func(machine string) bool { return strings.HasPrefix(opening, machine) })
}

// writeTurnBase snapshots dirt and HEAD for the session at a prompt or commit
// boundary, beside base's account of the turn: whether it is a review
// session, who holds the worktree in its place, and whether a typed message began it.
func writeTurnBase(session string, base turnBase) {
	facts, err := currentDirtyFacts()
	if err != nil {
		return
	}
	base.Head, base.Facts = gitHead(), facts
	data, err := json.Marshal(base)
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
// and SELF-HEALING: the very next attempt carries stop_hook_active, whose valve,
// with nothing changed since the block, allows (stuckSinceBlock). No wedge was
// ever observed.
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
