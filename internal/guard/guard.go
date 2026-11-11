// Package guard is the PreToolUse safety engine: it judges one harness tool
// call and denies commands that could destroy data or bypass the commit gate.
// Accident prevention, not an adversary barrier — the covered shapes are the
// ones an agent plausibly emits without evasive intent (direct forms, split
// flags, quoted targets, -c wrappers, eval, xargs/-exec). Enforcement that
// survives variable indirection lives in filesystem ACLs, not here.
//
// Ported from adaptive_new (incident lineage: 300 GB force-delete, 170 GB
// worktree-junction delete, backtick command substitution eating commit
// prose, heredoc backslash mangling); rule set current through the
// quoted-target and executable-substitution hardening (ADV-88/89).
package guard

import (
	"cmp"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
)

// deleteVerbs are the programs that delete what they name.
var deleteVerbs = []string{"del", "rd", "remove-item", "rm", "rmdir"}

// protectedVerbs are the programs that may not touch a protected data path.
var protectedVerbs = append([]string{"attrib", "chmod", "icacls", "move", "move-item", "mv", "shred", "takeown", "unlink"}, deleteVerbs...)

// findDeleteVerbs are the programs find's -exec deletes by.
var findDeleteVerbs = []string{"chmod", "del", "mv", "rm", "rmdir", "shred", "unlink"}

// protectedPaths are the data paths nothing deletes or moves, forward slashes for both separators.
var protectedPaths = []string{"checkpoints/", "datasets/", "docs/repodb", "experiments/", "models/", "overgodb-store", "repodb-store"}

// referenceTree is the tree ports are read from and compared with, beside the checkout: no session writes into it.
const referenceTree = "adaptive_new"

// creatingVerbs are the programs that write every path they name, besides those that delete or move.
var creatingVerbs = []string{"mkdir", "tee", "touch", "truncate"}

// copyVerbs are the programs that write only the last path they name, reading the others.
var copyVerbs = []string{"copy", "copy-item", "cp"}

// shellNames are the programs -c hands a command to.
var shellNames = []string{"bash", "dash", "ksh", "mksh", "sh", "zsh"}

// sourceExtensions name the files a heredoc must not write.
var sourceExtensions = []string{"cu", "cuh", "go", "json", "jsonl", "md", "py", "sh", "yaml"}

// payloadDepth is how many wrappers deep a command is read.
const payloadDepth = 4

// ToolCall is the harness PreToolUse payload subset the guard judges: the
// session the call is made in and the directory it works in, a shell tool's
// command, and the file an editing tool writes.
type ToolCall struct {
	ToolName  string `json:"tool_name"`
	Session   string `json:"session_id"`
	Directory string `json:"cwd"`
	ToolInput struct {
		Command  string `json:"command"`
		Path     string `json:"file_path"`
		Notebook string `json:"notebook_path"`
	} `json:"tool_input"`
}

// WorktreeHolder is the file, in a worktree's process state, that names the
// implementing session that holds the worktree. A landing takes the
// checkout's whole tree as its candidate, so two implementing sessions in one
// worktree would land each other's files; the second is sent to a worktree of
// its own. A hook is a short process and can keep no lock of the system's
// open for a session, so the file is the holding, and its creation is atomic.
const WorktreeHolder = "implementing_session"

// TakeoverDeclaration opens, in lower case, the message of a session that takes a worktree from the session that
// holds it, when that session is gone or its work here is this one's to continue. No worker prompt opens with it.
const TakeoverDeclaration = "take over this worktree"

// HoldText is the hold file's content for session: the session, and on the next line the process its Claude Code
// runs as, by which a later session knows whether the holder still runs.
func HoldText(session string, process int) string {
	return session + "\n" + strconv.Itoa(process)
}

// ReadHold reads a hold file's content: the holding session and its process, zero where the hold names none, as a
// hold written before the process was recorded does.
func ReadHold(raw []byte) (session string, process int) {
	first, rest, _ := strings.Cut(strings.TrimSpace(string(raw)), "\n")
	process, _ = strconv.Atoi(strings.TrimSpace(rest))
	return strings.TrimSpace(first), process
}

// SessionState names the file, in a checkout's process state, where the
// loop's hooks keep what they know of one session: the snapshot its turn
// began with, and whether it declared itself a review session. A session the
// harness does not name, or whose name is not one path element, has the one
// file all sessions shared before they were told apart.
func SessionState(session string) string {
	if session == "" || filepath.Base(session) != session {
		return "loop_state"
	}
	return "loop_state-" + session
}

// sessionEnd names the file, beside a session's state, that records the session ended.
func sessionEnd(session string) string { return SessionState(session) + "-ended" }

// RecordSessionEnd records, in the process state of the checkout at root, that session ended: the session-end hook
// does, and so does cmd/loop -leftovers -end-session for a session that ended before the hook was there. A session not
// named by one path element has no state of its own, so no end is recorded for it.
func RecordSessionEnd(root, session string) error {
	if SessionState(session) == SessionState("") {
		return errors.New("guard: a session's end is recorded only for a session named by one path element")
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(root, processlock.StateDirectory, sessionEnd(session)), []byte(time.Now().UTC().Format(time.RFC3339)+"\n"), 0o600)
}

// SessionEnded reports whether session has ended, by a positive sign alone: its end recorded, or the checkout's hold
// naming it and a process that has exited. A session with neither may be running, a review session among them, which
// holds nothing, so it is never taken to have ended; a folder's age is no sign.
func SessionEnded(root, session string) bool {
	state := filepath.Join(root, processlock.StateDirectory)
	if _, err := os.Stat(filepath.Join(state, sessionEnd(session))); err == nil {
		return true
	}
	raw, err := os.ReadFile(filepath.Join(state, WorktreeHolder))
	if err != nil {
		return false
	}
	held, process := ReadHold(raw)
	return held == session && process > 0 && !processcontrol.ProcessAlive(process)
}

// Landing reports whether a command lands a change through the gate: what makes a session an implementing one, whose
// turns hold its worktree.
func Landing(command string) bool {
	for _, payload := range payloads(command, 0) {
		if slices.ContainsFunc(readShell(payload).programs(), lands) {
			return true
		}
	}
	return false
}

// lands says a command runs the gate, or the loop's landing.
func lands(run shellCommand) bool {
	return run.names("./cmd/gate") || run.names("./cmd/loop") && run.holds("-land", "--land")
}

// removes says a command removes leftovers, or withdraws from the store the locations of a run's checkpoints; the
// listing alone removes nothing.
func removes(run shellCommand) bool {
	return run.names("./cmd/loop") && run.holds("-delete", "--delete", "-withdraw", "--withdraw")
}

// removal is any command of a line that removes.
func removal(command string) bool {
	return slices.ContainsFunc(readShell(command).programs(), removes)
}

// reviewVerdict refuses what a session that changes nothing here may not do,
// and allows the rest. A session that declared itself a review session (its
// state says so) lands nothing, removes no leftover and writes no file the
// repository tracks or would track, so it has no work of its own for the stop
// gate to hold it to, and nothing here that is its own to remove.
// Neither does any session in a checkout another session holds, whether its
// state named the holder as its turn began or the hold names another session
// that still runs (heldBy): a landing takes the checkout's changes, so its
// files would join the holder's commit, and on 2026-11-08 a document another
// session wrote into the lead's checkout stopped every landing. It is sent to
// a worktree of its own. What a call writes is the file a Write or Edit names
// and those a shell command writes (written); a session working elsewhere is
// refused a write into the trunk's checkout while another session holds it.
// A file outside the checkout, or one the repository ignores, is scratch and
// is any session's to write.
func reviewVerdict(call ToolCall, root string) string {
	if call.Session == "" {
		return ""
	}
	var state struct {
		Review bool
		Holder string
	}
	if data, err := os.ReadFile(filepath.Join(root, processlock.StateDirectory, SessionState(call.Session))); err == nil && json.Unmarshal(data, &state) != nil {
		state.Review, state.Holder = false, ""
	}
	state.Holder = cmp.Or(state.Holder, heldBy(root, call.Session))
	session, instead := "a review session", "report what it found (go run ./cmd/finding)"
	if !state.Review {
		session, instead = "a session in a worktree session "+state.Holder+" holds", "work in a worktree of its own (git worktree add)"
	}
	kept := state.Review || state.Holder != ""
	if kept && Landing(call.ToolInput.Command) {
		return session + " lands nothing here; " + instead
	}
	if kept && removal(call.ToolInput.Command) {
		return session + " removes nothing here; " + instead
	}
	trunk := sync.OnceValue(func() string { return trunkCheckout(root) })
	for _, path := range written(call, root) {
		if relative, inside := within(root, path); inside {
			if !kept {
				continue
			}
			if refusal := trackedWrite(root, relative, session, instead); refusal != "" {
				return refusal
			}
			continue
		}
		relative, inside := within(trunk(), path)
		if !inside || trunk() == root {
			continue
		}
		if holder := heldBy(trunk(), call.Session); holder != "" {
			if refusal := trackedWrite(trunk(), relative, "a session beside the checkout session "+holder+" holds", "work in a worktree of its own (git worktree add)"); refusal != "" {
				return refusal
			}
		}
	}
	return ""
}

// Verdict judges one tool call. Empty string allows; otherwise the deny
// reason. wrapped reports whether the denial came from an unwrapped payload.
func Verdict(call ToolCall, root string) (message string, wrapped bool) {
	if call.ToolName == "PowerShell" {
		return "the PowerShell tool is disabled; use the Bash tool", false
	}
	if message := reviewVerdict(call, root); message != "" {
		return message, false
	}
	if referencePath(cmp.Or(call.ToolInput.Path, call.ToolInput.Notebook)) {
		return referenceRefusal, false
	}
	for index, command := range payloads(call.ToolInput.Command, 0) {
		if m := ruleVerdict(command); m != "" {
			return m, index > 0
		}
	}
	return "", false
}

// payloads yields the command plus every shell -c / eval payload, recursively:
// a quoted string in -c position does not merely name a command, it IS one,
// and quote-stripping alone was a front door around every rule.
func payloads(command string, depth int) []string {
	out := []string{command}
	if depth >= payloadDepth {
		return out
	}
	for _, run := range readShell(command).programs() {
		for _, payload := range wrapped(run.words) {
			out = append(out, payloads(payload, depth+1)...)
		}
	}
	return out
}

// wrapped is what a command hands to a shell: the word after a shell's -c, and eval's arguments.
func wrapped(words []word) []string {
	var handed []string
	for index, current := range words {
		rest := words[index+1:]
		switch name := programName(current.text); {
		case name == "eval" && len(rest) > 0:
			texts := make([]string, len(rest))
			for at, next := range rest {
				texts[at] = next.text
			}
			handed = append(handed, strings.Join(texts, " "))
		case slices.Contains(shellNames, name):
			for at, next := range rest {
				if !strings.HasPrefix(next.text, "-") {
					break
				}
				if strings.Contains(next.text, "c") && !strings.HasPrefix(next.text, "--") && at+1 < len(rest) {
					handed = append(handed, rest[at+1].text)
					break
				}
			}
		}
	}
	return handed
}

func ruleVerdict(command string) string {
	parsed := readShell(command)
	programs := parsed.programs()
	any := func(rule func(shellCommand) bool) bool { return slices.ContainsFunc(programs, rule) }

	if any(invokesPowerShell) {
		return "PowerShell/pwsh invocation is disabled; use bash"
	}
	if any(func(run shellCommand) bool { return gitVerb(run, "clean") }) {
		return "git clean can delete gitignored model/dataset stores"
	}
	if any(forcesDelete) {
		return "force-delete flags are banned; never override read-only"
	}
	if any(touchesProtected) {
		return "delete/move/chmod under a protected data path (models/, datasets/, checkpoints/, experiments/, docs/repodb/, overgodb-store)"
	}
	if any(writesReference) {
		return referenceRefusal
	}
	if any(findDeletes) {
		return "find with -delete or -exec <delete-verb> over a protected data path; the delete target is the " +
			"substituted {} so no verb rule can see it -- ask the user"
	}
	if any(func(run shellCommand) bool { return gitVerb(run, "worktree", "remove") }) {
		return "git worktree remove recurses INTO junctioned model directories and has already destroyed 170 GB of " +
			"weights; retire the lane with 'git branch -d' and leave the directory. A helper copy whose commit is on the " +
			"trunk is cleared by 'go run ./cmd/loop -leftovers -delete helper-copy', which follows no link"
	}
	if any(func(run shellCommand) bool { return gitVerb(run, "commit") }) {
		return "raw git commit bypasses the small gate (format, vet, rules, tests, lanes, the row's verify); run " +
			"'go run ./cmd/loop -land <item>/<step> -message-file -' instead. --no-verify does not help: the gate is " +
			"the thing being skipped, not a hook"
	}
	if any(func(run shellCommand) bool { return gitVerb(run, "push") }) {
		return "a push publishes only on the owner's word and only fast-forwards origin's main: run " +
			"'go run ./cmd/loop -push <commit> -confirm-push', which goes through the credential manager"
	}
	if any(changesSecurity) {
		return "a session never changes the machine's security settings (Defender, the firewall, UAC): give the owner " +
			"the exact command and say what it unblocks"
	}
	if any(landsDirectly) {
		return "go run ./cmd/gate lands without the loop's confirmation from git and its refusal's class: land with " +
			"'go run ./cmd/loop -land <item>/<step> -message-file -', and preflight with the gate's -preflight"
	}
	if any(marksTrunkMove) {
		return "the trunk moves only by a landing: OVERGO_TRUNK_MOVE is the landing's own mark, and git's core.hooksPath is " +
			"where the trunk's hook lives; land with 'go run ./cmd/loop -land <item>/<step> -message-file -'"
	}
	if pipesLanding(programs) {
		return "a landing's verdict is its exit code, which a pipe reports as the last filter's: run the landing alone, " +
			"and read the commit it made with 'git log --oneline -1'"
	}
	if keepsLanding(programs) {
		return "a landing's output is not kept in a file: loop -land prints the lines that decide the outcome, and the " +
			"commit's 'Measured by the gate' line holds each phase's time; give the message on standard input with " +
			"-message-file -"
	}
	if heredocMangles(parsed, command) {
		return "a Bash heredoc EATS one level of backslash escaping and will write mangled content; use the Write or " +
			"Edit tool for anything containing a backslash"
	}
	if any(buildsCommandHere) {
		return "go build of one command without -o writes its binary into the working directory; at the repository " +
			"root that moves the root folder's time, and go test reruns every suite that stats it; build with -o bin/<name>.exe"
	}
	return ""
}

// referenceRefusal is why a write into the reference tree is refused.
const referenceRefusal = referenceTree + " is the reference tree, read to compare with or port from: nothing writes into it; write the port here"

// referencePath says a path lies in the reference tree: one of its segments, by either separator, names the tree.
func referencePath(path string) bool {
	return slices.Contains(strings.FieldsFunc(path, func(r rune) bool { return r == '/' || r == '\\' }), referenceTree)
}

// writesReference says a command writes into the reference tree (writes). Reading the tree is any session's.
func writesReference(run shellCommand) bool {
	return slices.ContainsFunc(writes(run), referencePath)
}

// writes are the paths a command writes: those it sends output to, and those a program that deletes, moves, creates,
// edits in place or copies onto what it names names, each as the shell reads it and as it was written. A copy writes
// only the last path it names.
func writes(run shellCommand) []string {
	targets := slices.Clone(run.sinks)
	if len(run.words) == 0 {
		return targets
	}
	program, named := programName(run.words[0].text), run.words[1:]
	switch {
	case slices.Contains(copyVerbs, program) && len(named) > 0:
		named = named[len(named)-1:]
	case program == "sed" && slices.ContainsFunc(named, func(current word) bool { return strings.HasPrefix(current.text, "-i") }):
	case !slices.Contains(protectedVerbs, program) && !slices.Contains(creatingVerbs, program):
		return targets
	}
	for _, current := range named {
		targets = append(targets, current.text, current.spelled)
	}
	return targets
}

// invokesPowerShell says a command names PowerShell outside a quotation.
func invokesPowerShell(run shellCommand) bool {
	return slices.ContainsFunc(run.words, func(current word) bool {
		if current.quoted {
			return false
		}
		return slices.ContainsFunc(wordsOf(strings.ToLower(current.text)), func(part string) bool { return part == "powershell" || part == "pwsh" })
	})
}

// wordsOf splits text at everything that is not a letter, a digit or an underscore.
func wordsOf(text string) []string {
	return strings.FieldsFunc(text, func(character rune) bool {
		return character != '_' && (character < '0' || character > '9' && character < 'A' || character > 'Z' && character < 'a' || character > 'z')
	})
}

// securityWords name what a command changing the machine's security settings touches: Defender and its services,
// the firewall, and UAC's policy key. securityPrograms change those settings, with the verbs that write them.
var (
	securityWords    = []string{"windefend", "wdnissvc", "securityhealthservice", "windowsdefender", "firewall", "policiessystem", "enablelua"}
	securityPrograms = map[string][]string{"reg": {"add", "delete", "import"}, "netsh": nil, "sc": {"config", "stop", "delete"}, "mpcmdrun": nil}
	// pathJoints are dropped before matching, so a key reads the same however the shell kept its separators.
	pathJoints = strings.NewReplacer(`\`, "", "/", "", " ", "")
)

// changesSecurity says a command changes the machine's security settings: Defender's own command, or reg, netsh or sc
// writing a key, rule or service that securityWords names.
func changesSecurity(run shellCommand) bool {
	words := run.invocation()
	if len(words) == 0 {
		return false
	}
	program := programName(words[0].text)
	verbs, known := securityPrograms[program]
	switch {
	case !known:
		return false
	case program == "mpcmdrun":
		return true
	}
	var text []string
	for _, current := range words[1:] {
		text = append(text, strings.ToLower(current.text))
	}
	joined := pathJoints.Replace(strings.Join(text, " "))
	writes := verbs == nil || len(text) != 0 && slices.Contains(verbs, text[0])
	return writes && slices.ContainsFunc(securityWords, func(word string) bool { return strings.Contains(joined, word) })
}

// gitVerb says a command is git running the given subcommand, past git's own options.
func gitVerb(run shellCommand, verb ...string) bool {
	words := run.invocation()
	if len(words) == 0 || programName(words[0].text) != "git" {
		return false
	}
	rest := skipFlags(words[1:])
	if len(rest) < len(verb) {
		return false
	}
	for index, want := range verb {
		if !strings.EqualFold(rest[index].text, want) {
			return false
		}
	}
	return true
}

// forcesDelete says a delete verb carries a force flag: a single-dash run of letters holding f, or a whole word that spells force.
func forcesDelete(run shellCommand) bool {
	words := run.invocation()
	if len(words) == 0 || !slices.Contains(deleteVerbs, programName(words[0].text)) {
		return false
	}
	return slices.ContainsFunc(words[1:], func(current word) bool {
		flag := strings.ToLower(current.text)
		if flag == "--force" {
			return true
		}
		letters, dashed := strings.CutPrefix(flag, "-")
		return dashed && strings.Contains(letters, "f") && len(wordsOf(letters)) == 1 && !strings.ContainsAny(letters, "-=/.0123456789_")
	})
}

// protectedPath says a word names a path under a protected data root, by either separator.
func protectedPath(text string) bool {
	normal := strings.ToLower(strings.ReplaceAll(text, `\`, "/"))
	return slices.ContainsFunc(protectedPaths, func(root string) bool { return strings.Contains(normal, root) })
}

// namesProtected says a word names a protected path as the shell reads it, or as it was written.
func namesProtected(current word) bool {
	return protectedPath(current.text) || protectedPath(current.spelled)
}

// touchesProtected says a program that deletes or moves names a protected path.
func touchesProtected(run shellCommand) bool {
	words := run.invocation()
	if len(words) == 0 || !slices.Contains(protectedVerbs, programName(words[0].text)) {
		return false
	}
	return slices.ContainsFunc(words[1:], namesProtected) || slices.ContainsFunc(run.sinks, protectedPath)
}

// findDeletes says find walks a protected path and deletes what it finds: the delete verb receives the substituted {},
// so the protected path lives in find's own arguments and no verb's rule can see it.
func findDeletes(run shellCommand) bool {
	words := run.invocation()
	if len(words) == 0 || programName(words[0].text) != "find" || !slices.ContainsFunc(words[1:], namesProtected) {
		return false
	}
	if run.holds("-delete") {
		return true
	}
	for _, runs := range runsFor(words) {
		if slices.Contains(findDeleteVerbs, programName(runs[0].text)) {
			return true
		}
	}
	return false
}

// harmlessSink says a redirection keeps nothing: the null device, or another stream.
func harmlessSink(target string) bool {
	return target == "/dev/null" || strings.HasPrefix(target, "&")
}

// landsDirectly says a command runs the gate to land rather than to preflight: the loop's landing runs the gate itself.
func landsDirectly(run shellCommand) bool {
	return run.names("./cmd/gate") && !run.holds("-preflight", "--preflight")
}

// marksTrunkMove says a command sets the landing's own mark for moving the trunk, or points git at hooks of its choosing.
func marksTrunkMove(run shellCommand) bool {
	return slices.ContainsFunc(run.words, func(current word) bool {
		lower := strings.ToLower(current.text)
		return strings.HasPrefix(lower, "overgo_trunk_move=") || strings.HasPrefix(lower, "core.hookspath")
	})
}

// pipesLanding says a landing's output is piped into another command, whose exit code the shell then reports.
func pipesLanding(programs []shellCommand) bool {
	for index, run := range programs {
		if run.names("./cmd/loop") && run.holds("-land", "--land") && index+1 < len(programs) && programs[index+1].piped {
			return true
		}
	}
	return false
}

// keepsLanding says a landing's output, or that of what it pipes into, goes to a file: a landing's goes to the terminal.
func keepsLanding(programs []shellCommand) bool {
	for index, run := range programs {
		if !run.names("./cmd/loop") && !run.names("./cmd/gate") {
			continue
		}
		for at := index; at < len(programs) && (at == index || programs[at].piped); at++ {
			following := programs[at]
			words := following.invocation()
			if !slices.Equal(following.sinks, nil) && !allHarmless(following.sinks) || at > index && len(words) > 0 && programName(words[0].text) == "tee" {
				return true
			}
		}
	}
	return false
}

func allHarmless(sinks []string) bool {
	return !slices.ContainsFunc(sinks, func(target string) bool { return !harmlessSink(target) })
}

// heredocMangles says a heredoc holding a backslash writes a source file.
func heredocMangles(parsed script, raw string) bool {
	if !slices.ContainsFunc(parsed.heredocs, func(body string) bool { return strings.Contains(body, `\`) }) {
		return false
	}
	return namesSourceFile(raw) && writesFile(raw)
}

// namesSourceFile says the text names a file of a source extension.
func namesSourceFile(raw string) bool {
	for _, token := range strings.FieldsFunc(strings.ToLower(raw), func(character rune) bool { return character != '.' && len(wordsOf(string(character))) == 0 }) {
		parts := strings.Split(token, ".")
		if slices.ContainsFunc(parts[1:], func(part string) bool { return slices.Contains(sourceExtensions, part) }) {
			return true
		}
	}
	return false
}

// writesFile says the text writes somewhere: a redirect, tee, sed -i, or a script's own open for writing.
func writesFile(raw string) bool {
	lower := strings.ToLower(raw)
	if strings.Contains(lower, "json.dump") || strings.Contains(lower, ".write(") || slices.Contains(wordsOf(lower), "tee") {
		return true
	}
	for rest, found := lower, true; found; {
		var opened string
		_, rest, found = strings.Cut(rest, ">")
		opened = strings.TrimLeft(rest, " \t\r\n")
		if found && opened != "" && opened[0] != '>' {
			return true
		}
	}
	for rest, found := lower, true; found; {
		_, rest, found = strings.Cut(rest, "open")
		arguments, opened := strings.CutPrefix(strings.TrimLeft(rest, " \t"), "(")
		if !found || !opened {
			continue
		}
		arguments, _, _ = strings.Cut(arguments, ")")
		if strings.Contains(arguments, "'w") || strings.Contains(arguments, `"w`) {
			return true
		}
	}
	fields := strings.Fields(lower)
	for index := 1; index < len(fields); index++ {
		if fields[index-1] == "sed" && strings.HasPrefix(fields[index], "-i") {
			return true
		}
	}
	return false
}

// goBuild is the two words a build begins with.
var goBuild = []string{"go", "build"}

// buildsCommandHere reports whether a command runs go build on exactly one ./cmd package without -o, which writes that
// command's binary into the working directory. Several packages, or a library, write nothing.
func buildsCommandHere(run shellCommand) bool {
	words := run.invocation()
	if len(words) < len(goBuild) || programName(words[0].text) != goBuild[0] || words[1].text != goBuild[1] {
		return false
	}
	var packages []string
	for _, current := range words[len(goBuild):] {
		switch {
		case current.text == "-o" || strings.HasPrefix(current.text, "-o="):
			return false
		case strings.HasPrefix(current.text, "./") || strings.HasPrefix(current.text, "overgo/"):
			packages = append(packages, current.text)
		}
	}
	return len(packages) == 1 && strings.Count(strings.TrimSuffix(strings.TrimPrefix(packages[0], "./"), "/"), "/") == 1 &&
		strings.HasPrefix(strings.TrimPrefix(packages[0], "./"), "cmd/")
}
