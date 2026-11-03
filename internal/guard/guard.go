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
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
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

// shellNames are the programs -c hands a command to.
var shellNames = []string{"bash", "dash", "ksh", "mksh", "sh", "zsh"}

// sourceExtensions name the files a heredoc must not write.
var sourceExtensions = []string{"cu", "cuh", "go", "json", "jsonl", "md", "py", "sh", "yaml"}

// payloadDepth is how many wrappers deep a command is read.
const payloadDepth = 4

// ToolCall is the harness PreToolUse payload subset the guard judges: the
// session the call is made in, a shell tool's command, and the file an
// editing tool writes.
type ToolCall struct {
	ToolName  string `json:"tool_name"`
	Session   string `json:"session_id"`
	ToolInput struct {
		Command string `json:"command"`
		Path    string `json:"file_path"`
	} `json:"tool_input"`
}

// WorktreeHolder is the file, in a worktree's process state, that names the
// implementing session that holds the worktree. A landing takes the
// checkout's whole tree as its candidate, so two implementing sessions in one
// worktree would land each other's files; the second is sent to a worktree of
// its own. A hook is a short process and can keep no lock of the system's
// open for a session, so the file is the holding, and its creation is atomic.
const WorktreeHolder = "implementing_session"

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
// state says so) lands nothing, removes no leftover and edits no file the
// repository tracks or would track, so it has no work of its own for the stop
// gate to hold it to, and nothing here that is its own to remove.
// Neither does an implementing session in a worktree another implementing
// session holds (its state names the holder): a landing takes the checkout's
// whole tree, so its files would join the holder's commit, and it is sent to
// a worktree of its own. A file outside the checkout, or one the repository
// ignores, is scratch and is any session's to write.
func reviewVerdict(call ToolCall, root string) string {
	var state struct {
		Review bool
		Holder string
	}
	data, _ := os.ReadFile(filepath.Join(root, processlock.StateDirectory, SessionState(call.Session)))
	if call.Session == "" || json.Unmarshal(data, &state) != nil || !state.Review && state.Holder == "" {
		return ""
	}
	session, instead := "a review session", "report what it found (go run ./cmd/finding)"
	if !state.Review {
		session, instead = "a session in a worktree session "+state.Holder+" holds", "work in a worktree of its own (git worktree add)"
	}
	if Landing(call.ToolInput.Command) {
		return session + " lands nothing here; " + instead
	}
	if removal(call.ToolInput.Command) {
		return session + " removes nothing here; " + instead
	}
	relative, err := filepath.Rel(root, cmp.Or(call.ToolInput.Path, root))
	if err != nil || relative == "." || !filepath.IsLocal(relative) {
		return ""
	}
	if _, ignored := gitauthority.Query(context.Background(), root, "check-ignore", "-q", relative); ignored == nil {
		return ""
	}
	return session + " edits no file the repository tracks (" + filepath.ToSlash(relative) + "); " + instead + ", or write to a scratch path"
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

// keepsLanding says a landing's output, or that of what it pipes into, goes to a file: a landing's goes to the terminal or a filter.
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
