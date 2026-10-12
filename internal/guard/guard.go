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
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/processlock"
)

const commandBoundary = "(?:^|[\\n;|&(`])\\s*"

// GateEnv set to "1" marks a commit issued by cmd/gate itself; the raw-commit
// rule stands down for exactly that process tree and nothing wider.
const GateEnv = "OVERGO_COMMIT_GATE"

var (
	doubleQuoted = regexp.MustCompile(`"(?:[^"\\]|\\.)*"`)
	singleQuoted = regexp.MustCompile(`'[^']*'`)
	wrapper      = regexp.MustCompile(`(?i)\b(?:bash|sh|zsh|dash|ksh|mksh)(?:\.exe)?\s+(?:-\S+\s+)*-\w*c\w*\s+|\beval\s+`)
	heredoc      = regexp.MustCompile(`<<-?\s*['"]?[A-Za-z_]`)
	sourcePath   = regexp.MustCompile(`(?i)\.(?:go|sh|py|json|jsonl|md|yaml|cu|cuh)\b`)
	writeSink    = regexp.MustCompile(`(?i)>\s*[^\s>]|\btee\b|open\s*\([^)]*['"]w|json\.dump|\.write\s*\(|\bsed\s+-i`)
	xargsRun     = regexp.MustCompile(`(?i)\bxargs\s+(?:-\S+\s+)*`)
	findExec     = regexp.MustCompile(`(?i)-exec\s+`)
	powerShell   = regexp.MustCompile(`(?i)\b(?:powershell|pwsh)\b`)
	gitClean     = regexp.MustCompile("(?i)" + commandBoundary + `git\s+clean\b`)
	forceDelete  = regexp.MustCompile("(?i)" + commandBoundary + `(?:rm|rmdir|del|rd|remove-item)\b[^\n;|&]*(?:-rf|-fr|-force|--force|-recurse\s+-force|-fdx)`)
	rmForce      = regexp.MustCompile("(?i)" + commandBoundary + `rm\b[^\n;|&]*\s-[a-z]*f`)
	protected    = regexp.MustCompile("(?i)" + commandBoundary + `(?:rm|rmdir|del|rd|remove-item|unlink|shred|mv|move-item|move|chmod|attrib|icacls|takeown)\b[^\n;|&]*(?:models[\\/]|datasets[\\/]|checkpoints[\\/]|docs[\\/]repodb|repodb-store|overgodb-store)`)
	// A landing's output goes to the terminal or a filter, never a file: the
	// sinks that keep nothing are removed before the rule looks for one.
	harmlessSink   = regexp.MustCompile(`\d?>>?\s*(?:/dev/null\b|&\d)`)
	landTranscript = regexp.MustCompile(`(?i)\./cmd/(?:loop|gate)\b[^\n;&]*(?:>|\|\s*tee\b)`)
	worktreeRm     = regexp.MustCompile("(?i)" + commandBoundary + `git\s+worktree\s+remove\b`)
	// find's delete verb receives the substituted {} as its target, so the
	// protected path lives in find's OWN argument and no verb-scoped rule can
	// see it; judge the pair (protected traversal root, -exec delete verb).
	findDelete = regexp.MustCompile("(?i)" + commandBoundary + `find\b[^\n;|&]*(?:models[\\/]|datasets[\\/]|checkpoints[\\/]|docs[\\/]repodb|repodb-store)[^\n;|&]*(?:-delete\b|-(?:exec|execdir)\s+(?:rm|rmdir|del|unlink|shred|mv|chmod)\b)`)
	// Whitespace before "commit", not \b: \b matches after a hyphen, so a
	// lookbehind-free rule would deny `git log --grep=pre-commit`.
	rawCommit = regexp.MustCompile("(?i)" + commandBoundary + `git\s+(?:[^\n;|&]*\s)?commit(?:\s|$)`)
)

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

// landing is a command that lands a change through the gate.
var landing = regexp.MustCompile(`(?i)\./cmd/(?:loop\b[^\n;&|]*\s-land\b|gate\b)`)

// reviewVerdict refuses what a session that changes nothing here may not do,
// and allows the rest. A session that declared itself a review session (its
// state says so) lands nothing and edits no file the repository tracks or
// would track, so it has no work of its own for the stop gate to hold it to.
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
	if landing.MatchString(call.ToolInput.Command) {
		return session + " lands nothing here; " + instead
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
		if m := ruleVerdict(command, root); m != "" {
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
	if depth >= 4 {
		return out
	}
	for _, match := range wrapper.FindAllStringIndex(command, -1) {
		if payload := shellArgument(command[match[1]:]); payload != "" {
			out = append(out, payloads(payload, depth+1)...)
		}
	}
	return out
}

func shellArgument(raw string) string {
	raw = strings.TrimLeft(raw, " \t\r\n")
	if raw == "" {
		return ""
	}
	quote := raw[0]
	if quote != '\'' && quote != '"' {
		if i := strings.IndexAny(raw, " \t\r\n"); i >= 0 {
			return raw[:i]
		}
		return raw
	}
	var out strings.Builder
	for i := 1; i < len(raw); i++ {
		if raw[i] == quote {
			return out.String()
		}
		if quote == '"' && raw[i] == '\\' && i+1 < len(raw) && strings.ContainsRune("\"\\$`", rune(raw[i+1])) {
			i++
		}
		out.WriteByte(raw[i])
	}
	return out.String()
}

// executableText keeps quoted arguments but masks their shell separators.
// Direct targets remain visible; commands merely described in messages do
// not. Unescaped $() and backticks inside double quotes stay executable
// boundaries; escaped substitutions and single-quoted text are data.
func executableText(raw string) string {
	var out strings.Builder
	var quote byte
	var escapedDollar bool
	for i := 0; i < len(raw); i++ {
		ch := raw[i]
		if quote == 0 {
			if ch == '\'' || ch == '"' {
				quote = ch
				continue
			}
			out.WriteByte(ch)
			continue
		}
		if ch == quote {
			quote = 0
			escapedDollar = false
			continue
		}
		escaped := false
		if quote == '"' && ch == '\\' && i+1 < len(raw) && strings.ContainsRune("\"\\$`", rune(raw[i+1])) {
			i++
			ch = raw[i]
			escaped = true
		}
		executableBoundary := quote == '"' && !escaped && (ch == '`' || (ch == '(' && i > 0 && raw[i-1] == '$' && !escapedDollar))
		if executableBoundary {
			out.WriteByte(ch)
			escapedDollar = false
			continue
		}
		if strings.ContainsRune("\n;|&(`", rune(ch)) {
			out.WriteByte(' ')
			escapedDollar = false
			continue
		}
		out.WriteByte(ch)
		escapedDollar = quote == '"' && escaped && ch == '$'
	}
	return out.String()
}

func ruleVerdict(command, root string) string {
	raw := command
	executable := executableText(raw)
	stripped := doubleQuoted.ReplaceAllString(raw, `""`)
	stripped = singleQuoted.ReplaceAllString(stripped, `''`)
	stripped = xargsRun.ReplaceAllString(stripped, "; ")
	stripped = findExec.ReplaceAllString(stripped, "; ")

	if powerShell.MatchString(stripped) {
		return "PowerShell/pwsh invocation is disabled; use bash"
	}
	if gitClean.MatchString(stripped) {
		return "git clean can delete gitignored model/dataset stores"
	}
	if forceDelete.MatchString(stripped) || rmForce.MatchString(stripped) {
		return "force-delete flags are banned; never override read-only"
	}
	if protected.MatchString(executable) {
		return "delete/move/chmod under a protected data path (models/, datasets/, checkpoints/, docs/repodb/, overgodb-store)"
	}
	if findDelete.MatchString(executable) {
		return "find with -delete or -exec <delete-verb> over a protected data path; the delete target is the " +
			"substituted {} so no verb rule can see it -- ask the user"
	}
	if worktreeRm.MatchString(stripped) {
		return "git worktree remove recurses INTO junctioned model directories and has already destroyed 170 GB of " +
			"weights; retire the lane with 'git branch -d' and leave the directory. If the directory must go, check " +
			"'dir /AL /S' for reparse points first and ask the user"
	}
	if os.Getenv(GateEnv) != "1" && !mergeInProgress(root) && rawCommit.MatchString(stripped) {
		return "raw git commit bypasses the small gate (format, vet, rules, tests, lanes, the row's verify); run " +
			"'go run ./cmd/loop -land <item>/<step> -message-file -' instead. --no-verify does not help: the gate is " +
			"the thing being skipped, not a hook"
	}
	if landTranscript.MatchString(harmlessSink.ReplaceAllString(stripped, "")) {
		return "a landing's output is not kept in a file: loop -land prints the lines that decide the outcome, and the " +
			"commit's 'Measured by the gate' line holds each phase's time; give the message on standard input with " +
			"-message-file -"
	}
	if heredoc.MatchString(raw) && strings.Contains(raw, `\`) && sourcePath.MatchString(raw) && writeSink.MatchString(raw) {
		return "a Bash heredoc EATS one level of backslash escaping and will write mangled content; use the Write or " +
			"Edit tool for anything containing a backslash"
	}
	if buildsCommandHere(stripped) {
		return "go build of one command without -o writes its binary into the working directory; at the repository " +
			"root that moves the root folder's time, and go test reruns every suite that stats it; build with -o bin/<name>.exe"
	}
	return ""
}

// buildsCommandHere reports whether a command segment runs go build on
// exactly one ./cmd package without -o, which writes that command's binary
// into the working directory. Several packages, or a library, write nothing.
func buildsCommandHere(command string) bool {
	for segment := range strings.FieldsFuncSeq(command, func(r rune) bool { return strings.ContainsRune(";&|\n", r) }) {
		fields := strings.Fields(segment)
		at := slices.Index(fields, "build")
		if at < 1 || fields[at-1] != "go" {
			continue
		}
		var packages []string
		output := false
		for _, field := range fields[at+1:] {
			switch {
			case field == "-o" || strings.HasPrefix(field, "-o="):
				output = true
			case strings.HasPrefix(field, "./") || strings.HasPrefix(field, "overgo/"):
				packages = append(packages, field)
			}
		}
		if !output && len(packages) == 1 && strings.Count(strings.TrimSuffix(strings.TrimPrefix(packages[0], "./"), "/"), "/") == 1 &&
			strings.HasPrefix(strings.TrimPrefix(packages[0], "./"), "cmd/") {
			return true
		}
	}
	return false
}

// mergeInProgress: MERGE_HEAD exists only while a merge is actually in
// progress; a merge commit cannot go through the gate by construction, so the
// exemption window is exactly the ungateable case. Linked worktrees keep
// MERGE_HEAD in the gitdir their .git file points at.
func mergeInProgress(root string) bool {
	for _, base := range []string{root, filepath.Dir(root)} {
		gitPath := filepath.Join(base, ".git")
		if info, err := os.Stat(gitPath); err == nil && info.IsDir() {
			if _, err := os.Stat(filepath.Join(gitPath, "MERGE_HEAD")); err == nil {
				return true
			}
			continue
		}
		file, err := os.Open(gitPath)
		if err != nil {
			continue
		}
		scanner := bufio.NewScanner(file)
		line := ""
		if scanner.Scan() {
			line = strings.TrimSpace(scanner.Text())
		}
		_ = file.Close()
		if !strings.HasPrefix(line, "gitdir:") {
			continue
		}
		gitDir := strings.TrimSpace(strings.TrimPrefix(line, "gitdir:"))
		if !filepath.IsAbs(gitDir) {
			gitDir = filepath.Join(filepath.Dir(gitPath), gitDir)
		}
		if _, err := os.Stat(filepath.Join(gitDir, "MERGE_HEAD")); err == nil {
			return true
		}
	}
	return false
}
