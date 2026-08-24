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
	"os"
	"path/filepath"
	"regexp"
	"strings"
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
	worktreeRm   = regexp.MustCompile("(?i)" + commandBoundary + `git\s+worktree\s+remove\b`)
	// find's delete verb receives the substituted {} as its target, so the
	// protected path lives in find's OWN argument and no verb-scoped rule can
	// see it; judge the pair (protected traversal root, -exec delete verb).
	findDelete = regexp.MustCompile("(?i)" + commandBoundary + `find\b[^\n;|&]*(?:models[\\/]|datasets[\\/]|checkpoints[\\/]|docs[\\/]repodb|repodb-store)[^\n;|&]*(?:-delete\b|-(?:exec|execdir)\s+(?:rm|rmdir|del|unlink|shred|mv|chmod)\b)`)
	// Whitespace before "commit", not \b: \b matches after a hyphen, so a
	// lookbehind-free rule would deny `git log --grep=pre-commit`.
	rawCommit = regexp.MustCompile("(?i)" + commandBoundary + `git\s+(?:[^\n;|&]*\s)?commit(?:\s|$)`)
)

// ToolCall is the harness PreToolUse payload subset the guard judges.
type ToolCall struct {
	ToolName  string `json:"tool_name"`
	ToolInput struct {
		Command string `json:"command"`
	} `json:"tool_input"`
}

// Verdict judges one tool call. Empty string allows; otherwise the deny
// reason. wrapped reports whether the denial came from an unwrapped payload.
func Verdict(call ToolCall, root string) (message string, wrapped bool) {
	if call.ToolName == "PowerShell" {
		return "the PowerShell tool is disabled; use the Bash tool", false
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
		return "raw git commit bypasses cmd/gate (hygiene + derived-scope tests + store record); run " +
			"'go run ./cmd/gate -message-file <path> -paths <csv>' instead. --no-verify does not help: the gate is " +
			"the thing being skipped, not a hook"
	}
	if heredoc.MatchString(raw) && strings.Contains(raw, `\`) && sourcePath.MatchString(raw) && writeSink.MatchString(raw) {
		return "a Bash heredoc EATS one level of backslash escaping and will write mangled content; use the Write or " +
			"Edit tool for anything containing a backslash"
	}
	return ""
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
