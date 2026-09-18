package gate

import (
	"context"
	"fmt"
	"strings"

	"overgo/internal/runrecord"
)

// reusableAcceptance is the strict acceptance grammar an owner package receipt
// can discharge: a bare go test of one package selected by -run, run once. An
// optional `test -f <file> &&` guard is allowed because the owner run already
// built the package. Anything else keeps its own execution.
type reusableAcceptance struct {
	packagePath string
	target      *runrecord.GoTestTarget
	short       bool
}

// parseReusableAcceptance recognizes the reusable grammar and returns false for
// every command outside it, so an unrecognized verifier always runs.
func parseReusableAcceptance(verify string) (reusableAcceptance, bool) {
	command := strings.TrimSpace(verify)
	// One optional file-existence guard before the go test command.
	if rest, ok := strings.CutPrefix(command, "test -f "); ok {
		guard, after, found := strings.Cut(rest, " && ")
		if !found || strings.ContainsAny(guard, " \t") {
			return reusableAcceptance{}, false
		}
		command = strings.TrimSpace(after)
	}
	// No shell composition, redirection, environment prefix or second command.
	if strings.ContainsAny(command, ";|&<>\n") || strings.Contains(command, "=") && strings.Index(command, "=") < strings.Index(command, "go test") {
		return reusableAcceptance{}, false
	}
	fields, ok := shellFields(command)
	if !ok {
		return reusableAcceptance{}, false
	}
	for _, token := range []string{"go", "test"} {
		if len(fields) == 0 || fields[0] != token {
			return reusableAcceptance{}, false
		}
		fields = fields[1:]
	}
	parsed := reusableAcceptance{}
	packages, runs, sawCount := 0, 0, false
	for i := 0; i < len(fields); i++ {
		field := fields[i]
		switch {
		case field == "-run":
			if i+1 >= len(fields) {
				return reusableAcceptance{}, false
			}
			i++
			targets, err := runrecord.GoTestTargets("go test -run "+quoteField(fields[i]), true)
			if err != nil || len(targets) != 1 {
				return reusableAcceptance{}, false
			}
			parsed.target, runs = targets[0], runs+1
		case field == "-short":
			parsed.short = true
		case field == "-count=1":
			sawCount = true
		case strings.HasPrefix(field, "-"):
			return reusableAcceptance{}, false // an unrecognized flag changes the run
		default:
			parsed.packagePath, packages = field, packages+1
		}
	}
	if packages != 1 || runs != 1 || !sawCount {
		return reusableAcceptance{}, false
	}
	return parsed, true
}

// shellFields splits a command on unquoted whitespace, honoring single and
// double quotes; it refuses backslash escapes and unbalanced quotes so an
// ambiguous command is never parsed as reusable.
func shellFields(command string) ([]string, bool) {
	var fields []string
	var field strings.Builder
	var quote byte
	inQuote, inField := false, false
	for i := range len(command) {
		c := command[i]
		switch {
		case c == '\\':
			return nil, false
		case inQuote:
			if c == quote {
				inQuote = false
			} else {
				field.WriteByte(c)
			}
		case c == '\'' || c == '"':
			quote, inQuote, inField = c, true, true
		case c == ' ' || c == '\t':
			if inField {
				fields = append(fields, field.String())
				field.Reset()
				inField = false
			}
		default:
			field.WriteByte(c)
			inField = true
		}
	}
	if inQuote {
		return nil, false
	}
	if inField {
		fields = append(fields, field.String())
	}
	return fields, true
}

// quoteField restores one -run value for re-parsing, choosing a quote the value
// does not contain; a value carrying both quote styles yields an unparseable
// command, which the caller treats as not reusable.
func quoteField(value string) string {
	if !strings.Contains(value, "'") {
		return "'" + value + "'"
	}
	return `"` + value + `"`
}

// reuseAcceptanceVerdict returns the acceptance verdict a completed owner run
// already proved, without executing the verifier again. It reuses a receipt
// only when the package ran in this exact mode, its receipt passed, and every
// recorded test the -run target selects passed with none skipped; any mismatch
// returns reused=false so the verifier runs.
func (g *gateContext) reuseAcceptanceVerdict(ctx context.Context, verify string) (runrecord.VerdictClass, bool, error) {
	acceptance, ok := parseReusableAcceptance(verify)
	if !ok || g.testPlan == nil || g.testPlan.ledger == nil {
		return "", false, nil
	}
	graph, err := g.inputGraph()
	if err != nil {
		return "", false, err
	}
	pkg, ok := graph.canonicalPackage(acceptance.packagePath)
	if !ok {
		return "", false, nil
	}
	mode := "complete"
	if acceptance.short {
		mode = "short"
	}
	obligation, found := g.testPlan.ledger.obligations[pkg]
	if !found || obligation.Scope != mode+":"+pkg {
		return "", false, nil
	}
	store, err := g.openStore()
	if err != nil {
		return "", false, err
	}
	receipt, found, err := packageReceiptCodec.Resolve(ctx, store, packageReceiptAlias+obligation.ID.String())
	if err != nil || !found || !receipt.Passed || !receiptDischarges(acceptance.target, receipt.Tests) {
		return "", false, err
	}
	g.note(fmt.Sprintf("acceptance reuse: owner %s receipt discharged %s without re-execution", mode, pkg))
	return runrecord.ClassifyVerifyCommand(verify), true, nil
}

// receiptDischarges reports whether a completed receipt's named verdicts already
// prove a -run target: at least one recorded test the target selects passed, and
// no test it selects was skipped or otherwise short of a pass. An empty verdict
// set (a historical receipt) discharges nothing.
func receiptDischarges(target *runrecord.GoTestTarget, tests map[string]string) bool {
	matched := false
	for name, action := range tests {
		if !target.MatchString(name) {
			continue
		}
		if action != "pass" {
			return false
		}
		matched = true
	}
	return matched
}
