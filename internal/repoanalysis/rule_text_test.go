package repoanalysis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unicode"

	"overgo/internal/testcheck"
)

// ruleMarkers are the forms skill.md writes a rule in: a condition and what follows from it, and the check that holds
// it. Text a session reads that holds one states a rule beside skill.md.
var ruleMarkers = []string{" => ", " <=> ", "(check: "}

// personWords are the words that state a rule as a person's: the guidance states every rule as the system's own.
var personWords = map[string]bool{
	"user": true, "users": true, "human": true, "humans": true,
	"owner": true, "owners": true, "operator": true, "operators": true,
}

// sessionTextSources are the Go files whose strings a session reads: the hook's start text and messages, the plan
// command's prompt and help, the driver's prompt and the scheduled session's instructions, and the guard's refusals.
var sessionTextSources = []string{
	"cmd/loophook/doctrine.go", "cmd/loophook/main.go", "cmd/plan/main.go", "cmd/plan/dispatch.go",
	"cmd/loop/session_prompt.go", "cmd/loop/init_config.go", "internal/guard/guard.go",
}

// personSources are the texts a rule is stated in as the system's own: skill.md, the hook's start text and messages,
// and the plan command's prompt and help. The plan's reasoning and the guard's refusals may quote whom they serve.
var personSources = []string{"skill.md", "cmd/loophook/doctrine.go", "cmd/loophook/main.go", "cmd/plan/main.go"}

// linesWhere names each line of texts, by source and line, that holds says true of.
func linesWhere(texts map[string]string, holds func(line string) bool) []string {
	var found []string
	for source, text := range texts {
		for number, line := range strings.Split(text, "\n") {
			if holds(line) {
				found = append(found, source+" line "+strconv.Itoa(number+1)+": "+line)
			}
		}
	}
	return found
}

// ruleForm says a line states a rule in skill.md's form.
func ruleForm(line string) bool {
	return slices.ContainsFunc(ruleMarkers, func(marker string) bool { return strings.Contains(line, marker) })
}

// namesPerson says a line names a person; a path's segments are names.
func namesPerson(line string) bool {
	return slices.ContainsFunc(strings.FieldsFunc(strings.ToLower(line), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '_' && r != '/'
	}), func(word string) bool { return !strings.Contains(word, "/") && personWords[strings.Trim(word, "-")] })
}

// TestRuleTextLivesOnlyInSkillAndThePointer holds the written rules to skill.md and AGENTS.md: rules lived in more
// than a dozen places, and where they disagreed a session reconciled them, asking the owner as its safe way out. The
// strings the hook, the plan and loop commands and the guard print, and the plan, state no rule in skill.md's form and
// name no person; the session's start text is one sentence pointing at skill.md; and a rule sentence added to any of
// them is found.
func TestRuleTextLivesOnlyInSkillAndThePointer(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	read := func(path string) string {
		return string(testcheck.Must(os.ReadFile(filepath.Join(root, filepath.FromSlash(path))))(t))
	}
	texts, persons := map[string]string{"docs/plan.json": read("docs/plan.json")}, map[string]string{"skill.md": read("skill.md")}
	var start string
	for _, path := range sessionTextSources {
		files := token.NewFileSet()
		file := testcheck.Must(parser.ParseFile(files, filepath.Join(root, filepath.FromSlash(path)), nil, 0))(t)
		ast.Inspect(file, func(node ast.Node) bool {
			if literal, ok := node.(*ast.BasicLit); ok && literal.Kind == token.STRING {
				value := testcheck.Must(strconv.Unquote(literal.Value))(t)
				if texts[files.Position(literal.Pos()).String()] = value; slices.Contains(personSources, path) {
					persons[files.Position(literal.Pos()).String()] = value
				}
			}
			if spec, ok := node.(*ast.ValueSpec); ok && path == "cmd/loophook/doctrine.go" && spec.Names[0].Name == "doctrineText" {
				start = testcheck.Must(strconv.Unquote(spec.Values[0].(*ast.BasicLit).Value))(t)
			}
			return true
		})
	}
	for _, line := range linesWhere(texts, ruleForm) {
		t.Errorf("rule text outside skill.md: %s", line)
	}
	for _, line := range linesWhere(persons, namesPerson) {
		t.Errorf("a rule names a person: %s", line)
	}
	if strings.Count(start, ". ") > 1 || strings.Contains(start, "\n") || !strings.Contains(start, "skill.md") {
		t.Errorf("the session's start text says more than where the guidance is: %q", start)
	}
	if added := linesWhere(map[string]string{"a hook's output": "a typed message => its turn ends (check: cmd/loophook TestStopDecision)"}, ruleForm); len(added) == 0 {
		t.Error("a rule sentence added to a hook's output was not found")
	}
}
