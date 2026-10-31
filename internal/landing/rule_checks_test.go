package landing

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// citation is a rule's citation of the checks that hold it: "(check: <package directory> <TestName>, ...)".
var citation = regexp.MustCompile(`\(check: ([^)]+)\)`)

// sentenceEnd ends a sentence of a standing paragraph: a full stop before the capital that opens the next.
var sentenceEnd = regexp.MustCompile(`\.\s+([A-Z])`)

// ruleLine is a line of skill.md that states a rule: an item of a list, plain or numbered.
var ruleLine = regexp.MustCompile(`^\s*(- |\d+\. )`)

// definitionSections are the sections of skill.md whose lines define its terms and state its aims, which no check
// holds: every other list line is a rule.
var definitionSections = []string{"# Intent", "## Terms"}

// TestEveryRuleCitesItsCheck holds the doctrine to rules that are code: every rule line of skill.md outside its
// definitions, and every standing paragraph of the plan, cites the tests that fail when the rule is broken, each a
// test that exists in the package named, or the rule goes. The review of 2026-10-30 found the run-order rule
// restated in prose beside the tests that enforce it, and a rule kept only in memory.
func TestEveryRuleCitesItsCheck(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	tests := map[string]string{}
	exists := func(directory, name string) bool {
		source, read := tests[directory]
		if !read {
			files, _ := filepath.Glob(filepath.Join(root, filepath.FromSlash(directory), "*_test.go"))
			var all strings.Builder
			for _, file := range files {
				data, err := os.ReadFile(file)
				if err != nil {
					t.Fatal(err)
				}
				all.Write(data)
			}
			source = all.String()
			tests[directory] = source
		}
		return strings.Contains(source, "func "+name+"(")
	}
	check := func(where, text string) {
		t.Helper()
		cited := citation.FindAllStringSubmatch(text, -1)
		if len(cited) == 0 {
			t.Errorf("%s cites no check: %s", where, text)
		}
		for _, match := range cited {
			for entry := range strings.SplitSeq(match[1], ",") {
				directory, name, named := strings.Cut(strings.TrimSpace(entry), " ")
				if !named || !strings.HasPrefix(name, "Test") || !exists(directory, name) {
					t.Errorf("%s cites %q, which is no test of a package: %s", where, strings.TrimSpace(entry), text)
				}
			}
		}
	}
	section := ""
	for number, line := range strings.Split(read("skill.md"), "\n") {
		if strings.HasPrefix(line, "#") {
			section = strings.TrimSpace(line)
			continue
		}
		if ruleLine.MatchString(line) && !slices.Contains(definitionSections, section) {
			check("skill.md:"+strconv.Itoa(number+1), line)
		}
	}
	var document struct {
		Paragraphs []struct {
			Governs string `json:"governs"`
			Text    string `json:"text"`
		} `json:"paragraphs"`
	}
	if err := json.Unmarshal([]byte(read("docs/plan.json")), &document); err != nil {
		t.Fatal(err)
	}
	for index, paragraph := range document.Paragraphs {
		if paragraph.Governs != "standing" {
			continue
		}
		// Each sentence is a rule of its own.
		for sentence := range strings.SplitSeq(sentenceEnd.ReplaceAllString(paragraph.Text, ".\n$1"), "\n") {
			check("docs/plan.json standing paragraph "+strconv.Itoa(index), sentence)
		}
	}
}

// quoted is a span of code in a rule line, which the rule's own form does not govern.
var quoted = regexp.MustCompile("`[^`]*`")

// history is a date or a commit id: what the commit message holds, never a rule.
var history = regexp.MustCompile(`\b(\d{4}-\d{2}-\d{2}|[0-9a-f]{9,40})\b`)

// TestTheDoctrineHoldsItsOwnForm holds each rule line of skill.md, outside its code and its citation, to the form its
// Writing section states: no semicolon joins two clauses, no date or commit id dates it, and no rule is stated on two
// lines.
func TestTheDoctrineHoldsItsOwnForm(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join("..", "..", "skill.md"))
	if err != nil {
		t.Fatal(err)
	}
	stated := map[string]int{}
	for number, line := range strings.Split(string(data), "\n") {
		if !ruleLine.MatchString(line) {
			continue
		}
		rule := strings.TrimSpace(citation.ReplaceAllString(quoted.ReplaceAllString(line, "code"), ""))
		switch earlier, seen := stated[strings.ToLower(rule)]; {
		case strings.Contains(rule, ";"):
			t.Errorf("skill.md:%d joins two clauses with a semicolon: %s", number+1, line)
		case history.MatchString(rule):
			t.Errorf("skill.md:%d holds a date or a commit id: %s", number+1, line)
		case seen:
			t.Errorf("skill.md:%d states the rule of line %d again: %s", number+1, earlier, line)
		}
		stated[strings.ToLower(rule)] = number + 1
	}
}
