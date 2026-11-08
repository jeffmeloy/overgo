package repoanalysis

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unicode"

	"overgo/internal/testcheck"
)

// skillJudgement is the section of skill.md whose lines no code can hold, each saying why.
const skillJudgement = "## Judgement"

// skillRule says a line of skill.md is a rule: a point of a list, or a numbered step.
func skillRule(line string) bool {
	text := strings.TrimSpace(line)
	number, _, numbered := strings.Cut(text, ". ")
	return strings.HasPrefix(text, "- ") || numbered && number != "" && strings.TrimFunc(number, unicode.IsDigit) == ""
}

// TestEverySkillLineOutsideJudgementCitesItsCheck holds skill.md to rules a check holds: every line outside Judgement,
// its definitions and aims among them, cites as "(check: <package> <test>, ...)" tests that exist in the packages it
// names, so a weaker model reads the whole guidance as what the code does. A line no check holds is judgement, saying
// why, or it goes to code, data or a command's help.
func TestEverySkillLineOutsideJudgementCitesItsCheck(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	sources := map[string]string{}
	exists := func(directory, name string) bool {
		if _, read := sources[directory]; !read {
			files, _ := filepath.Glob(filepath.Join(root, filepath.FromSlash(directory), "*_test.go"))
			var all strings.Builder
			for _, file := range files {
				all.Write(testcheck.Must(os.ReadFile(file))(t))
			}
			sources[directory] = all.String()
		}
		return strings.Contains(sources[directory], "func "+name+"(")
	}
	section := ""
	for number, line := range strings.Split(string(testcheck.Must(os.ReadFile(filepath.Join(root, "skill.md")))(t)), "\n") {
		if strings.HasPrefix(line, "#") {
			section = strings.TrimSpace(line)
			continue
		}
		if section == skillJudgement || !skillRule(line) {
			continue
		}
		where := "skill.md:" + strconv.Itoa(number+1)
		_, cited, citing := strings.Cut(line, "(check: ")
		if !citing {
			t.Errorf("%s cites no check: %s", where, line)
			continue
		}
		cited, _, _ = strings.Cut(cited, ")")
		for entry := range strings.SplitSeq(cited, ",") {
			directory, name, named := strings.Cut(strings.TrimSpace(entry), " ")
			if !named || !strings.HasPrefix(name, "Test") || !exists(directory, name) {
				t.Errorf("%s cites %q, which is no test of a package: %s", where, strings.TrimSpace(entry), line)
			}
		}
	}
}
