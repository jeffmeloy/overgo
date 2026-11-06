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

	"overgo/internal/testcheck"
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

// judgementSection is the section of skill.md whose rules no code can hold, each saying why in its because clause.
const judgementSection = "## Judgement"

// TestEveryRuleCitesItsCheck holds the doctrine to rules that are code: every rule line of skill.md outside its
// definitions, and every standing paragraph of the plan, cites the tests that fail when the rule is broken, each a
// test that exists in the package named, or the rule goes. The review of 2026-10-30 found the run-order rule
// restated in prose beside the tests that enforce it, and a rule kept only in memory.
func TestEveryRuleCitesItsCheck(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	read := func(path string) string {
		t.Helper()
		data := testcheck.Must(os.ReadFile(filepath.Join(root, filepath.FromSlash(path))))(t)
		return string(data)
	}
	tests := map[string]string{}
	exists := func(directory, name string) bool {
		source, read := tests[directory]
		if !read {
			files, _ := filepath.Glob(filepath.Join(root, filepath.FromSlash(directory), "*_test.go"))
			var all strings.Builder
			for _, file := range files {
				data := testcheck.Must(os.ReadFile(file))(t)
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
		if ruleLine.MatchString(line) && !slices.Contains(definitionSections, section) && section != judgementSection {
			check("skill.md:"+strconv.Itoa(number+1), line)
		}
	}
	var document struct {
		Paragraphs []struct {
			Governs string `json:"governs"`
			Text    string `json:"text"`
		} `json:"paragraphs"`
	}
	testcheck.NoError(t, json.Unmarshal([]byte(read("docs/plan.json")), &document))
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

// TestAJudgementLineSaysWhyNoCodeHoldsIt holds each rule of skill.md's Judgement section to a because clause saying why
// no code holds it, and to no citation of a check, since a rule a check holds belongs with the rules that cite theirs.
// The section is what the system leaves to a model, read at a glance.
func TestAJudgementLineSaysWhyNoCodeHoldsIt(t *testing.T) {
	t.Parallel()
	data := testcheck.Must(os.ReadFile(filepath.Join("..", "..", "skill.md")))(t)
	section, judged := "", 0
	for number, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "#") {
			section = strings.TrimSpace(line)
			continue
		}
		if section != judgementSection || !ruleLine.MatchString(line) {
			continue
		}
		judged++
		if _, why, said := strings.Cut(line, " because "); !said || strings.TrimSpace(why) == "" {
			t.Errorf("skill.md:%d is judgement and says no reason no code holds it: %s", number+1, line)
		}
		if strings.Contains(line, "(check:") {
			t.Errorf("skill.md:%d is judgement and cites a check, so it belongs with the rules that cite theirs: %s", number+1, line)
		}
	}
	if judged == 0 {
		t.Errorf("skill.md has no %s section of rules", judgementSection)
	}
}

// unsoundCitations are rules skill.md stated with a check that does not prove them, each as the rule read and the test
// it cited: the review that sorted the doctrine found that the test held something else. Each rule was narrowed to what
// its test holds, moved to the Judgement section or given the test it named.
var unsoundCitations = []struct{ rule, test string }{
	{"rank by recorded outcomes and expected cost", "TestSchedulerRankedByRealizedDelta"},
	{"missing resource => explicit resume condition and continue independent work", "TestDispatchPassesRowsAwaitingADecision"},
	{"missing resource => explicit resume condition and continue independent work", "TestALandingWaitingOnARunLeavesTheCheckoutFree"},
	{"preflight first = `go run ./cmd/gate -plan <item>/<step> -message-file - -preflight`", "TestPreflightRunsTheCheapStagesOnly"},
	{"native `go test` selects", "TestSelectionAnalyzersGone"},
	{"an A/B => its failed requests counted beside its rates", "TestServingCountsWhatUsersHit"},
	{"scale up <=> measured benefit", "TestDispatchSchedulingPreservesResults"},
	{"`adaptive_new` = read-only", "TestNoTestReadsAdaptiveNew"},
}

// TestNoRuleCitesACheckThatDoesNotProveIt refuses each unsound citation the doctrine's review found, written back: a
// rule line of skill.md that states one of those rules and cites the test that did not prove it. No test can prove a
// citation in general, so this holds the ones found.
func TestNoRuleCitesACheckThatDoesNotProveIt(t *testing.T) {
	t.Parallel()
	data := testcheck.Must(os.ReadFile(filepath.Join("..", "..", "skill.md")))(t)
	for number, line := range strings.Split(string(data), "\n") {
		rule, cited, citing := strings.Cut(strings.TrimPrefix(line, ruleLine.FindString(line)), "(check: ")
		if !ruleLine.MatchString(line) || !citing {
			continue
		}
		cited, _, _ = strings.Cut(cited, ")")
		for entry := range strings.SplitSeq(cited, ",") {
			_, test, _ := strings.Cut(strings.TrimSpace(entry), " ")
			for _, unsound := range unsoundCitations {
				if strings.HasPrefix(rule, unsound.rule) && test == unsound.test {
					t.Errorf("skill.md:%d states %q citing %s, which does not prove it: %s", number+1, unsound.rule, test, line)
				}
			}
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
	data := testcheck.Must(os.ReadFile(filepath.Join("..", "..", "skill.md")))(t)
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

// retiredClaims are claims the doctrine no longer makes, each with the text it left: what the owner declined, what the
// code no longer does, and what the code already says. "docs/plan.json" is the plan's paragraphs, which every session
// reads; a row may quote a retired claim to say why it went.
var retiredClaims = []struct{ source, claim string }{
	{"skill.md", "never a promotion"},
	{"docs/plan.json", "no passed state and no promotion"},
	{"skill.md", "supervises unattended work"},
	{"skill.md", "and `build/` live in `OVERGO_DATA_ROOT`"},
	{"docs/plan.json", "puts every evaluation bound to it again"},
	{"cmd/loophook/doctrine.go", "holds its worktree from its first message"},
	{"cmd/loophook/doctrine.go", "For unattended continuity run"},
	{"cmd/loophook/doctrine.go", "continuity is owned by cmd/loop"},
	{"cmd/loophook/doctrine.go", "The guard refuses\n    a landing whose output is sent to a file"},
	{"cmd/loophook/doctrine.go", "Do not run a separate"},
	{"README.md", "99 to 104 tok/s against Strata's 133 to 136"},
}

// TestDoctrineHoldsNoRetiredClaim refuses a retired claim written back into the text it left.
func TestDoctrineHoldsNoRetiredClaim(t *testing.T) {
	t.Parallel()
	texts := map[string]string{}
	for _, retired := range retiredClaims {
		text, read := texts[retired.source]
		if !read {
			data := testcheck.Must(os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(retired.source))))(t)
			text = string(data)
			if retired.source == "docs/plan.json" {
				var document struct {
					Paragraphs []struct {
						Text string `json:"text"`
					} `json:"paragraphs"`
				}
				testcheck.NoError(t, json.Unmarshal(data, &document))
				var paragraphs strings.Builder
				for _, paragraph := range document.Paragraphs {
					paragraphs.WriteString(paragraph.Text + "\n")
				}
				text = paragraphs.String()
			}
			texts[retired.source] = text
		}
		if strings.Contains(text, retired.claim) {
			t.Errorf("%s says %q, which the doctrine retired", retired.source, retired.claim)
		}
	}
}
