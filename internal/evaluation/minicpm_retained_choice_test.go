package evaluation

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
)

// Retained answers can validate extraction without another model acquisition.
// This checks the generated protocol; it grants no native likelihood credit.
func TestMiniCPMRetainedChoiceAcceptance(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(value string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(value)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	model := parse("model:sha256:3007c05b8ece556726a37980069cf6c0f1f966a48572b1c130c5643810c23a32")
	active, found, err := modelrecipe.ActiveRecord(t.Context(), store, model, recipe.TaskInference)
	if err != nil || !found || active.Definition.ID.String() != "recipe:sha256:5155b720ce15e6b05acf713a5f9c5bf86bf7e52963178c18503b03206578543c" {
		t.Fatalf("MiniCPM activation differs: %v", err)
	}
	definition, found := active.Definition.PrimaryDependency(recipe.DependencyDefinition)
	if !found {
		t.Fatal("MiniCPM model definition is absent")
	}
	total := 0
	for _, cell := range []struct {
		name, evidence         string
		cases, correct, groups int
	}{
		{"mmlu", "evidence:sha256:f55dc094513358a89f795262ae70db434b52f51825a4495dbe3194f60f7d0a60", 100, 26, 0},
		{"mmlu-pro", "evidence:sha256:7eaedf641d5a8f51cd1495ad66fbabf213bd29a17fd9a492d28b1fd9238ac49f", 12032, 3505, 14},
		{"bbh", "evidence:sha256:fee6ec46595eab51e1f9975f8920ab3c8b7126e9e5bdf445806b3c43d44ebcb3", 5761, 2372, 24},
		{"musr", "evidence:sha256:4cc9db91a0dd52367871c6f664455a650cfd5a4b231a302a6460a9228286be75", 756, 284, 3},
	} {
		t.Run(cell.name, func(t *testing.T) {
			value, err := RequireEvaluationEvidence(t.Context(), store, parse(cell.evidence))
			if err != nil {
				t.Fatal(err)
			}
			if value.Recipe != active.Definition.ID || value.ModelDefinition != definition || value.CodeCommit != "f278ca230121df1d0bedf8fc58bf7881a9ebc475" {
				t.Fatal("retained model, recipe or producer differs")
			}
			var body planBody
			readRetainedEvidence(t, store, value.Plan, &body)
			var suite MultipleChoiceSuite
			readRetainedEvidence(t, store, body.CaseProfile, &suite)
			var execution ExecutionPolicy
			readRetainedEvidence(t, store, body.Execution, &execution)
			if execution.Prompting != PromptingChatTemplate {
				t.Fatal("generated evidence replaced by a different execution protocol")
			}
			compiled, err := CompileMultipleChoice(suite)
			if err != nil {
				t.Fatal(err)
			}
			bound, err := BindMultipleChoice(compiled, ExactAuthorities{ModelDefinition: value.ModelDefinition, RuntimeRecipe: value.Recipe, Environment: value.Environment, CodeCommit: value.CodeCommit, Execution: execution})
			if err != nil || bound.Identity() != value.Plan || compiled.identity != body.CaseProfile {
				t.Fatalf("retained generated protocol no longer binds exactly: %v", err)
			}
			var stored struct {
				GroupedChoiceReport
				Categories []AccuracyGroup `json:"categories"`
			}
			readRetainedEvidence(t, store, value.Report, &stored)
			report := stored.GroupedChoiceReport
			if cell.name == "mmlu-pro" {
				if len(report.Groups) != 0 {
					t.Fatal("MMLU-Pro report has two group authorities")
				}
				report.Groups = stored.Categories
			} else if len(stored.Categories) != 0 {
				t.Fatal("unexpected MMLU-Pro category authority")
			}
			if report.Plan != value.Plan || report.Dataset != value.Dataset || len(report.Groups) != cell.groups || len(suite.Cases) != cell.cases {
				t.Fatalf("report authority or denominator differs: cases=%d groups=%d", len(suite.Cases), len(report.Groups))
			}
			if err := checkRetainedGeneratedChoices(suite.Cases, report, cell.correct); err != nil {
				t.Fatal(err)
			}
			metric := slices.IndexFunc(value.Metrics, func(metric runrecord.Metric) bool { return metric.Name == "accuracy" })
			if metric < 0 || value.Metrics[metric].Value != report.Accuracy {
				t.Fatal("retained aggregate differs from the scored answers")
			}
			total += cell.cases
			t.Logf("%d/%d retained generated answers; original plan and current extraction match", cell.correct, cell.cases)
			for _, mutation := range []string{"omitted", "duplicate", "target", "raw-answer", "likelihood", "aggregate"} {
				t.Run(mutation, func(t *testing.T) {
					bad := report
					bad.Observations = slices.Clone(report.Observations)
					switch mutation {
					case "omitted":
						bad.Observations = bad.Observations[1:]
					case "duplicate":
						bad.Observations[1] = bad.Observations[0]
					case "target":
						bad.Observations[0].Answer = -1
					case "raw-answer":
						letters := chatChoiceLetters(suite.Cases[0])
						index := (bad.Observations[0].Selected + 1) % len(letters)
						bad.Observations[0].Raw = letters[index]
					case "likelihood":
						bad.Observations[0].Scores = []sequencescore.Score{{LogProbability: -1, Tokens: 1}}
					case "aggregate":
						bad.Accuracy = -1
					}
					if checkRetainedGeneratedChoices(suite.Cases, bad, cell.correct) == nil {
						t.Fatal("accepted changed retained evidence")
					}
				})
			}
			if len(report.Groups) != 0 {
				t.Run("group-total", func(t *testing.T) {
					bad := report
					bad.Groups = slices.Clone(report.Groups)
					bad.Groups[0].Total++
					if checkRetainedGeneratedChoices(suite.Cases, bad, cell.correct) == nil {
						t.Fatal("accepted changed group denominator")
					}
				})
			}
		})
	}
	t.Logf("%d answers reused across four benchmark protocols; no model loads, likelihood relabeling or new quality floor", total)
}

func checkRetainedGeneratedChoices(cases []MultipleChoiceCase, report GroupedChoiceReport, wantCorrect int) error {
	if len(cases) == 0 || len(report.Observations) != len(cases) {
		return fmt.Errorf("retained choices: incomplete case denominator")
	}
	correct := 0
	seen := map[string]bool{}
	for index, c := range cases {
		row := report.Observations[index]
		if seen[row.Name] || row.Name != c.Name || row.Answer != c.Answer || row.Tied || len(row.Values) != 0 || len(row.Scores) != 0 {
			return fmt.Errorf("retained choices: case %d identity, target or generated method differs", index)
		}
		seen[row.Name] = true
		_, letters, err := chatChoicePrompt(c)
		if err != nil {
			return err
		}
		selected := unmatchedChoice
		if value, matched := matchChoiceLetter(row.Raw, letters); matched {
			selected = value
		}
		if selected != row.Selected {
			return fmt.Errorf("retained choices: %s extraction differs", row.Name)
		}
		if selected == c.Answer {
			correct++
		}
	}
	if correct != wantCorrect || report.Accuracy != float64(correct)/float64(len(cases)) {
		return fmt.Errorf("retained choices: aggregate differs: correct=%d expected=%d", correct, wantCorrect)
	}
	if len(report.Groups) != 0 {
		groups := map[string]bool{}
		var groupCorrect, groupTotal uint64
		for _, group := range report.Groups {
			if group.Name == "" || groups[group.Name] || group.Total == 0 || group.Correct > group.Total || group.Accuracy != float64(group.Correct)/float64(group.Total) {
				return fmt.Errorf("retained choices: group authority differs")
			}
			groups[group.Name] = true
			groupCorrect += group.Correct
			groupTotal += group.Total
		}
		if groupCorrect != uint64(correct) || groupTotal != uint64(len(cases)) {
			return fmt.Errorf("retained choices: group totals differ")
		}
	}
	return nil
}
