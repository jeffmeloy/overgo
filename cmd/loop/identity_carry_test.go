package main

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// carryMeasurement is the command the measurement run's launch starts in the
// test's checkout, in place of cmd/validate: it evaluates one suite of twelve
// cases with each of two models, a text runner's and a thought bank's, as
// cmd/evaluate evaluates a benchmark, over the store it is given, and prints
// how many cases each was put. Each model answers by rule and counts what it
// is asked, so the count is of cases put to a model and of nothing read back.
const carryMeasurement = `// Command validate stands in for the validation command in a test of the
// measurement run's launch (cmd/loop TestMeasurementRunsNoCaseAgainAfterAnUnrelatedCommit).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/thoughtbank"
	"overgo/internal/tokenizer"
)

type counting struct{ asked int }

func (*counting) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("the stand-in scores likelihoods only")
}

func (model *counting) ScoreContinuations(_ context.Context, _ string, candidates []string) ([]sequencescore.Score, error) {
	model.asked++
	scores := make([]sequencescore.Score, len(candidates))
	for index := range candidates {
		scores[index] = sequencescore.Score{LogProbability: -float64(index + 1), Tokens: 1}
	}
	return scores, nil
}

func measure(repository string) error {
	ctx := context.Background()
	commit, err := runrecord.VerifyingCommit(".")
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	if err != nil {
		return err
	}
	suite := evaluation.MultipleChoiceSuite{
		Kind: evaluation.MultipleChoiceKind, Schema: "fixture/carry", Source: "store/carry",
		Normalization: sequencescore.NormalizationSum, Aggregation: evaluation.AggregationAccuracy,
	}
	for index := range 12 {
		suite.Cases = append(suite.Cases, evaluation.MultipleChoiceCase{
			Name: fmt.Sprintf("case-%02d", index), Prompt: fmt.Sprintf("Question %d\nAnswer:", index), Candidates: []string{" A", " B"}, Answer: index % 2,
		})
	}
	source, err := json.Marshal(suite)
	if err != nil {
		return err
	}
	for _, capability := range []struct {
		name string
		told func(*evaluation.Campaign) *evaluation.Campaign
	}{
		{"text", evaluation.AnsweredBy[inference.Runner]},
		{"thought", evaluation.AnsweredBy[thoughtbank.Generator]},
	} {
		var identity modelrecipe.ProgramIdentity
		for kind, id := range map[artifact.Kind]*artifact.ID{artifact.KindModel: &identity.Model, artifact.KindModelDefinition: &identity.Definition, artifact.KindRecipe: &identity.Recipe} {
			if *id, err = artifact.JSONID(kind, "carry/"+capability.name); err != nil {
				return err
			}
			if _, found, err := store.Artifact(ctx, *id); err != nil {
				return err
			} else if !found {
				if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/" + id.String(), Artifacts: []artifact.Descriptor{{ID: *id}}}); err != nil {
					return err
				}
			}
		}
		model := &counting{}
		campaign, err := evaluation.NewIsolatedCampaign(store, model, identity, environment, commit)
		if err != nil {
			return err
		}
		compiled, err := evaluation.CompileSuite(source, capability.told(campaign).Authorities())
		if err != nil {
			return err
		}
		if _, err := campaign.Evaluate(ctx, compiled); err != nil {
			return err
		}
		fmt.Printf("put %s %d of %d at %s\n", capability.name, model.asked, len(suite.Cases), commit)
	}
	return nil
}

func main() {
	repository := flag.String("repo", "", "the store the stand-in evaluates over")
	flag.Parse()
	if err := measure(*repository); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
`

// carryTree is what of the repository the test's checkout holds: the module,
// every package under internal, which the capability catalog and an
// evaluation reach, the kernel sources an identity counts, and the command
// whose sources an evaluation's identity counts.
var carryTree = []string{"go.mod", "go.sum", "internal", "kernels", "cmd/evaluate"}

// TestMeasurementRunsNoCaseAgainAfterAnUnrelatedCommit holds a measurement
// run to what its code has not answered, through the launch every phase of
// the run starts its command with (validateTo): the checkout's identities
// counted and linked into a go run, a process of its own each time, over one
// store.
//
// The checkout is this tree as it stands, committed in a repository of its
// own, with a stand-in for the command the launch starts (carryMeasurement).
// The first measurement puts every case to each of two models. The checkout
// then commits a change outside every identity, a note and an edit to the
// stand-in command itself, and the measurement at that commit puts no case to
// either. It then commits a change to a source of the thought bank's package,
// which only the thought bank's identity holds, and the measurement at that
// commit puts every case to the thought bank's model and none to the text
// runner's.
func TestMeasurementRunsNoCaseAgainAfterAnUnrelatedCommit(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	checkout, store := filepath.Join(t.TempDir(), "checkout"), filepath.Join(t.TempDir(), "store")
	git := func(arguments ...string) string { return gitFixture(t, checkout, arguments...) }
	write := func(name string, content []byte) {
		t.Helper()
		path := filepath.Join(checkout, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// The files a commit of this tree would hold, without its tests, which no build of a command reads.
	listed := gitFixture(t, root, append([]string{"ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"}, carryTree...)...)
	var thought string
	for name := range strings.SplitSeq(listed, "\x00") {
		if name == "" || strings.HasSuffix(name, "_test.go") {
			continue
		}
		content, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		write(name, content)
		if thought == "" && strings.HasPrefix(name, "internal/thoughtbank/") && strings.HasSuffix(name, ".go") {
			thought = name
		}
	}
	if thought == "" {
		t.Fatal("the tree holds no source of the thought bank's package to change")
	}
	write("cmd/validate/main.go", []byte(carryMeasurement))
	if err := os.MkdirAll(store, 0o755); err != nil {
		t.Fatal(err)
	}
	git("init", "-q", "-b", "master")
	git("config", "core.autocrlf", "false")
	git("add", "-A")
	git("commit", "-q", "-m", "measured")

	// A build's cache is keyed by its directory unless paths are trimmed; trimmed, a checkout in a new
	// temporary directory builds from the cache the test's run before it filled.
	t.Setenv("GOFLAGS", strings.TrimSpace(os.Getenv("GOFLAGS")+" -trimpath"))
	put := regexp.MustCompile(`(?m)^put (\w+) (\d+) of (\d+) at ([0-9a-f]+)$`)
	// measure launches the stand-in at the checkout's commit as the run launches cmd/validate, and returns how many cases each model was put.
	measure := func() map[string]int {
		t.Helper()
		var output bytes.Buffer
		if err := validateTo(&output, checkout, store, nil); err != nil {
			t.Fatalf("the launch at %s: %v\n%s", git("rev-parse", "HEAD"), err, output.String())
		}
		asked := map[string]int{}
		for _, line := range put.FindAllStringSubmatch(output.String(), -1) {
			count, err := strconv.Atoi(line[2])
			if err != nil || line[3] != "12" || line[4] != git("rev-parse", "HEAD") {
				t.Fatalf("the stand-in printed %q at commit %s: %v", line[0], git("rev-parse", "HEAD"), err)
			}
			asked[line[1]] = count
		}
		if len(asked) != 2 {
			t.Fatalf("the stand-in reported %v; want a count for each of its two models:\n%s", asked, output.String())
		}
		t.Logf("at commit %.12s the models were put %v of their 12 cases", git("rev-parse", "HEAD"), asked)
		return asked
	}
	commits := []string{git("rev-parse", "HEAD")}
	change := func(message string, files map[string]string) {
		t.Helper()
		for name, added := range files {
			content, err := os.ReadFile(filepath.Join(checkout, filepath.FromSlash(name)))
			if err != nil && !os.IsNotExist(err) {
				t.Fatal(err)
			}
			write(name, append(content, added...))
		}
		git("add", "-A")
		git("commit", "-q", "-m", message)
		commits = append(commits, git("rev-parse", "HEAD"))
	}

	if first := measure(); first["text"] != 12 || first["thought"] != 12 {
		t.Fatalf("the first measurement put %v cases to the models; want all 12 to each", first)
	}
	change("a change outside every identity", map[string]string{
		"docs/note.txt":        "a note no identity holds\n",
		"cmd/validate/main.go": "\n// The stand-in itself changed: it is built again and is no part of an identity.\n",
	})
	if second := measure(); second["text"] != 0 || second["thought"] != 0 {
		t.Fatalf("the measurement after an unrelated commit put %v cases to the models; want none to either", second)
	}
	change("a change inside the thought bank's identity", map[string]string{thought: "\n// A change to a source only the thought bank's identity holds.\n"})
	if third := measure(); third["text"] != 0 || third["thought"] != 12 {
		t.Fatalf("the measurement after a commit to %s put %v cases to the models; want all 12 to the thought bank's and none to the text runner's", thought, third)
	}
	if distinct := slices.Compact(slices.Sorted(slices.Values(commits))); len(distinct) != 3 {
		t.Fatalf("the three measurements were of commits %v; want three commits", commits)
	}
}
