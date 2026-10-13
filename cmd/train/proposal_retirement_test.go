package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestProposalRetirementPreservesTrainingAndEvidence holds the training
// command to having no change proposer, and to what stays without one.
//
// The proposer's files, and the block resampling nothing else called, are not
// in the tree. The command refuses -propose as a flag it does not define, and
// its usage lists none of the proposal, evidence and corpus flags while it
// lists the flags of an objective run, of resuming and of the pilot.
//
// A run through the command's own flags trains one step from the base model,
// and a second run resumes that run's checkpoint and ends one step further.
//
// The three records of the experiment read as they did. The proposal record
// and the evidence record judge the corpus the corpus record names, each
// judgment they name is an identity, and each judgment's document says the
// proposal was not kept and that the control it lost to was the plain mean of
// the runs.
//
// Each child is this test binary run as the command on the host, its device
// trainer the stand-in.
func TestProposalRetirementPreservesTrainingAndEvidence(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	// command runs the training command over the arguments and returns the
	// status it ended with and what it wrote to standard error. A flag the
	// command does not define is refused by the flag parser, which writes there
	// and never reaches the command's error file.
	command := func(t *testing.T, arguments ...string) (int, string) {
		t.Helper()
		var said bytes.Buffer
		receipt, err := processcontrol.Run(t.Context(), processcontrol.Command{
			Path: os.Args[0], Args: arguments, Stderr: &said, Env: append(os.Environ(), commandChildEnvironment+"="+childStandIn),
		})
		if err != nil {
			t.Fatal(err)
		}
		return receipt.ExitCode, said.String()
	}

	t.Run("the proposer's files are not in the tree", func(t *testing.T) {
		t.Parallel()
		for _, path := range []string{
			"internal/trainingworkflow/change_proposal.go", "internal/trainingworkflow/change_geometry.go",
			"internal/trainingworkflow/change_corpus.go", "internal/tensorstats/resampling.go",
		} {
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(path))); !os.IsNotExist(err) {
				t.Errorf("%s still exists (%v)", path, err)
			}
		}
	})

	t.Run("the command has no proposal, evidence or corpus flag", func(t *testing.T) {
		t.Parallel()
		// The store is named so that a command that took the flag would open
		// one under the test's directory, not one beside this package.
		status, said := command(t, "-store", filepath.Join(t.TempDir(), "store"), "-propose", "named")
		if status != 2 || !strings.Contains(said, "flag provided but not defined: -propose") {
			t.Errorf("the training command given -propose ended with status %d and said %q, want it refused as a flag the command does not define", status, said)
		}
		status, usage := command(t, "-h")
		if status != 0 {
			t.Fatalf("the training command given -h ended with status %d and said %q, want its usage", status, usage)
		}
		// A flag's usage opens with two spaces, a dash and its whole name, so
		// -propose is not read out of -proposal, nor -resume out of -resume-run.
		listed := map[string]bool{}
		for line := range strings.Lines(usage) {
			if rest, found := strings.CutPrefix(line, "  -"); found {
				if words := strings.Fields(rest); len(words) != 0 {
					listed[words[0]] = true
				}
			}
		}
		for _, name := range []string{"propose", "pool", "confirm", "candidate", "proposal", "evidence", "evidence-to-sequence", "evidence-max-bytes", "runs"} {
			if listed[name] {
				t.Errorf("the training command's usage lists -%s, a flag of the change proposer or of its corpus", name)
			}
		}
		for _, name := range []string{"objective", "resume", "resume-run", "save-every", "order-seed", "pilot"} {
			if !listed[name] {
				t.Errorf("the training command's usage does not list -%s, which training still takes", name)
			}
		}
	})

	t.Run("a run trains and a second run resumes its checkpoint", func(t *testing.T) {
		t.Parallel()
		directory := t.TempDir()
		base, dataset, storePath := writeServingFixture(t, directory), filepath.Join(directory, "dataset.txt"), filepath.Join(directory, "store")
		if err := os.WriteFile(dataset, bytes.Repeat([]byte("a b c d "), 64), 0o600); err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.Open(storePath)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := modelrecipe.PublishArchitectureProfileCatalog(t.Context(), store); err != nil {
			t.Fatal(err)
		}
		recipeID, err := trainingworkflow.BootstrapTokenRecipe(t.Context(), store, base, dataset)
		if err != nil {
			t.Fatal(err)
		}
		// The command writes through a handle of its own.
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		first, resumed := filepath.Join(directory, "first"), filepath.Join(directory, "resumed")
		shared := []string{"-store", storePath, "-recipe", recipeID.String(), "-dataset", dataset, "-steps", "1", "-seq", "16", "-host"}
		for _, run := range []struct {
			what      string
			arguments []string
			out       string
			step      int
		}{
			{"a run from the base model", []string{"-model", base, "-out", first}, first, 1},
			{"a run resumed from that run's checkpoint", []string{"-resume", first, "-out", resumed}, resumed, 2},
		} {
			if status, said := command(t, append(slices.Clone(shared), run.arguments...)...); status != 0 {
				t.Fatalf("%s ended with status %d and said %q, want it to train", run.what, status, said)
			}
			checkpoint, err := trainingprogram.LoadCheckpoint(run.out)
			if err != nil || checkpoint.Optimizer.Step != run.step {
				t.Fatalf("%s left a checkpoint at optimizer step %d (%v), want step %d", run.what, checkpoint.Optimizer.Step, err, run.step)
			}
		}
	})

	t.Run("the three records read as they did", func(t *testing.T) {
		t.Parallel()
		// read is a record's lines of a key, a colon and a value, by key.
		read := func(name string) map[string]string {
			t.Helper()
			raw, err := os.ReadFile(filepath.Join(root, "docs", "verification", name))
			if err != nil {
				t.Fatal(err)
			}
			fields := map[string]string{}
			for line := range strings.Lines(string(raw)) {
				if key, value, found := strings.Cut(strings.TrimSpace(line), ": "); found && !strings.Contains(key, " ") {
					fields[key] = value
				}
			}
			return fields
		}
		corpus := read("change-corpus-qwen25-05b.txt")["corpus"]
		if identity, err := artifact.ParseID(corpus); err != nil || identity.Kind() != artifact.KindEvidence {
			t.Fatalf("the corpus record names corpus %q (%v), want the identity of a corpus", corpus, err)
		}
		for name, judgments := range map[string][]string{
			"change-proposal-qwen25-05b.txt":       {""},
			"evidence-conditioning-qwen25-05b.txt": {"-none", "-now", "-cutoff"},
		} {
			record := read(name)
			if judged, _, _ := strings.Cut(record["corpus"], " "); judged != corpus {
				t.Errorf("%s judges corpus %q, and the corpus record names %q", name, judged, corpus)
			}
			for _, arm := range judgments {
				if identity, err := artifact.ParseID(record["judgment"+arm]); err != nil || identity.Kind() != artifact.KindEvaluation {
					t.Errorf("%s names judgment%s %q (%v), want the identity of a judgment", name, arm, record["judgment"+arm], err)
				}
				var document struct {
					Corpus  string `json:"corpus"`
					Control string `json:"control"`
					Kept    *bool  `json:"kept"`
				}
				if err := json.Unmarshal([]byte(record["document"+arm]), &document); err != nil {
					t.Errorf("%s: document%s does not read: %v", name, arm, err)
					continue
				}
				if document.Corpus != corpus || document.Control != "mean" || document.Kept == nil || *document.Kept {
					t.Errorf("%s: document%s is of corpus %q, with control %q and kept %v; want corpus %q, the mean as the control the proposal lost to, and the proposal not kept",
						name, arm, document.Corpus, document.Control, document.Kept, corpus)
				}
			}
		}
	})
}
