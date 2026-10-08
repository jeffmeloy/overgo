package trainingworkflow

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

// TestChangeProposalRecordIsTheJudgmentItNames reads the record of the
// change proposed from the installed text model's corpus. The judgment
// document the record carries is the judgment it names: it canonicalizes to
// that identity, so no figure in it was typed. It judges the corpus the
// corpus record names; it scores the proposal beside the best run, the mean
// and a random direction; and what it records is a refusal, the proposal
// below the best control and its bounds not above zero.
func TestChangeProposalRecordIsTheJudgmentItNames(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	read := func(name string) map[string]string {
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
	record := read("change-proposal-qwen25-05b.txt")
	named, err := artifact.ParseID(record["judgment"])
	if err != nil {
		t.Fatal(err)
	}
	var judgment ChangeJudgment
	if err := strictjson.DecodeBytes([]byte(record["document"]), &judgment); err != nil {
		t.Fatal(err)
	}
	canonical, err := changeJudgmentCodec.New(judgment)
	if err != nil || canonical.ID != named {
		t.Fatalf("the record's document is judgment %s (%v), and the record names %s", canonical.ID, err, named)
	}
	corpus, _, _ := strings.Cut(record["corpus"], " ")
	if corpus != read("change-corpus-qwen25-05b.txt")["corpus"] || corpus != canonical.Corpus.String() {
		t.Fatalf("the judgment is of corpus %s, the record says %s and the corpus record another", canonical.Corpus, corpus)
	}
	scores := map[string]float64{}
	for _, candidate := range canonical.Candidates {
		scores[candidate.Name] = candidate.Score
	}
	for _, name := range []string{changeProposed, changeBestRun, changeMean, changeRandom} {
		if !(scores[name] > canonical.Base) {
			t.Fatalf("the judgment scores %s at %v against a base of %v", name, scores[name], canonical.Base)
		}
	}
	if canonical.Kept || canonical.Lower > 0 || canonical.Difference != scores[changeProposed]-scores[canonical.Control] || !(canonical.Difference < 0) {
		t.Fatalf("the record is of a refusal, and the judgment reads kept=%t with the proposal %v from %s, between %v and %v", canonical.Kept, canonical.Difference, canonical.Control, canonical.Lower, canonical.Upper)
	}
	t.Logf("judgment %s: proposed %.4f, %s %.4f, random %.4f; rank %d explains %.3f", named, scores[changeProposed], canonical.Control, scores[canonical.Control], scores[changeRandom], canonical.Rank, canonical.Explained)
}
