package trainingworkflow

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

// TestEvidenceConditioningRecordIsOfItsJudgments reads the record of the
// comparison run on the installed text model's corpus,
// docs/verification/evidence-conditioning-qwen25-05b.txt, and holds each arm's
// line to the judgment it names: the document on the line is the judgment of
// that identity, of the corpus the proposal record names, shown as many
// verdicts as the line says and judged on as many of its runs. The two arms
// shown verdicts score the proposal the same code makes without them, and
// that candidate is the first arm's proposal, weight for weight and reading
// for reading; what each says the evidence moved is its own proposal's score
// less that one's. Nothing is run: the record's figures are the documents'.
func TestEvidenceConditioningRecordIsOfItsJudgments(t *testing.T) {
	t.Parallel()
	read := func(name string) map[string]string {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", name))
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
	record := read("evidence-conditioning-qwen25-05b.txt")
	corpus, _, _ := strings.Cut(read("change-corpus-qwen25-05b.txt")["corpus"], " ")
	arms := map[string]ChangeJudgment{}
	candidates := map[string]map[string]ChangeCandidate{}
	for arm, shown := range map[string][2]int{"none": {0, 0}, "now": {32, 24}, "cutoff": {24, 16}} {
		named, err := artifact.ParseID(record["judgment-"+arm])
		if err != nil {
			t.Fatalf("%s: %v", arm, err)
		}
		var judgment ChangeJudgment
		if err := strictjson.DecodeBytes([]byte(record["document-"+arm]), &judgment); err != nil {
			t.Fatalf("%s: %v", arm, err)
		}
		canonical, err := changeJudgmentCodec.New(judgment)
		if err != nil || canonical.ID != named {
			t.Fatalf("%s: the record's document is judgment %s (%v), and the record names %s", arm, canonical.ID, err, named)
		}
		if canonical.Corpus.String() != corpus || len(canonical.Shown) != shown[0] || canonical.Used != shown[1] {
			t.Fatalf("%s: the judgment is of corpus %s, shown %d verdicts that judged %d of its runs; want corpus %s, %d and %d", arm, canonical.Corpus, len(canonical.Shown), canonical.Used, corpus, shown[0], shown[1])
		}
		arms[arm], candidates[arm] = canonical, map[string]ChangeCandidate{}
		for _, candidate := range canonical.Candidates {
			candidates[arm][candidate.Name] = candidate
		}
	}
	plain := candidates["none"][changeProposed]
	if _, scored := candidates["none"][changeUnconditioned]; scored || arms["none"].EvidenceDifference != 0 {
		t.Fatal("the arm shown nothing scores a proposal without evidence beside its own")
	}
	for _, arm := range []string{"now", "cutoff"} {
		without, proposed := candidates[arm][changeUnconditioned], candidates[arm][changeProposed]
		if !slices.Equal(without.Weights, plain.Weights) || without.Report != plain.Report || without.Score != plain.Score {
			t.Fatalf("%s: the proposal without evidence is not the proposal of the arm shown nothing", arm)
		}
		if arms[arm].EvidenceDifference != proposed.Score-without.Score || arms[arm].EvidenceLower > arms[arm].EvidenceUpper {
			t.Fatalf("%s: the judgment says the evidence moved the proposal %v in [%v, %v], and the two scored %v and %v", arm, arms[arm].EvidenceDifference, arms[arm].EvidenceLower, arms[arm].EvidenceUpper, proposed.Score, without.Score)
		}
	}
	// The cut-off arm was shown what the store held earlier: every verdict it was shown, the other arm was shown too.
	for _, verdict := range arms["cutoff"].Shown {
		if !slices.Contains(arms["now"].Shown, verdict) {
			t.Fatalf("the arm shown the store as it stood was shown %s, which the store as it stands does not hold", verdict)
		}
	}
}
