package trainingworkflow

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

// What the three records of the change proposer's experiment carry, under
// docs/verification: a line for each fine-tune of the corpus it read, and the
// judgment of a change proposed from that corpus. Nothing writes either now.
// The tests of those records read them here. A judgment's identity is taken
// over these fields under this media type and schema, so a field or a tag
// that moves makes a record's document another judgment than the one the
// record names.

const (
	changeJudgmentMediaType = "application/vnd.overgo.change-judgment+json"
	changeJudgmentSchema    = "overgo/change-judgment/v1"

	// The candidates a judgment scores: the one the code of the changes
	// proposed, and the plain controls it had to beat.
	changeProposed = "proposed"
	changeBestRun  = "best-run"
	changeMean     = "mean"
	changeRandom   = "random-direction"
	// changeUnconditioned is the proposal the same code made of the gains read
	// on the pool alone: the control a proposer shown past verdicts was held to.
	changeUnconditioned = "without-evidence"
)

// weightChange is one bounded fine-tune as the change it made to its base
// model: the checkpoint the run published, which with the base's weights is
// the change; the order seed the run read its training records by, which is
// all that sets it apart from the corpus's other runs; and the held-out
// verdict the change earned, with the metric it read for the base model and
// for the changed one.
type weightChange struct {
	Order       uint64      `json:"order"`
	Observation artifact.ID `json:"observation"`
	Trained     artifact.ID `json:"trained"`
	Verdict     artifact.ID `json:"verdict"`
	Base        float64     `json:"base"`
	Changed     float64     `json:"changed"`
}

// changeCandidate is one candidate change of a corpus's base model: the base
// plus a weighted sum of the corpus's changes, the weights in the corpus's own
// order, with the reading that scored it on the selection part of its pool
// and the mean it read there, of its documents' token log-likelihoods. A
// judgment recorded before proposals were made on a pool names instead the
// report that scored the candidate on the held-out records its corpus's runs
// were judged on, and the corpus's metric.
type changeCandidate struct {
	Name    string      `json:"name"`
	Weights []float64   `json:"weights"`
	Report  artifact.ID `json:"report"`
	Score   float64     `json:"score"`
}

// changeJudgment is a change proposed from a corpus, judged against plain
// controls. The proposer was a linear code of the changes: the directions the
// runs' changes differ along most, as many of them as best predicted the gain
// of a run left out, and the proposed change was the mean change moved along
// the direction that code said gain rises in, as far as a typical run sits
// from the mean. The controls were the best single run, the mean of all runs,
// the mean of the runs that improved, and the mean moved the same distance in
// a random direction. The proposal was kept only when its score exceeded the
// best control's across the middle of the redrawn documents it was scored on.
type changeJudgment struct {
	ID         artifact.ID `json:"-"`
	Version    uint16      `json:"version"`
	Corpus     artifact.ID `json:"corpus"`
	Resampling artifact.ID `json:"resampling"`
	// Rank is how many directions the code kept, and Explained the share of
	// the runs' gain differences it predicted for a run left out: at or below
	// zero, the changes say nothing of the gain that the mean gain does not.
	Rank      int     `json:"rank"`
	Explained float64 `json:"explained"`
	// GainSpread is the standard deviation of the runs' gains, and GainNoise
	// the standard error of one run's gain over the documents it was read on:
	// a spread no larger than the noise leaves no gain difference to explain.
	GainSpread float64 `json:"gain_spread"`
	GainNoise  float64 `json:"gain_noise"`
	// Base is the base model's score, read again beside the candidates.
	Base       float64           `json:"base"`
	Candidates []changeCandidate `json:"candidates"`
	// Control names the control with the best score. Difference is the
	// proposal's score less that control's, and Lower and Upper bound the
	// middle of that difference over the redrawn records.
	Control    string  `json:"control"`
	Difference float64 `json:"difference"`
	Lower      float64 `json:"lower"`
	Upper      float64 `json:"upper"`
	Kept       bool    `json:"kept"`
	// Shown names the past held-out verdicts the proposer was shown of what
	// was tried before it, and Used how many of the corpus's runs they judged.
	// When they judged enough to fit, the code was fitted to those runs'
	// recorded gains beside the gains read on the pool, the proposal the code
	// made of the pool's gains alone was scored among the candidates as
	// without-evidence, and EvidenceDifference is the proposal's score less
	// that one's, with EvidenceLower and EvidenceUpper bounding the middle of
	// it over the redrawn documents. A proposer shown nothing recorded none of
	// these.
	Shown              []artifact.ID `json:"shown,omitempty"`
	Used               int           `json:"used,omitzero"`
	EvidenceDifference float64       `json:"evidence_difference,omitzero"`
	EvidenceLower      float64       `json:"evidence_lower,omitzero"`
	EvidenceUpper      float64       `json:"evidence_upper,omitzero"`
}

var changeJudgmentCodec = artifact.JSONDocumentCodec(
	"change judgment", artifact.KindEvaluation, changeJudgmentMediaType, changeJudgmentSchema,
	canonicalizeChangeJudgment,
	func(value changeJudgment) artifact.ID { return value.ID },
	func(value *changeJudgment, id artifact.ID) { value.ID = id },
	func(value changeJudgment) changeJudgment {
		value.Shown = slices.Clone(value.Shown)
		value.Candidates = slices.Clone(value.Candidates)
		for index := range value.Candidates {
			value.Candidates[index].Weights = slices.Clone(value.Candidates[index].Weights)
		}
		return value
	},
)

// canonicalizeChangeJudgment refuses a judgment without its proposal, its
// control or their reports, and recomputes whether the proposal is kept from
// the recorded bounds.
func canonicalizeChangeJudgment(value *changeJudgment) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Corpus.Kind() != artifact.KindEvidence ||
		value.Resampling.Kind() != artifact.KindProfile || value.Rank <= 0 {
		return errors.New("training workflow: incomplete change judgment")
	}
	names := map[string]bool{}
	for _, candidate := range value.Candidates {
		if candidate.Name == "" || names[candidate.Name] || candidate.Report.Kind() != artifact.KindEvaluation || len(candidate.Weights) == 0 {
			return errors.New("training workflow: a change candidate lacks a name of its own, its weights or its report")
		}
		names[candidate.Name] = true
	}
	if !names[changeProposed] || !names[value.Control] || value.Control == changeProposed {
		return errors.New("training workflow: a change judgment needs its proposal and the control it is judged against")
	}
	value.Kept = value.Lower > 0
	return nil
}

// TestChangeCorpusRecordListsEachRunOnce reads the record of the corpus
// collected from the installed text model: one line a run, each a change with
// its verdict. The record names its corpus by identity. Each line names a
// session, and a checkpoint and a verdict no other line names. The order seeds
// count up from one with none repeated, every run was judged against one base
// reading, and every run improved on it.
func TestChangeCorpusRecordListsEachRunOnce(t *testing.T) {
	t.Parallel()
	record, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", "change-corpus-qwen25-05b.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var named artifact.ID
	var changes []weightChange
	for line := range strings.Lines(string(record)) {
		line = strings.TrimSpace(line)
		if id, found := strings.CutPrefix(line, "corpus: "); found {
			if named, err = artifact.ParseID(id); err != nil {
				t.Fatal(err)
			}
		}
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var change weightChange
		if err := strictjson.DecodeBytes([]byte(line), &change); err != nil {
			t.Fatalf("record line %q: %v", line, err)
		}
		changes = append(changes, change)
	}
	if named.Kind() != artifact.KindEvidence || len(changes) == 0 {
		t.Fatalf("the record names corpus %s and %d runs", named, len(changes))
	}
	trained, verdicts := map[artifact.ID]bool{}, map[artifact.ID]bool{}
	for index, change := range changes {
		if change.Observation.Kind() != artifact.KindEvidence || change.Trained.Kind() != artifact.KindCheckpoint || change.Verdict.Kind() != artifact.KindEvaluation {
			t.Fatalf("run %d of the record names no session, checkpoint or verdict", index+1)
		}
		// A checkpoint or a verdict named twice is one change counted twice.
		if trained[change.Trained] || verdicts[change.Verdict] {
			t.Fatalf("run %d of the record names a checkpoint or a verdict that an earlier run names", index+1)
		}
		trained[change.Trained], verdicts[change.Verdict] = true, true
		if change.Order != uint64(index+1) || change.Base != changes[0].Base || !(change.Changed > change.Base) {
			t.Fatalf("run %d of the record has order %d and reads %v from a base of %v", index+1, change.Order, change.Changed, change.Base)
		}
	}
	t.Logf("corpus %s: %d runs from a base of %.4f", named, len(changes), changes[0].Base)
}
