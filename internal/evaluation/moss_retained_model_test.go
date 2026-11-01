//go:build models

package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const (
	// mossTrainingRecord is the retained reading of a MOSS adapter run, and
	// mossTrainingClaim the claim docs/TRAINING_COMPATIBILITY.md publishes from it.
	mossTrainingRecord = "docs/verification/moss-image-training-run.json"
	mossTrainingClaim  = "docs/verification/moss-image-training.json"
	// mossHeldoutRounding is how closely a recorded mean restates its own values.
	mossHeldoutRounding = 1e-9
)

// mossManifest pins one picture manifest under the data root's datasets.
type mossManifest struct {
	Path     string `json:"path"`
	SHA256   string `json:"sha256"`
	Pictures int    `json:"pictures"`
}

// mossSample pins the same request drawn by the base and by the trained model.
type mossSample struct {
	Prompt  string `json:"prompt"`
	Size    int    `json:"size"`
	Seed    int    `json:"seed"`
	Base    string `json:"base_sha256"`
	Trained string `json:"trained_sha256"`
}

// mossTrainingReading is one run: what ran, on what, and what held-out pictures said.
type mossTrainingReading struct {
	Model       artifact.ID `json:"model"`
	Commit      string      `json:"commit"`
	Recipe      artifact.ID `json:"recipe"`
	Checkpoint  artifact.ID `json:"checkpoint"`
	Observation artifact.ID `json:"session_observation"`
	Run         struct {
		Size         int     `json:"size"`
		Rank         int     `json:"rank"`
		Alpha        float64 `json:"alpha"`
		Batch        int     `json:"batch"`
		Steps        uint64  `json:"steps"`
		LearningRate float64 `json:"learning_rate"`
	} `json:"run"`
	Train   mossManifest `json:"train"`
	Heldout mossManifest `json:"heldout"`
	Reading struct {
		Pictures []string  `json:"pictures"`
		Paths    int       `json:"paths_per_picture"`
		Base     []float64 `json:"base"`
		Trained  []float64 `json:"trained"`
	} `json:"reading"`
	BaseMean    float64      `json:"base_mean"`
	TrainedMean float64      `json:"trained_mean"`
	Improved    int          `json:"improved"`
	Worse       int          `json:"worse"`
	BeatsBase   bool         `json:"beats_base"`
	SampleRoot  string       `json:"sample_root"`
	Samples     []mossSample `json:"samples"`
	Judgement   string       `json:"judgement"`
}

// TestMOSSRetainedEvidence holds each MOSS capability claim to what was measured.
func TestMOSSRetainedEvidence(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	t.Run("training", func(t *testing.T) {
		t.Parallel()
		checkMOSSTraining(t, root, roots, store)
	})
}

// checkMOSSTraining reads the retained run against its sources. The verdict is
// recomputed from the per-picture values, the split from the manifests, and the
// run's wall and peak from the session the store recorded, so the claim states
// what was measured, whichever way it came out.
func checkMOSSTraining(t *testing.T, root string, roots dataroot.Roots, store *overgodb.Store) {
	t.Helper()
	ctx := t.Context()
	var record mossTrainingReading
	if err := jsonfile.DecodeStrict(filepath.Join(root, mossTrainingRecord), &record); err != nil {
		t.Fatal(err)
	}
	if _, err := gitauthority.Query(ctx, root, "cat-file", "-e", record.Commit+"^{commit}"); err != nil {
		t.Fatalf("commit %s: %v", record.Commit, err)
	}
	trained, held := mossManifestSources(t, roots, record.Train), mossManifestSources(t, roots, record.Heldout)
	for name := range held {
		if trained[name] {
			t.Fatalf("source video %s is both trained on and held out", name)
		}
	}
	reading := record.Reading
	if len(reading.Pictures) != record.Heldout.Pictures || len(reading.Base) != len(reading.Pictures) || len(reading.Trained) != len(reading.Pictures) || reading.Paths == 0 {
		t.Fatalf("reading covers %d pictures on %d paths, the manifest holds %d", len(reading.Base), reading.Paths, record.Heldout.Pictures)
	}
	var base, adapted float64
	improved, worse := 0, 0
	for index := range reading.Base {
		base += reading.Base[index]
		adapted += reading.Trained[index]
		switch {
		case reading.Trained[index] < reading.Base[index]:
			improved++
		case reading.Trained[index] > reading.Base[index]:
			worse++
		}
	}
	base, adapted = base/float64(len(reading.Base)), adapted/float64(len(reading.Base))
	if math.Abs(base-record.BaseMean) > mossHeldoutRounding || math.Abs(adapted-record.TrainedMean) > mossHeldoutRounding || improved != record.Improved || worse != record.Worse {
		t.Fatalf("the record says %.6f to %.6f with %d better and %d worse; its values say %.6f to %.6f with %d and %d",
			record.BaseMean, record.TrainedMean, record.Improved, record.Worse, base, adapted, improved, worse)
	}
	if record.BeatsBase != (adapted < base) {
		t.Fatalf("the record says beats_base %t on %.6f against %.6f", record.BeatsBase, adapted, base)
	}
	observation, err := runrecord.RequireServingObservation(ctx, store, record.Observation)
	if err != nil {
		t.Fatal(err)
	}
	if observation.Task != recipe.TaskTraining || observation.Outcome != runrecord.OutcomeSucceeded || observation.Model != record.Model || observation.Recipe != record.Recipe || !observation.Environment.Valid() {
		t.Fatalf("session %+v does not record this run", observation)
	}
	if observation.Resources.PeakDeviceBytes == 0 {
		t.Fatal("the session measured no device peak")
	}
	if _, found, err := store.Artifact(ctx, record.Checkpoint); err != nil || !found {
		t.Fatalf("checkpoint %s is not retained: %v", record.Checkpoint, err)
	}
	checkMOSSClaim(t, ctx, root, record, observation)
	for _, sample := range record.Samples {
		for _, want := range []string{sample.Base, sample.Trained} {
			data, err := os.ReadFile(filepath.Join(filepath.Dir(roots.Datasets), filepath.FromSlash(record.SampleRoot), want+".png"))
			if err != nil {
				t.Fatal(err)
			}
			if digest := sha256.Sum256(data); hex.EncodeToString(digest[:]) != want {
				t.Fatalf("sample %s holds other bytes", want)
			}
		}
	}
	if len(record.Samples) == 0 || strings.TrimSpace(record.Judgement) == "" {
		t.Fatal("the record holds no pictures, or no one looked at them")
	}
	t.Logf("held-out loss %.5f to %.5f over %d pictures: %d better, %d worse", base, adapted, len(reading.Base), improved, worse)
}

// checkMOSSClaim holds the published claim to the run and the session it names.
func checkMOSSClaim(t *testing.T, ctx context.Context, root string, record mossTrainingReading, observation runrecord.ServingObservation) {
	t.Helper()
	var specification struct {
		Model         artifact.ID                 `json:"model"`
		EvidenceFiles []string                    `json:"evidence_files"`
		Claims        []runrecord.CapabilityClaim `json:"claims"`
	}
	if err := jsonfile.Decode(filepath.Join(root, mossTrainingClaim), &specification); err != nil {
		t.Fatal(err)
	}
	if specification.Model != record.Model || len(specification.Claims) != 1 || !slices.Contains(specification.EvidenceFiles, mossTrainingRecord) {
		t.Fatalf("specification %+v does not rest on this run", specification)
	}
	evidence, err := artifact.IdentifyFile(artifact.KindEvidence, filepath.Join(root, mossTrainingRecord))
	if err != nil {
		t.Fatal(err)
	}
	claim := specification.Claims[0]
	if claim.Commit != record.Commit || claim.SpanSteps != record.Run.Steps || claim.WallNS != observation.MeasuredNS || claim.PeakDeviceBytes != observation.Resources.PeakDeviceBytes || !slices.Contains(claim.Evidence, evidence) {
		t.Fatalf("claim %+v does not state the run: steps %d, wall %d, peak %d, evidence %s", claim, record.Run.Steps, observation.MeasuredNS, observation.Resources.PeakDeviceBytes, evidence)
	}
}

// mossManifestSources checks a manifest's bytes and count, and names the source
// videos its pictures came from.
func mossManifestSources(t *testing.T, roots dataroot.Roots, manifest mossManifest) map[string]bool {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(roots.Datasets, filepath.FromSlash(manifest.Path)))
	if err != nil {
		t.Fatal(err)
	}
	if digest := sha256.Sum256(data); hex.EncodeToString(digest[:]) != manifest.SHA256 {
		t.Fatalf("manifest %s holds other bytes", manifest.Path)
	}
	sources := map[string]bool{}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != manifest.Pictures {
		t.Fatalf("manifest %s holds %d pictures, the record says %d", manifest.Path, len(lines), manifest.Pictures)
	}
	for _, line := range lines {
		path, _, _ := strings.Cut(line, "\t")
		name, _, _ := strings.Cut(filepath.Base(filepath.FromSlash(path)), "-Scene-")
		sources[name] = true
	}
	return sources
}
