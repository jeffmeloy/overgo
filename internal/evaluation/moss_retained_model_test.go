//go:build models

package evaluation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const (
	// mossTrainingRecord is the retained reading of a MOSS adapter run, and
	// mossTrainingClaim the claim docs/TRAINING_COMPATIBILITY.md publishes from it.
	mossTrainingRecord = "docs/verification/moss-image-training-run.json"
	mossTrainingClaim  = "docs/verification/moss-image-training.json"
	// mossHeldoutRounding is how closely a recorded mean restates its own values.
	mossHeldoutRounding = 1e-9
	// mossParityRecordPath and mossResourceRecordPath hold the retained comparisons
	// with the pinned upstream source.
	mossParityRecordPath   = "docs/verification/moss-image-parity.json"
	mossResourceRecordPath = "docs/verification/moss-image-resources.json"
	// mossParityPasses is the generations each case compares, the second holding
	// the first to repeatability.
	mossParityPasses = 2
	// mossPage is the page that states all of it for a person.
	mossPage = "docs/MOSS_IMAGE.md"
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
	testcheck.NoError(t, err)
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	testcheck.NoError(t, err)
	t.Cleanup(func() { store.Close() })
	t.Run("complete", func(t *testing.T) {
		t.Parallel()
		checkMOSSComplete(t, root, roots, store)
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
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, mossTrainingRecord), &record))
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
	testcheck.NoError(t, err)
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
			testcheck.NoError(t, err)
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
	testcheck.NoError(t, jsonfile.Decode(filepath.Join(root, mossTrainingClaim), &specification))
	if specification.Model != record.Model || len(specification.Claims) != 1 || !slices.Contains(specification.EvidenceFiles, mossTrainingRecord) {
		t.Fatalf("specification %+v does not rest on this run", specification)
	}
	evidence, err := artifact.IdentifyFile(artifact.KindEvidence, filepath.Join(root, mossTrainingRecord))
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
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

// mossParityCase pins one oracle and the native run compared with it.
type mossParityCase struct {
	Name       string      `json:"name"`
	Report     string      `json:"report"`
	Prompt     string      `json:"prompt"`
	Negative   string      `json:"negative_prompt"`
	Width      int         `json:"width"`
	Height     int         `json:"height"`
	Seed       int64       `json:"seed"`
	Oracle     artifact.ID `json:"oracle"`
	Boundaries int         `json:"boundaries"`
	WorstAbs   float64     `json:"worst_abs"`
	Image      artifact.ID `json:"image"`
}

// mossParityRecord is every full generation compared with the pinned upstream source.
type mossParityRecord struct {
	Model      artifact.ID      `json:"model"`
	Commit     string           `json:"commit"`
	Contract   string           `json:"oracle_contract_sha256"`
	ReportRoot string           `json:"report_root"`
	Oracles    string           `json:"oracle_root"`
	Cases      []mossParityCase `json:"cases"`
	Scope      string           `json:"scope"`
}

// checkMOSSParity recomputes each case from the retained report, line by line:
// every boundary on both passes within its frozen tolerance, each pass equal to
// the other, and one picture from both. The cases must cover the matrix the
// parity step names: two languages, a layout, two shapes, a negative prompt and
// a second seed.
func checkMOSSParity(t *testing.T, ctx context.Context, root string, roots dataroot.Roots) {
	t.Helper()
	var record mossParityRecord
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, mossParityRecordPath), &record))
	if _, err := gitauthority.Query(ctx, root, "cat-file", "-e", record.Commit+"^{commit}"); err != nil {
		t.Fatalf("commit %s: %v", record.Commit, err)
	}
	contract, err := os.ReadFile(filepath.Join(root, "fixtures", "moss", "oracle-contract.json"))
	testcheck.NoError(t, err)
	if digest := sha256.Sum256(contract); hex.EncodeToString(digest[:]) != record.Contract {
		t.Fatal("the oracle contract moved since these cases were compared")
	}
	data := filepath.Dir(roots.Datasets)
	shapes, seeds, languages, negatives, layouts := map[[2]int]bool{}, map[int64]bool{}, map[bool]bool{}, 0, 0
	for _, want := range record.Cases {
		oracle, err := artifact.IdentifyFile(artifact.KindTensorSet, filepath.Join(data, filepath.FromSlash(record.Oracles), want.Name, "full_reference.safetensors"))
		testcheck.NoError(t, err)
		if oracle != want.Oracle {
			t.Fatalf("%s: the oracle is %s, the record says %s", want.Name, oracle, want.Oracle)
		}
		raw, err := os.ReadFile(filepath.Join(data, filepath.FromSlash(record.ReportRoot), want.Report))
		testcheck.NoError(t, err)
		boundaries, worst, images := 0, 0.0, map[artifact.ID]int{}
		for line := range strings.SplitSeq(strings.TrimSpace(string(raw)), "\n") {
			var entry struct {
				Stage       string      `json:"stage"`
				Case        string      `json:"case"`
				Failed      int         `json:"failed"`
				MaxAbs      float64     `json:"max_abs"`
				RepeatEqual bool        `json:"repeat_equal"`
				Image       artifact.ID `json:"image"`
			}
			testcheck.NoError(t, json.Unmarshal([]byte(line), &entry))
			if entry.Case != want.Name {
				continue
			}
			switch entry.Stage {
			case "generation":
				if entry.Failed != 0 || !entry.RepeatEqual {
					t.Fatalf("%s: a boundary failed: %s", want.Name, line)
				}
				boundaries++
				worst = max(worst, entry.MaxAbs)
			case "generation-image":
				images[entry.Image]++
			}
		}
		if boundaries != want.Boundaries || worst != want.WorstAbs || images[want.Image] != mossParityPasses || len(images) != 1 {
			t.Fatalf("%s: the report says %d boundaries, worst %g, images %v; the record says %d, %g, %s", want.Name, boundaries, worst, images, want.Boundaries, want.WorstAbs, want.Image)
		}
		shapes[[2]int{want.Width, want.Height}] = true
		seeds[want.Seed] = true
		languages[strings.IndexFunc(want.Prompt, func(r rune) bool { return r > unicode.MaxASCII }) >= 0] = true
		if want.Negative != "" {
			negatives++
		}
		if strings.HasPrefix(want.Prompt, "{") {
			layouts++
		}
	}
	if len(shapes) < 2 || len(seeds) < 2 || len(languages) < 2 || negatives == 0 || layouts == 0 {
		t.Fatalf("the cases leave part of the matrix uncovered: %v %v %v, %d negative, %d layout", shapes, seeds, languages, negatives, layouts)
	}
	if strings.TrimSpace(record.Scope) == "" {
		t.Fatal("the record does not say what it establishes")
	}
}

// mossResourceRun is one request measured natively and upstream.
type mossResourceRun struct {
	Label           string             `json:"label"`
	Prompt          string             `json:"prompt"`
	Height          int                `json:"height"`
	Width           int                `json:"width"`
	Seed            int64              `json:"seed"`
	WallSeconds     float64            `json:"wall_seconds"`
	Phases          map[string]float64 `json:"phases_seconds"`
	PeakDeviceBytes uint64             `json:"peak_device_bytes"`
	PeakHostBytes   uint64             `json:"peak_host_bytes"`
}

// mossResourceRecord compares native and upstream on one protocol and one device.
type mossResourceRecord struct {
	Commit   string `json:"commit"`
	Device   string `json:"device"`
	Protocol string `json:"protocol"`
	Upstream struct {
		Settings    string            `json:"settings"`
		Packages    map[string]string `json:"packages"`
		LoadSeconds float64           `json:"load_seconds"`
		Requests    []mossResourceRun `json:"requests"`
	} `json:"upstream"`
	Native struct {
		LoadSeconds float64           `json:"load_seconds"`
		Requests    []mossResourceRun `json:"requests"`
		Largest     mossResourceRun   `json:"largest"`
	} `json:"native"`
	Scope string `json:"scope"`
}

// checkMOSSResources holds the comparison to equal requests: the two sides of a
// row must be the same picture request, and each must have measured what it
// reports, so a missing reading fails rather than reads as zero.
func checkMOSSResources(t *testing.T, ctx context.Context, root string) {
	t.Helper()
	var record mossResourceRecord
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, mossResourceRecordPath), &record))
	if _, err := gitauthority.Query(ctx, root, "cat-file", "-e", record.Commit+"^{commit}"); err != nil {
		t.Fatalf("commit %s: %v", record.Commit, err)
	}
	if record.Device == "" || record.Protocol == "" || record.Upstream.Settings == "" || len(record.Upstream.Packages) == 0 || strings.TrimSpace(record.Scope) == "" {
		t.Fatalf("the record does not state its hardware, protocol and scope: %+v", record)
	}
	if len(record.Native.Requests) == 0 || len(record.Native.Requests) != len(record.Upstream.Requests) {
		t.Fatalf("%d native rows against %d upstream", len(record.Native.Requests), len(record.Upstream.Requests))
	}
	measured := func(run mossResourceRun) {
		t.Helper()
		if run.WallSeconds <= 0 || run.PeakDeviceBytes == 0 || run.PeakHostBytes == 0 || len(run.Phases) == 0 {
			t.Fatalf("%s measured nothing: %+v", run.Label, run)
		}
	}
	for index, native := range record.Native.Requests {
		upstream := record.Upstream.Requests[index]
		if native.Label != upstream.Label || native.Prompt != upstream.Prompt || native.Height != upstream.Height || native.Width != upstream.Width || native.Seed != upstream.Seed {
			t.Fatalf("row %d compares different requests: %+v against %+v", index, native, upstream)
		}
		measured(native)
		measured(upstream)
	}
	measured(record.Native.Largest)
	if largest := record.Native.Largest; largest.Height*largest.Width <= record.Native.Requests[0].Height*record.Native.Requests[0].Width {
		t.Fatalf("the largest picture measured is %dx%d", largest.Width, largest.Height)
	}
}

// checkMOSSComplete holds the campaign's end: every retained reading stands, the
// model is active for pictures and for training in the store, the review has
// closed every finding it filed against MOSS, and the page a person reads states
// the figures the records hold.
func checkMOSSComplete(t *testing.T, root string, roots dataroot.Roots, store *overgodb.Store) {
	t.Helper()
	ctx := t.Context()
	checkMOSSTraining(t, root, roots, store)
	checkMOSSParity(t, ctx, root, roots)
	checkMOSSResources(t, ctx, root)
	var training mossTrainingReading
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, mossTrainingRecord), &training))
	for _, task := range []recipe.Task{recipe.TaskImageGen, recipe.TaskTraining} {
		if _, active, err := modelrecipe.ActiveRecord(ctx, store, training.Model, task); err != nil || !active {
			t.Fatalf("%s is not active: %v", task, err)
		}
	}
	var register struct {
		Findings []struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		} `json:"findings"`
	}
	testcheck.NoError(t, jsonfile.Decode(filepath.Join(root, "docs", "sqa_findings.json"), &register))
	for _, finding := range register.Findings {
		if strings.HasPrefix(finding.ID, "SQA-MOSS-") && finding.Status != "resolved" {
			t.Fatalf("%s is %s", finding.ID, finding.Status)
		}
	}
	var resources mossResourceRecord
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, mossResourceRecordPath), &resources))
	page, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(mossPage)))
	testcheck.NoError(t, err)
	figures := []string{fmt.Sprintf("%.4f", training.BaseMean), fmt.Sprintf("%.4f", training.TrainedMean), fmt.Sprintf("%.1f s", resources.Native.Largest.WallSeconds)}
	for index, native := range resources.Native.Requests {
		figures = append(figures, fmt.Sprintf("%.1f s", native.WallSeconds), fmt.Sprintf("%.1f s", resources.Upstream.Requests[index].WallSeconds))
	}
	for _, figure := range figures {
		if !strings.Contains(string(page), figure) {
			t.Fatalf("%s does not state %s", mossPage, figure)
		}
	}
}
