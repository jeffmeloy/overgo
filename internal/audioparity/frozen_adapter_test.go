package audioparity

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/binaryschema"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

type ctcTrainingRecord struct {
	Normalization   string `json:"normalization"`
	TargetText      string `json:"target_text"`
	SourceTargetIDs []int  `json:"source_target_ids"`
	GeneratorSHA256 string `json:"generator_sha256"`
	Schema          string `json:"schema"`
	Split           string `json:"split"`
	Shard           string `json:"shard"`
	ShardSHA256     string `json:"shard_sha256"`
	ShardBytes      uint64 `json:"shard_bytes"`
	Row             uint64 `json:"row"`
	RowID           string `json:"row_id"`
	Text            string `json:"text"`
	AudioSHA256     string `json:"audio_sha256"`
	TargetIDs       []int  `json:"target_ids"`
	PyArrow         string `json:"pyarrow"`
	Tokenizers      string `json:"tokenizers"`
	Samples         uint64 `json:"samples"`
	SampleRate      uint64 `json:"sample_rate"`
}

type ctcTrainingFixture struct {
	root, storeRoot, modelRoot string
	election                   Election
	record                     ctcTrainingRecord
	frontend                   audiodsp.FrontendConfig
	grouping                   audiodsp.GroupedFeatureConfig
	declaration                speechrecognition.Declaration
	encoder                    *speechrecognition.Encoder
	features                   []float32
	frames, blank              int
	targets                    []int
	payload                    []byte
}

func loadCTCTrainingFixture(t *testing.T) ctcTrainingFixture {
	t.Helper()
	root := testutil.RepoRoot(t)
	storeRoot := cmp.Or(os.Getenv("OVERGO_AUDIO_REFERENCE_STORE"), filepath.Join(root, "overgodb-store"))
	store, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	encoded, err := os.ReadFile("testdata/granite_speech_5_election.json")
	if err != nil {
		t.Fatal(err)
	}
	election, err := NormalizeElection(encoded)
	if err != nil {
		t.Fatal(err)
	}
	modelRoot, weights := loadASRModel(t, store, election)
	encoded, err = os.ReadFile("testdata/ctc_training_record.json")
	if err != nil {
		t.Fatal(err)
	}
	var record ctcTrainingRecord
	if err := strictjson.DecodeBytes(encoded, &record); err != nil {
		t.Fatal(err)
	}
	if record.Schema != "overgo/ctc-training-record/v1" || record.Split != "train.100" || record.Shard != "0000.parquet" ||
		record.Row != 0 || record.RowID != "374-180298-0000" || record.PyArrow != "24.0.0" || record.Tokenizers != "0.22.2" || len(record.TargetIDs) == 0 ||
		record.Normalization != "lowercase" || record.GeneratorSHA256 != "8c1a76d8fff3cee8ca6442f93a936c7c47224c84f3fecc5d988e2ab2fa46dc39" {
		t.Fatal("training record provenance differs")
	}
	corpus := filepath.Join(filepath.Dir(storeRoot), "datasets", "librispeech_asr-clean-xet", "clean", record.Split, record.Shard)
	shardID, err := artifact.ParseID("file:sha256:" + record.ShardSHA256)
	if err != nil {
		t.Fatal(err)
	}
	verifyASRFile(t, corpus, shardID, record.ShardBytes)
	// Explicit per-component integration ceilings; neither is a process peak.
	const sourceMemoryBytes = 512 << 20
	const modelMemoryBytes = 4 << 30
	source, err := dataset.OpenParquetRows(t.Context(), corpus, []string{"audio.bytes", "text", "id"}, sourceMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	row, err := source.Read(t.Context(), record.Row)
	if err != nil {
		t.Fatal(err)
	}
	if row["id"] == nil || *row["id"] != record.RowID || row["text"] == nil || *row["text"] != record.Text || row["audio.bytes"] == nil {
		t.Fatal("training source row differs")
	}
	payload := []byte(*row["audio.bytes"])
	audioID, err := artifact.IdentifyBytes(artifact.KindFile, payload)
	if err != nil || audioID.DigestHex() != record.AudioSHA256 {
		t.Fatalf("audio identity differs: %v", err)
	}
	tokenizer, err := hfbpe.Load(modelRoot)
	if err != nil {
		t.Fatal(err)
	}
	sourceTargets, err := tokenizer.Encode(*row["text"])
	if err != nil || !slices.Equal(sourceTargets, record.SourceTargetIDs) {
		t.Fatalf("raw source tokenization differs: %v", err)
	}
	transform, err := trainingdata.NewTextTransform(true)
	if err != nil {
		t.Fatal(err)
	}
	targetText, err := transform.Apply(*row["text"])
	if err != nil {
		t.Fatal(err)
	}
	if targetText != record.TargetText {
		t.Fatal("declared target normalization differs")
	}
	targets, err := tokenizer.Encode(targetText)
	if err != nil || !slices.Equal(targets, record.TargetIDs) {
		t.Fatalf("target tokenization differs from independent oracle: %v", err)
	}
	audio, _, err := media.DecodeAudio(t.Context(), payload, record.Samples)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = os.ReadFile("../audiodsp/testdata/power_logmel_frontend.json")
	if err != nil {
		t.Fatal(err)
	}
	var frontendConfig audiodsp.FrontendConfig
	if err := strictjson.DecodeBytes(encoded, &frontendConfig); err != nil {
		t.Fatal(err)
	}
	frontend, err := audiodsp.NewFrontend(frontendConfig, modelMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	recipe, _, err := graniteRecipe()
	if err != nil {
		t.Fatal(err)
	}
	grouping := audiodsp.GroupedFeatureConfig{StackFrames: int(recipe.Preprocessor.StackFactor), DeltaRadius: int(recipe.Preprocessor.DeltaWinLength-1) / 2, FinalFrameSamples: 1}
	var fw audiodsp.Workspace
	if audio.Format.Channels != 1 || audio.Format.SampleRate != record.SampleRate || uint64(len(audio.Samples)) != record.Samples {
		t.Fatal("training fixture audio geometry differs from source FLAC metadata")
	}
	features, frames, _, err := frontend.ProcessGrouped(t.Context(), audio.Samples, int(audio.Format.SampleRate), &fw, grouping)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = os.ReadFile("recipes/granite5asr_execution.json")
	if err != nil {
		t.Fatal(err)
	}
	var declaration speechrecognition.Declaration
	if err := strictjson.DecodeBytes(encoded, &declaration); err != nil {
		t.Fatal(err)
	}
	encoder, err := speechrecognition.LoadEncoder(t.Context(), weights, declaration, modelMemoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	return ctcTrainingFixture{root: root, storeRoot: storeRoot, modelRoot: modelRoot, election: election, record: record,
		frontend: frontendConfig, grouping: grouping, declaration: declaration, encoder: encoder,
		features: features, frames: frames, blank: int(recipe.Config.PadTokenID), targets: targets, payload: payload}
}

// TestASRFrozenAdapterAcceptance performs one CPU update on an independently
// pinned training record. It proves the native training boundary, not WER,
// general tokenizer coverage, checkpoint reload, or a deployed adapted recipe.
func TestASRFrozenAdapterAcceptance(t *testing.T) {
	fixture := loadCTCTrainingFixture(t)
	root, election, record := fixture.root, fixture.election, fixture.record
	encoder, features, frames, targets := fixture.encoder, fixture.features, fixture.frames, fixture.targets
	sourceTargets, declaration := record.SourceTargetIDs, fixture.declaration
	var ew speechrecognition.Workspace
	policy := trainingprogram.BuiltinOptimizerPolicy()
	adapter, err := encoder.NewOutputAdapter(t.Context(), &ew, frames, len(targets), fixture.blank, policy)
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	hidden, outputFrames, err := encoder.Encode(t.Context(), features, frames, &ew, nil)
	if err != nil {
		t.Fatal(err)
	}
	base, err := encoder.Project(t.Context(), hidden, outputFrames, &ew)
	if err != nil {
		t.Fatal(err)
	}
	beforeHidden, beforeBase, beforeAdapter := slices.Clone(hidden), slices.Clone(base), adapter.WeightSnapshot()
	// The source's uppercase tokenization is a real impossible-alignment
	// negative control. Do not silently zero its loss or modify decoder semantics.
	ctcSize, err := trainingprogram.CTCLossWorkspaceSize(outputFrames, encoder.VocabularySize(), len(sourceTargets))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := trainingprogram.CTCLossF32(t.Context(), nil, base, sourceTargets, outputFrames, encoder.VocabularySize(), fixture.blank, make([]float64, ctcSize)); err == nil {
		t.Fatal("impossible raw-source alignment was admitted")
	}
	execution, err := adapter.Bind(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	forward, err := execution.Select(trainingprogram.PhaseForward)
	if err != nil {
		t.Fatal(err)
	}
	example := adaptertrain.LinearCTCExample{Hidden: hidden, Targets: targets, Frames: outputFrames}
	if err := forward.Run(&example); err != nil || !slices.Equal(base, example.Logits) {
		t.Fatalf("identity adapter differs: %v", err)
	}
	beforeLoss := example.Loss
	if err := execution.Run(&example); err != nil || example.Update.Step != 1 {
		t.Fatalf("real CPU update failed: %v", err)
	}
	trained := adapter.WeightSnapshot()
	if slices.Equal(beforeAdapter, trained) {
		t.Fatal("real training did not modify adapter")
	}
	if err := forward.Run(&example); err != nil || slices.Equal(base, example.Logits) {
		t.Fatalf("adapted output unchanged: %v", err)
	}
	afterLoss := example.Loss
	hidden, outputFrames, err = encoder.Encode(t.Context(), features, frames, &ew, nil)
	if err != nil {
		t.Fatal(err)
	}
	base, err = encoder.Project(t.Context(), hidden, outputFrames, &ew)
	if err != nil || !slices.Equal(beforeHidden, hidden) || !slices.Equal(beforeBase, base) {
		t.Fatalf("frozen base replay changed: %v", err)
	}
	elapsed := time.Since(started)
	placement, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/frozen-output-placement-observation/v1"), struct {
		Base     artifact.ID                   `json:"base"`
		Encoder  speechrecognition.Declaration `json:"encoder"`
		Position string                        `json:"position"`
		Width    int                           `json:"width"`
	}{election.Model.ID, declaration, "final-hidden-before-frozen-output", len(hidden) / outputFrames})
	if err != nil {
		t.Fatal(err)
	}
	retainFrozenAdapterObservation(t, root, election.Model.ID, execution.ProgramID(), policy, placement, record, beforeAdapter, trained, beforeLoss, afterLoss, adapter.StorageBytes(), elapsed)
	t.Logf("CPU update: train.100 rows=1, frames=%d, targets=%d, parameters=%d, adapter_numeric_bytes=%d, CTC_sum_before=%.9g after=%.9g; wall=%s (not benchmark); frozen replay exact; held-out WER, resume, reload, serving and GPU not exercised", outputFrames, len(targets), len(trained), adapter.StorageBytes(), beforeLoss, afterLoss, time.Since(started))
}

// Store immutable observation identities, not a new checkpoint format. The
// snapshots below identify raw row-major F32 values. The separate lifecycle
// acceptance tests loadable publication; this test covers observation lineage.
func retainFrozenAdapterObservation(t *testing.T, root string, base, program artifact.ID, policy trainingprogram.OptimizerPolicy, placement artifact.Content, record ctcTrainingRecord, before, after []float32, beforeLoss, afterLoss float64, numericBytes uint64, elapsed time.Duration) {
	t.Helper()
	snapshot := func(values []float32) artifact.Descriptor {
		encoded := make([]byte, len(values)*binaryschema.Uint32Bytes)
		for index, value := range values {
			binary.LittleEndian.PutUint32(encoded[index*binaryschema.Uint32Bytes:], math.Float32bits(value))
		}
		id, err := artifact.IdentifyBytes(artifact.KindAdapter, encoded)
		if err != nil {
			t.Fatal(err)
		}
		return artifact.Descriptor{ID: id, Size: uint64(len(encoded))}
	}
	beforeID, afterID := snapshot(before), snapshot(after)
	source, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/ctc-training-record/v1"), record)
	if err != nil {
		t.Fatal(err)
	}
	observation, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/frozen-adapter-observation/v1"), struct {
		Base         artifact.ID `json:"base"`
		Program      artifact.ID `json:"program"`
		Before       artifact.ID `json:"adapter_before"`
		After        artifact.ID `json:"adapter_after"`
		Placement    artifact.ID `json:"placement"`
		Source       artifact.ID `json:"source"`
		Optimizer    artifact.ID `json:"optimizer"`
		LossBefore   float64     `json:"ctc_sum_before"`
		LossAfter    float64     `json:"ctc_sum_after"`
		NumericBytes uint64      `json:"adapter_numeric_bytes"`
		Scope        string      `json:"scope"`
	}{base, program, beforeID.ID, afterID.ID, placement.Descriptor.ID, source.Descriptor.ID, policy.ID, beforeLoss, afterLoss, numericBytes,
		"one CPU update; exact frozen replay; snapshot identities only, no checkpoint publication, resume, held-out WER or served-recipe claim"})
	if err != nil {
		t.Fatal(err)
	}
	code, err := repoanalysis.DiscoverGo(root, "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.CurrentEnvironment("cpu", fmt.Sprintf("go-host-reference/source=%s", code.Identity()))
	if err != nil {
		t.Fatal(err)
	}
	commit := strings.TrimSpace(baselineCommand(t, root, "git", "rev-parse", "HEAD"))
	run, err := runrecord.NewBoundRun(program, runrecord.OutcomeSucceeded,
		[]artifact.ID{base, beforeID.ID, placement.Descriptor.ID, source.Descriptor.ID, policy.ID},
		[]artifact.ID{afterID.ID, observation.Descriptor.ID}, "", commit, environment.ID, uint64(elapsed),
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseTest, DurationNS: uint64(elapsed)}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := run.Batch("frozen-adapter/" + run.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	for _, get := range []func() (artifact.Content, error){environment.Content, policy.Content} {
		content, err := get()
		if err != nil {
			t.Fatal(err)
		}
		batch.Contents = append(batch.Contents, content)
	}
	batch.Contents = append(batch.Contents, placement, source, observation)
	batch.Artifacts = append(batch.Artifacts, beforeID, afterID, artifact.Descriptor{ID: base}, artifact.Descriptor{ID: program})
	batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(observation.Descriptor.ID, base, program, beforeID.ID, afterID.ID, placement.Descriptor.ID, source.Descriptor.ID, policy.ID)...)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.RequireExactRun(t.Context(), store, run.ID); err != nil {
		t.Fatal(err)
	}
	if content, err := artifact.RequireTypedContent(t.Context(), store, observation.Descriptor.ID); err != nil || !slices.Equal(content.Data, observation.Data) {
		t.Fatalf("observation round trip differs: %v", err)
	}
	t.Logf("isolated-store run=%s recipe=%s adapter=%s placement=%s; gate retains this observation transcript, not a reloadable adapter", run.ID, program, afterID.ID, placement.Descriptor.ID)
}
