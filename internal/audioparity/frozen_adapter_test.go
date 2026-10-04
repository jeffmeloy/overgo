package audioparity

import (
	"os"
	"slices"
	"testing"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/overgodb"
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
	storeRoot := resolveReferenceRoots(t).store
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
	corpus := librispeechPath(t, "clean", record.Split, record.Shard)
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
	t.Parallel()
	fixture := loadCTCTrainingFixture(t)
	encoder, features, frames, targets := fixture.encoder, fixture.features, fixture.frames, fixture.targets
	sourceTargets := fixture.record.SourceTargetIDs
	var ew speechrecognition.Workspace
	policy := trainingprogram.BuiltinOptimizerPolicy()
	adapter, err := encoder.NewOutputAdapter(t.Context(), &ew, frames, len(targets), fixture.blank, policy)
	if err != nil {
		t.Fatal(err)
	}
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
	t.Logf("CPU update: train.100 rows=1, frames=%d, targets=%d, parameters=%d, adapter_numeric_bytes=%d, CTC_sum_before=%.9g after=%.9g; frozen replay exact; held-out WER, resume, reload, serving and GPU not exercised", outputFrames, len(targets), len(trained), adapter.StorageBytes(), beforeLoss, afterLoss)
}
