package audioparity

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/modelartifact"
	"overgo/internal/pathidentity"
	"overgo/internal/processcontrol"
)

const (
	// ElectionMediaType identifies a selected audio-oracle evidence document.
	ElectionMediaType = "application/vnd.overgo.audio-oracle-election+json"
	// ElectionSchema identifies the audio-oracle election contract version.
	ElectionSchema = "overgo/audio-oracle-election/v1"
	// StandardTranscriptNormalization identifies case, punctuation, and whitespace normalization.
	StandardTranscriptNormalization = "lowercase-alphanumeric-whitespace-v1"
	// ApostropheSEquivalence identifies the diagnostic apostrophe-s equivalence policy.
	ApostropheSEquivalence = "english-apostrophe-s-equivalence-v1"
	// FirstOfflineASRAlias is the stable store alias for the elected offline ASR oracle.
	FirstOfflineASRAlias = "audio/oracle/offline-asr/first"
)

// SourceIdentity binds an external repository to one immutable revision.
type SourceIdentity struct {
	Repository string `json:"repository"`
	Commit     string `json:"commit"`
}

// FileBinding identifies one complete byte member of an external model snapshot.
type FileBinding struct {
	Path      string      `json:"path"`
	Role      string      `json:"role"`
	Artifact  artifact.ID `json:"artifact"`
	Bytes     uint64      `json:"bytes"`
	MediaType string      `json:"media_type"`
}

// LicenseBinding binds the declared license to the exact model-card bytes carrying it.
type LicenseBinding struct {
	SPDX     string      `json:"spdx"`
	URL      string      `json:"url"`
	Document artifact.ID `json:"document"`
}

// DatasetBinding identifies the exact corpus shard used by the election.
type DatasetBinding struct {
	Name      string         `json:"name"`
	Source    SourceIdentity `json:"source"`
	Split     string         `json:"split"`
	Path      string         `json:"path"`
	Artifact  artifact.ID    `json:"artifact"`
	Bytes     uint64         `json:"bytes"`
	MediaType string         `json:"media_type"`
}

// ReferenceRuntime identifies the external CPU execution environment and entrypoints.
type ReferenceRuntime struct {
	Implementation  string   `json:"implementation"`
	Python          string   `json:"python"`
	Torch           string   `json:"torch"`
	Transformers    string   `json:"transformers"`
	Device          string   `json:"device"`
	DType           string   `json:"dtype"`
	Host            string   `json:"host"`
	OS              string   `json:"os"`
	Arch            string   `json:"arch"`
	ObserverCommit  string   `json:"observer_commit"`
	ModelLoadWallNS uint64   `json:"model_load_wall_ns"`
	Entrypoints     []string `json:"entrypoints"`
}

// TranscriptObservation binds one decoded fixture to its reference transcript and timing.
type TranscriptObservation struct {
	Fixture    string      `json:"fixture"`
	Audio      artifact.ID `json:"audio"`
	AudioBytes uint64      `json:"audio_bytes"`
	Samples    uint64      `json:"samples"`
	SampleRate uint32      `json:"sample_rate"`
	Expected   string      `json:"expected"`
	Observed   string      `json:"observed"`
	WallNS     uint64      `json:"wall_ns"`
}

// Election is the immutable, content-addressed decision to use one external audio oracle.
type Election struct {
	Version        uint16                  `json:"version"`
	Capability     string                  `json:"capability"`
	Name           string                  `json:"name"`
	Family         string                  `json:"family"`
	Model          artifact.Manifest       `json:"model"`
	ModelSource    SourceIdentity          `json:"model_source"`
	Files          []FileBinding           `json:"files"`
	License        LicenseBinding          `json:"license"`
	Dataset        DatasetBinding          `json:"dataset"`
	Runtime        ReferenceRuntime        `json:"runtime"`
	Normalizations []string                `json:"normalizations"`
	Observations   []TranscriptObservation `json:"observations"`
	Reason         string                  `json:"reason"`
	ReopenWhen     []string                `json:"reopen_when"`
	ID             artifact.ID             `json:"-"`
}

// ElectionSummary derives the bounded parity measurements from an election.
type ElectionSummary struct {
	Fixtures          int
	StandardMatches   int
	EquivalentMatches int
	AudioSamples      uint64
	WallNS            uint64
}

var electionCodec = artifact.JSONDocumentCodec(
	"audio oracle election", artifact.KindEvidence, ElectionMediaType, ElectionSchema,
	canonicalizeElection,
	func(value Election) artifact.ID { return value.ID },
	func(value *Election, id artifact.ID) { value.ID = id },
	cloneElection,
)

var fileRoles = map[string]bool{
	"configuration": true, "generation-config": true, "license": true,
	"preprocessor": true, "reference-code": true, "signature": true,
	"source-metadata": true, "tokenizer": true, "weights": true,
}

// NormalizeElection accepts strict external JSON and returns its canonical election value.
func NormalizeElection(data []byte) (Election, error) {
	value, _, err := electionCodec.Normalize(data)
	return value, err
}

// Content returns the canonical election document.
func (e Election) Content() (artifact.Content, error) {
	return electionCodec.Content(e)
}

// Summary derives transcript agreement and total CPU work without stored counters.
func (e Election) Summary() ElectionSummary {
	summary := ElectionSummary{Fixtures: len(e.Observations)}
	for _, observation := range e.Observations {
		if normalizeTranscript(observation.Expected) == normalizeTranscript(observation.Observed) {
			summary.StandardMatches++
		}
		if normalizeApostropheS(observation.Expected) == normalizeApostropheS(observation.Observed) {
			summary.EquivalentMatches++
		}
		summary.AudioSamples += observation.Samples
		summary.WallNS += observation.WallNS
	}
	return summary
}

// Batch compiles the complete external artifact and evidence graph for atomic publication.
func (e Election) Batch(key string) (artifact.Batch, error) {
	content, err := e.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	descriptors := make([]artifact.Descriptor, 0, len(e.Files)+len(e.Observations)+1)
	parents := make([]artifact.ID, 0, len(e.Files)+len(e.Observations)+2)
	parents = append(parents, e.Model.ID)
	for _, file := range e.Files {
		descriptors = append(descriptors, artifact.Descriptor{
			ID: file.Artifact, Size: file.Bytes, MediaType: file.MediaType,
		})
		parents = append(parents, file.Artifact)
	}
	descriptors = append(descriptors, artifact.Descriptor{
		ID: e.Dataset.Artifact, Size: e.Dataset.Bytes, MediaType: e.Dataset.MediaType,
	})
	parents = append(parents, e.Dataset.Artifact)
	for _, observation := range e.Observations {
		descriptors = append(descriptors, artifact.Descriptor{
			ID: observation.Audio, Size: observation.AudioBytes, MediaType: "audio/flac",
		})
		parents = append(parents, observation.Audio)
	}
	batch, err := artifact.NewDocumentBatch(
		key,
		[]artifact.Content{content},
		artifact.UniqueDependencyLineage(e.ID, parents...),
		[]artifact.AliasBinding{{Name: FirstOfflineASRAlias, Target: e.ID}},
	)
	if err != nil {
		return artifact.Batch{}, err
	}
	batch.Artifacts = descriptors
	batch.Manifests = []artifact.Manifest{e.Model}
	if err := batch.Validate(); err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
}

// VerifyInstalled checks the exact model Git snapshot, model manifest, and corpus-shard bytes.
func (e Election) VerifyInstalled(ctx context.Context, modelRoot, datasetRoot string) error {
	if ctx == nil {
		return errors.New("audio parity: nil verification context")
	}
	if err := e.verifyRepository(ctx, modelRoot, e.ModelSource, true); err != nil {
		return fmt.Errorf("audio parity: model source: %w", err)
	}
	if err := e.verifyRepository(ctx, datasetRoot, e.Dataset.Source, false); err != nil {
		return fmt.Errorf("audio parity: dataset source: %w", err)
	}
	inventory, err := modelartifact.FromHFPath(modelRoot)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(inventory.Manifest, e.Model) {
		return errors.New("audio parity: installed model manifest differs from election")
	}
	for _, file := range e.Files {
		if err := verifyFile(modelRoot, file.Path, file.Artifact, file.Bytes); err != nil {
			return err
		}
	}
	if err := verifyFile(datasetRoot, e.Dataset.Path, e.Dataset.Artifact, e.Dataset.Bytes); err != nil {
		return err
	}
	return nil
}

func (e Election) verifyRepository(ctx context.Context, root string, source SourceIdentity, complete bool) error {
	absolute, err := pathidentity.Canonical(root)
	if err != nil {
		return err
	}
	commit, err := gitOutput(ctx, absolute, "rev-parse", "HEAD")
	if err != nil || strings.TrimSpace(string(commit)) != source.Commit {
		return errors.Join(errors.New("repository commit differs"), err)
	}
	if !complete {
		return nil
	}
	tracked, err := gitOutput(ctx, absolute, "ls-files", "-z")
	if err != nil {
		return err
	}
	paths := strings.Split(strings.TrimSuffix(string(tracked), "\x00"), "\x00")
	want := make([]string, len(e.Files))
	for index, file := range e.Files {
		want[index] = file.Path
	}
	slices.Sort(paths)
	if !slices.Equal(paths, want) {
		return fmt.Errorf("tracked files differ: got %v, want %v", paths, want)
	}
	return nil
}

func gitOutput(ctx context.Context, root string, args ...string) ([]byte, error) {
	arguments := append([]string{"-C", root}, args...)
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "git", Args: arguments, Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("git %s: exit=%d: %s", strings.Join(args, " "), receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func verifyFile(root, relative string, id artifact.ID, wantBytes uint64) error {
	absoluteRoot, err := pathidentity.Canonical(root)
	if err != nil {
		return err
	}
	absolute, err := pathidentity.Canonical(filepath.Join(absoluteRoot, filepath.FromSlash(relative)))
	if err != nil {
		return err
	}
	contained, err := pathidentity.Contains(absoluteRoot, absolute)
	if err != nil || !contained {
		return fmt.Errorf("audio parity: file %q escapes its root", relative)
	}
	file, err := os.Open(absolute)
	if err != nil {
		return err
	}
	got, size, hashErr := artifact.Identify(id.Kind(), file)
	closeErr := file.Close()
	if err := errors.Join(hashErr, closeErr); err != nil {
		return err
	}
	if got != id || size != wantBytes {
		return fmt.Errorf("audio parity: file %q differs from election", relative)
	}
	return nil
}

func canonicalizeElection(election *Election) error {
	if election == nil || election.Version != artifact.InitialDocumentVersion {
		return errors.New("audio parity: invalid election version")
	}
	if election.Capability != "offline-asr" || strings.TrimSpace(election.Name) == "" || strings.TrimSpace(election.Family) == "" {
		return errors.New("audio parity: incomplete elected capability")
	}
	if err := election.Model.Validate(); err != nil || election.Model.ID.Kind() != artifact.KindModel {
		return errors.Join(errors.New("audio parity: invalid model manifest"), err)
	}
	if err := validateSource(election.ModelSource); err != nil {
		return err
	}
	slices.SortFunc(election.Files, func(left, right FileBinding) int { return strings.Compare(left.Path, right.Path) })
	seenPaths := make(map[string]bool, len(election.Files))
	seenArtifacts := make(map[artifact.ID]bool, len(election.Files))
	roles := make(map[string]int)
	for _, file := range election.Files {
		if err := validateRelativePath(file.Path); err != nil || !fileRoles[file.Role] || !file.Artifact.Valid() || file.Bytes == 0 || strings.TrimSpace(file.MediaType) == "" {
			return errors.Join(fmt.Errorf("audio parity: invalid file binding %q", file.Path), err)
		}
		if seenPaths[file.Path] || seenArtifacts[file.Artifact] {
			return fmt.Errorf("audio parity: duplicate file binding %q", file.Path)
		}
		seenPaths[file.Path], seenArtifacts[file.Artifact] = true, true
		roles[file.Role]++
	}
	for _, role := range []string{"configuration", "license", "preprocessor", "reference-code", "tokenizer", "weights"} {
		if roles[role] == 0 {
			return fmt.Errorf("audio parity: model snapshot lacks %s bytes", role)
		}
	}
	for _, component := range election.Model.Components {
		if !seenArtifacts[component.Artifact] {
			return fmt.Errorf("audio parity: manifest component %q is absent from snapshot", component.Name)
		}
	}
	licenseBound := false
	for _, file := range election.Files {
		licenseBound = licenseBound || file.Role == "license" && file.Artifact == election.License.Document
	}
	if election.License.SPDX != "Apache-2.0" || strings.TrimSpace(election.License.URL) == "" || !licenseBound {
		return errors.New("audio parity: invalid license binding")
	}
	if err := validateDataset(election.Dataset); err != nil {
		return err
	}
	if err := validateRuntime(&election.Runtime); err != nil {
		return err
	}
	slices.Sort(election.Normalizations)
	if !slices.Equal(election.Normalizations, []string{ApostropheSEquivalence, StandardTranscriptNormalization}) {
		return errors.New("audio parity: normalization policies differ")
	}
	slices.SortFunc(election.Observations, func(left, right TranscriptObservation) int {
		return strings.Compare(left.Fixture, right.Fixture)
	})
	seenFixtures := make(map[string]bool, len(election.Observations))
	for _, observation := range election.Observations {
		if strings.TrimSpace(observation.Fixture) == "" || observation.Audio.Kind() != artifact.KindFile || observation.AudioBytes == 0 ||
			observation.Samples == 0 || observation.SampleRate == 0 || strings.TrimSpace(observation.Expected) == "" ||
			strings.TrimSpace(observation.Observed) == "" || observation.WallNS == 0 || seenFixtures[observation.Fixture] {
			return fmt.Errorf("audio parity: invalid transcript observation %q", observation.Fixture)
		}
		seenFixtures[observation.Fixture] = true
	}
	summary := election.Summary()
	if summary.Fixtures == 0 || summary.StandardMatches == 0 || summary.EquivalentMatches != summary.Fixtures {
		return errors.New("audio parity: election does not preserve reference parity")
	}
	if strings.TrimSpace(election.Reason) == "" || len(election.ReopenWhen) == 0 {
		return errors.New("audio parity: election lacks decision or reopen conditions")
	}
	slices.Sort(election.ReopenWhen)
	for index, condition := range election.ReopenWhen {
		if strings.TrimSpace(condition) != condition || condition == "" || index > 0 && condition == election.ReopenWhen[index-1] {
			return errors.New("audio parity: invalid reopen condition")
		}
	}
	return nil
}

func validateSource(source SourceIdentity) error {
	if strings.TrimSpace(source.Repository) != source.Repository || source.Repository == "" ||
		!gitauthority.ValidObjectID(source.Commit) || strings.ToLower(source.Commit) != source.Commit {
		return errors.New("audio parity: invalid source identity")
	}
	return nil
}

func validateDataset(dataset DatasetBinding) error {
	if strings.TrimSpace(dataset.Name) == "" || strings.TrimSpace(dataset.Split) == "" || dataset.Artifact.Kind() != artifact.KindDatasetShard ||
		dataset.Bytes == 0 || dataset.MediaType != "application/vnd.apache.parquet" {
		return errors.New("audio parity: invalid dataset binding")
	}
	if err := validateSource(dataset.Source); err != nil {
		return err
	}
	return validateRelativePath(dataset.Path)
}

func validateRuntime(runtime *ReferenceRuntime) error {
	if runtime == nil || runtime.Implementation == "" || runtime.Python == "" || runtime.Torch == "" || runtime.Transformers == "" ||
		runtime.Device != "cpu" || runtime.DType == "" || runtime.Host == "" || runtime.OS == "" || runtime.Arch == "" ||
		!gitauthority.ValidObjectID(runtime.ObserverCommit) || strings.ToLower(runtime.ObserverCommit) != runtime.ObserverCommit ||
		runtime.ModelLoadWallNS == 0 || len(runtime.Entrypoints) == 0 {
		return errors.New("audio parity: invalid reference runtime")
	}
	slices.Sort(runtime.Entrypoints)
	for index, entrypoint := range runtime.Entrypoints {
		if strings.TrimSpace(entrypoint) != entrypoint || entrypoint == "" || index > 0 && entrypoint == runtime.Entrypoints[index-1] {
			return errors.New("audio parity: invalid reference entrypoint")
		}
	}
	return nil
}

func validateRelativePath(value string) error {
	if value == "" || strings.Contains(value, "\\") || path.IsAbs(value) || path.Clean(value) != value || strings.HasPrefix(value, "../") {
		return fmt.Errorf("invalid relative path %q", value)
	}
	return nil
}

func normalizeApostropheS(value string) string {
	value = strings.ReplaceAll(strings.ToLower(value), "'s", " is")
	value = strings.ReplaceAll(value, "’s", " is")
	return normalizeTranscript(value)
}

func normalizeTranscript(value string) string {
	var normalized strings.Builder
	for _, char := range strings.ToLower(value) {
		if unicode.IsLetter(char) || unicode.IsNumber(char) {
			normalized.WriteRune(char)
		} else {
			normalized.WriteByte(' ')
		}
	}
	return strings.Join(strings.Fields(normalized.String()), " ")
}

func cloneElection(value Election) Election {
	value.Model = value.Model.Clone()
	value.Files = slices.Clone(value.Files)
	value.Runtime.Entrypoints = slices.Clone(value.Runtime.Entrypoints)
	value.Normalizations = slices.Clone(value.Normalizations)
	value.Observations = slices.Clone(value.Observations)
	value.ReopenWhen = slices.Clone(value.ReopenWhen)
	return value
}
