package dataset

import (
	"context"
	"encoding/binary"
	"errors"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/recipecontract"
)

const audioDecodeProfileSchema = "overgo/audio-decode-profile/v1"

// The complete publication envelope includes the immutable source descriptor.
// Earlier unversioned receipts omitted it; their request hashes must not be
// reused for this envelope even when all published facts are already present.
const audioInspectionBatchPrefix = "audio/inspection/v2/"

// AudioPayloadOrigin locates encoded audio within a registered container.
// ValueIndex is the non-null value ordinal of Column, not a source row number.
// An empty Column denotes the complete container file; dataset paths embedded
// in records are not followed. Container must already exist in the store.
type AudioPayloadOrigin struct {
	Container  artifact.ID `json:"container"`
	Column     string      `json:"column,omitzero"`
	ValueIndex uint64      `json:"value_index"`
}

// AudioInspectionPolicy supplies explicit resource and admission decisions.
// MaximumEncodedBytes bounds the individual payload, not dataset-reader memory.
type AudioInspectionPolicy struct {
	MaximumEncodedBytes uint64                              `json:"maximum_encoded_bytes"`
	MaximumSamples      uint64                              `json:"maximum_samples"`
	ClipThreshold       float64                             `json:"clip_threshold"`
	Admission           recipecontract.AudioAdmissionPolicy `json:"admission"`
}

// Validate rejects unspecified resource bounds and invalid admission policies.
func (policy AudioInspectionPolicy) Validate() error {
	if policy.MaximumEncodedBytes == 0 || policy.MaximumEncodedBytes >= uint64(math.MaxInt) || policy.MaximumSamples == 0 ||
		policy.MaximumSamples > uint64(math.MaxInt)/uint64(binary.Size(float32(0))) ||
		!checked.PositiveFinite64(policy.ClipThreshold) {
		return errors.New("dataset: invalid audio inspection bounds")
	}
	return policy.Admission.Validate()
}

// AudioInspection reports durable measurement and admission identities.
// Samples is populated only for accepted input and is never stored as a copy of
// the source corpus. DecodeError explains quarantined decoder failures.
type AudioInspection struct {
	SignalID      artifact.ID                              `json:"signal_id"`
	PolicyID      artifact.ID                              `json:"policy_id"`
	DecisionID    artifact.ID                              `json:"decision_id"`
	Signal        recipecontract.DecodedAudioSignalProfile `json:"signal"`
	Decision      recipecontract.AudioAdmissionDecision    `json:"decision"`
	DecodedSHA256 string                                   `json:"decoded_sha256,omitzero"`
	DecodeError   string                                   `json:"decode_error,omitzero"`
	Samples       []float32                                `json:"-"`
}

type audioDecodeProfile struct {
	Version       uint16                           `json:"version"`
	Source        artifact.ID                      `json:"source"`
	Origin        AudioPayloadOrigin               `json:"origin"`
	Policy        AudioInspectionPolicy            `json:"policy"`
	Format        recipecontract.AudioFormat       `json:"format,omitzero"`
	Status        recipecontract.AudioDecodeStatus `json:"status"`
	DecodedSHA256 string                           `json:"decoded_sha256,omitzero"`
	DecodeError   string                           `json:"decode_error,omitzero"`
}

// InspectAudio decodes actual bytes, measures the signal, and atomically
// publishes source-bound profiles and admission evidence. Invalid audio yields
// a quarantined result; invalid requests, cancellation, and store failures yield
// an error. The caller must prove that data belongs to the named container and
// selector; this function binds the inspected bytes independently by SHA-256.
func InspectAudio(ctx context.Context, repository artifact.Repository, data []byte, origin AudioPayloadOrigin, policy AudioInspectionPolicy) (AudioInspection, error) {
	if ctx == nil || repository == nil {
		return AudioInspection{}, errors.New("dataset: incomplete audio inspection")
	}
	if err := ctx.Err(); err != nil {
		return AudioInspection{}, err
	}
	if err := policy.Validate(); err != nil {
		return AudioInspection{}, err
	}
	if origin.Container.Kind() != artifact.KindFile && origin.Container.Kind() != artifact.KindDatasetShard ||
		origin.Column == "" && origin.ValueIndex != 0 {
		return AudioInspection{}, errors.New("dataset: invalid audio payload origin")
	}
	if _, found, err := repository.Artifact(ctx, origin.Container); err != nil || !found {
		if err != nil {
			return AudioInspection{}, err
		}
		return AudioInspection{}, errors.New("dataset: audio container is not registered")
	}
	source, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		return AudioInspection{}, err
	}
	if origin.Column == "" && origin.Container != source {
		return AudioInspection{}, errors.New("dataset: whole-file audio identity differs from container")
	}
	var audio media.DecodedAudio
	status := recipecontract.AudioDecodeResourceLimit
	decodeErr := errors.New("dataset: encoded audio exceeds payload budget")
	if uint64(len(data)) <= policy.MaximumEncodedBytes {
		audio, status, decodeErr = media.DecodeAudio(ctx, data, policy.MaximumSamples)
	}
	if err := ctx.Err(); err != nil {
		return AudioInspection{}, err
	}
	profile := audioDecodeProfile{Version: artifact.InitialDocumentVersion, Source: source,
		Origin: origin, Policy: policy, Format: audio.Format, Status: status}
	if decodeErr != nil {
		profile.DecodeError = decodeErr.Error()
	}
	if status == recipecontract.AudioDecodeComplete {
		profile.DecodedSHA256 = media.SamplesSHA256(audio.Samples)
	}
	formatContent, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, audioDecodeProfileSchema), profile)
	if err != nil {
		return AudioInspection{}, err
	}
	signal := recipecontract.DecodedAudioSignalProfile{Source: recipecontract.AudioReference{
		Audio: source, Profile: formatContent.Descriptor.ID,
	}, SchemaValid: true, DecodeStatus: status}
	if status == recipecontract.AudioDecodeComplete {
		signal, err = audiodsp.MeasureSignal(ctx, signal.Source, audio, policy.ClipThreshold)
		if err != nil {
			return AudioInspection{}, err
		}
	}
	measurement, err := audioSignalProfileCodec.NewInitial(AudioSignalProfileDocument{Profile: signal})
	if err != nil {
		return AudioInspection{}, err
	}
	admission, err := audioAdmissionPolicyCodec.NewInitial(AudioAdmissionPolicyDocument{Policy: policy.Admission})
	if err != nil {
		return AudioInspection{}, err
	}
	outcome, err := policy.Admission.Evaluate(signal)
	if err != nil {
		return AudioInspection{}, err
	}
	decision, err := audioAdmissionDecisionCodec.NewInitial(AudioAdmissionDecisionDocument{
		Signal: measurement.ID, Policy: admission.ID, Decision: outcome,
	})
	if err != nil {
		return AudioInspection{}, err
	}
	measurementContent, err := audioSignalProfileCodec.Content(measurement)
	if err != nil {
		return AudioInspection{}, err
	}
	admissionContent, err := audioAdmissionPolicyCodec.Content(admission)
	if err != nil {
		return AudioInspection{}, err
	}
	decisionContent, err := audioAdmissionDecisionCodec.Content(decision)
	if err != nil {
		return AudioInspection{}, err
	}
	lineage := artifact.DependencyLineage(formatContent.Descriptor.ID, source)
	if source != origin.Container {
		lineage = append(lineage, artifact.DependencyLineage(formatContent.Descriptor.ID, origin.Container)...)
	}
	lineage = append(lineage, artifact.DependencyLineage(measurement.ID, source, formatContent.Descriptor.ID)...)
	lineage = append(lineage, artifact.DependencyLineage(decision.ID, measurement.ID, admission.ID)...)
	batch, err := artifact.NewDocumentBatch(audioInspectionBatchPrefix+decision.ID.DigestHex(),
		[]artifact.Content{formatContent, measurementContent, admissionContent, decisionContent}, lineage, nil)
	if err != nil {
		return AudioInspection{}, err
	}
	descriptor, found, err := repository.Artifact(ctx, source)
	if err != nil {
		return AudioInspection{}, err
	}
	if found && descriptor.Size != uint64(len(data)) {
		return AudioInspection{}, errors.New("dataset: audio source descriptor size differs")
	}
	if !found {
		descriptor = artifact.Descriptor{ID: source, Size: uint64(len(data))}
	}
	batch.Artifacts = append(batch.Artifacts, descriptor)
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return AudioInspection{}, err
	}
	result := AudioInspection{SignalID: measurement.ID, PolicyID: admission.ID, DecisionID: decision.ID,
		Signal: signal, Decision: outcome, DecodedSHA256: profile.DecodedSHA256, DecodeError: profile.DecodeError}
	if outcome.Outcome == recipecontract.AudioAdmissionAccepted {
		result.Samples = audio.Samples
	}
	return result, nil
}
