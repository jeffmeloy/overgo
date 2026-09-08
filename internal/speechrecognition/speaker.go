package speechrecognition

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/checked"
	"overgo/internal/dataset"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
)

var speechTurnsContract = artifact.JSONContract(artifact.KindOutput, "overgo/speech-turns/v1")

type speakerModel struct {
	repository artifact.Repository
	definition recipe.Definition
	profile    SpeakerProfile
	frontend   *audiodsp.Frontend
	activity   *SpeakerActivity
}

func loadSpeaker(ctx context.Context, repository artifact.Repository, definition recipe.Definition, memory uint64) (*speakerModel, error) {
	id, _ := definition.PrimaryDependency(recipe.DependencyProcessorProfile)
	profile, err := RequireSpeakerProfile(ctx, repository, id)
	if err != nil {
		return nil, err
	}
	model, _ := definition.PrimaryDependency(recipe.DependencyModel)
	inventoryID, _ := definition.PrimaryDependency(recipe.DependencyTensorInventory)
	if profile.Model != model || profile.Inventory != inventoryID {
		return nil, errors.New("speaker activity: profile dependency mismatch")
	}
	license, found, err := artifact.ReadContent(ctx, repository, profile.License)
	if err != nil || !found || len(license.Data) == 0 {
		return nil, errors.Join(errors.New("speaker activity: license evidence absent"), err)
	}
	if err := license.Validate(); err != nil {
		return nil, err
	}
	inventory, err := modelartifact.ReinspectFiles(ctx, repository, model)
	if err != nil || inventory.TensorInventory.ID != inventoryID {
		return nil, errors.Join(errors.New("speaker activity: physical tensor inventory differs"), err)
	}
	weights := 0
	var weightsID artifact.ID
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentWeights {
			weights++
			weightsID = component.Artifact
		} else if component.Role == artifact.ComponentWeightsShard {
			return nil, errors.New("speaker activity: sharded checkpoint not declared")
		}
	}
	if weights != 1 {
		return nil, errors.New("speaker activity: one checkpoint component required")
	}
	root, err := artifact.AvailablePath(ctx, repository, model, artifact.LocationDirectory)
	if err != nil {
		return nil, err
	}
	source, err := safetensors.OpenSource(root)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	digests, err := source.ShardDigests()
	if err != nil {
		return nil, err
	}
	if len(digests) != 1 || source.Indexed() {
		return nil, errors.New("speaker activity: opened checkpoint layout differs")
	}
	actual, err := artifact.NewID(artifact.KindTensorSet, digests[0].Digest)
	if err != nil || actual != weightsID {
		return nil, errors.New("speaker activity: opened checkpoint identity differs")
	}
	activity, err := LoadSpeakerActivity(ctx, source, profile.Encoder, profile.Activity, memory)
	if err != nil {
		return nil, err
	}
	frontend, err := audiodsp.NewFrontend(profile.Frontend, memory)
	if err != nil {
		return nil, err
	}
	return &speakerModel{repository, definition, profile, frontend, activity}, nil
}

// Diarize executes source-bound speaker attribution through the shared speech
// component lease. It records inspection, input, recipe and code/environment
// identities and preserves overlaps. It neither recognizes words nor links
// identities across independent recordings.
func (lease *SpeechLease) Diarize(ctx context.Context, data []byte, origin dataset.AudioPayloadOrigin, policy dataset.AudioInspectionPolicy, binding RunBinding) (recipecontract.SpeechTurns, runrecord.Run, error) {
	if lease == nil || ctx == nil {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, errors.New("speaker activity: incomplete invocation")
	}
	lease.mu.Lock()
	defer lease.mu.Unlock()
	if lease.released || lease.definition.Task != recipe.TaskDiarization || len(lease.components) != 1 {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, errors.New("speaker activity: admitted diarization lease required")
	}
	component := lease.components[0].Model()
	if component == nil || component.speaker == nil {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, errors.New("speaker activity: component absent")
	}
	return component.speaker.diarize(ctx, data, origin, policy, binding, component)
}

func (p *speakerModel) diarize(ctx context.Context, data []byte, origin dataset.AudioPayloadOrigin, policy dataset.AudioInspectionPolicy, binding RunBinding, component *transcriptionComponent) (recipecontract.SpeechTurns, runrecord.Run, error) {
	if binding.Key == "" || binding.Dataset.Valid() != binding.Split.Valid() || binding.Dataset.Valid() && (binding.Dataset.Kind() != artifact.KindDataset || binding.Split.Kind() != artifact.KindDatasetShard) {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, errors.New("speaker activity: invalid run binding")
	}
	started := time.Now()
	inspection, err := dataset.InspectAudio(ctx, p.repository, data, origin, policy)
	if err != nil {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, err
	}
	phases := completedTranscriptionPhases(elapsedNanoseconds(started), 0, 0, 0)
	inputs := uniqueIDs(p.profile.Model, p.profile.License, binding.Dataset, binding.Split, inspection.Signal.Source.Audio, inspection.Signal.Source.Profile, inspection.SignalID, inspection.PolicyID, inspection.DecisionID)
	fail := func(code string, cause error) (recipecontract.SpeechTurns, runrecord.Run, error) {
		if ctx.Err() != nil {
			return recipecontract.SpeechTurns{}, runrecord.Run{}, cause
		}
		run, err := persistSpeechRun(ctx, p.repository, p.definition.ID, binding, runrecord.OutcomeFailed, inputs, nil, code, elapsedNanoseconds(started), phases)
		return recipecontract.SpeechTurns{}, run, errors.Join(cause, err)
	}
	if inspection.Decision.Outcome != recipecontract.AudioAdmissionAccepted {
		return fail(AudioAdmissionFailure, ErrAudioAdmissionRefused)
	}
	format := inspection.Signal.Format
	if format.SampleRate != uint64(p.profile.Frontend.SampleRate) || format.Channels != 1 || format.Encoding != "pcm-f32le" {
		return fail(AudioFormatFailure, errors.New("speaker activity: decoded format differs"))
	}
	prepare := time.Now()
	samples := inspection.Samples
	if len(samples) == 0 {
		return fail("speaker-normalization-invalid", errors.New("speaker activity: empty decoded audio"))
	}
	// Inspection owns this fresh decoded slice. Scaling does not alter source
	// bytes or the recorded pre-transform signal measurements.
	maximum := slices.Max(samples)
	gain := float32(1) / (maximum + p.profile.MaximumOffset)
	if !checked.PositiveFinite64(float64(gain)) {
		return fail("speaker-normalization-invalid", errors.New("speaker activity: maximum normalization is undefined"))
	}
	for i := range samples {
		samples[i] *= gain
		if !checked.Finite32(samples[i]) {
			return fail("speaker-normalization-invalid", errors.New("speaker activity: non-finite scaled sample"))
		}
	}
	frames := len(samples) / int(p.profile.Frontend.Geometry.HopSamples)
	if frames == 0 {
		return fail("speaker-input-too-short", errors.New("speaker activity: no complete feature hop"))
	}
	features, frames, err := p.frontend.Process(ctx, [][]float32{samples}, int(format.SampleRate), &component.text.Frontend, audiodsp.ProcessOptions{FrameLimit: frames})
	phases[1].DurationNS = elapsedNanoseconds(prepare)
	if err != nil {
		return fail("speaker-frontend-failed", err)
	}
	execute := time.Now()
	probabilities, frames, err := p.activity.Predict(ctx, features, frames, &component.speakers, nil)
	phases[2].DurationNS = elapsedNanoseconds(execute)
	if err != nil {
		return fail("speaker-execution-failed", err)
	}
	post := time.Now()
	turns, err := speakerTurns(ctx, probabilities, frames, p.activity.output.out, uint64(len(samples)), p.profile.Boundary)
	result := recipecontract.SpeechTurns{Source: inspection.Signal.Source, Turns: turns}
	err = cmp.Or(err, result.Validate())
	phases[3].DurationNS = elapsedNanoseconds(post)
	if err != nil {
		return fail("speaker-output-invalid", err)
	}
	output, err := artifact.JSONContent(speechTurnsContract, result)
	if err != nil {
		return fail("speaker-output-invalid", err)
	}
	run, err := runrecord.NewBoundRun(p.definition.ID, runrecord.OutcomeSucceeded, inputs, []artifact.ID{output.Descriptor.ID}, "", binding.CodeCommit, binding.Environment, elapsedNanoseconds(started), phases)
	if err != nil {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, err
	}
	lineage := append(run.Lineage(), artifact.DependencyLineage(output.Descriptor.ID, p.definition.ID, result.Source.Audio, result.Source.Profile)...)
	batch, err := artifact.NewDocumentBatch(binding.Key, []artifact.Content{output, runContent}, lineage, nil)
	if err != nil {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, err
	}
	if _, err := artifact.CommitBatch(ctx, p.repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return recipecontract.SpeechTurns{}, runrecord.Run{}, err
	}
	return result, run, nil
}

// RequireSpeechTurns loads canonical source-bound speaker output.
func RequireSpeechTurns(ctx context.Context, reader artifact.Reader, id artifact.ID) (recipecontract.SpeechTurns, error) {
	var result recipecontract.SpeechTurns
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil || !found {
		return result, errors.Join(errors.New("speaker activity: output absent"), err)
	}
	if err := speechTurnsContract.ValidateContent(content, id); err != nil {
		return result, err
	}
	if err := strictjson.DecodeBytes(content.Data, &result); err != nil {
		return result, err
	}
	if err := result.Validate(); err != nil {
		return result, err
	}
	canonical, err := artifact.JSONContent(speechTurnsContract, result)
	if err != nil || canonical.Descriptor.ID != id {
		return recipecontract.SpeechTurns{}, errors.New("speaker activity: output is not canonical")
	}
	return result, nil
}
