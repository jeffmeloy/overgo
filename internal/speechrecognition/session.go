package speechrecognition

import (
	"context"
	"errors"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechactivity"
	"overgo/internal/workflowruntime"
)

// StreamRequest binds one source and optional completed cursor for native
// transcription session admission. Numerical state remains in the checkpoint.
type StreamRequest struct {
	Source recipecontract.AudioReference `json:"source"`
	Resume artifact.ID                   `json:"resume,omitzero"`
}

// Session owns the compiled component lifetimes for one CPU speech
// recipe. Evaluation and serving share this owner, its exclusive workspaces and
// cancellation behavior. Memory is a per-component numeric ceiling, not RSS.
type Session struct {
	repository artifact.Repository
	definition recipe.Definition
	base       recipe.Definition
	activity   recipe.Definition
	resources  modelrecipe.ComponentSessionPlan
	memory     uint64
	director   *capabilityruntime.ModelSessionDirector[struct{}, *transcriptionComponent, struct{}]
}

type transcriptionComponent struct {
	transcriber *transcriptionModel
	detector    *speechactivity.Detector
	text        transcriptionWorkspace
	activity    speechactivity.DetectionWorkspace
	stream      transcriptionStreamWorkspace
	alignment   CTCAlignmentWorkspace
}

// Close releases the component's model and numeric workspace references.
func (component *transcriptionComponent) Close(context.Context) error {
	*component = transcriptionComponent{}
	return nil
}

// LoadSession validates one complete recipe and preloads its declared model
// components through the existing bounded session director. A partial load
// releases every earlier component before returning an error.
func LoadSession(ctx context.Context, repository artifact.Repository, id artifact.ID, memoryBytes uint64) (*Session, error) {
	if ctx == nil || repository == nil || memoryBytes == 0 {
		return nil, errors.New("transcription session: incomplete load")
	}
	definition, err := recipe.RequireDefinition(ctx, repository, id)
	if err != nil {
		return nil, err
	}
	base, activity, err := modelrecipe.SpeechComponents(definition)
	if err != nil {
		return nil, err
	}
	if err := requireRecipeDependencyLineage(ctx, repository, definition); err != nil {
		return nil, err
	}
	if definition.Task == recipe.TaskAlignment {
		profile, _ := definition.PrimaryDependency(recipe.DependencyDerivationProfile)
		if _, err := RequireAlignmentProfile(ctx, repository, profile); err != nil {
			return nil, err
		}
		processor, _ := base.PrimaryDependency(recipe.DependencyProcessorProfile)
		declaration, err := RequireExecutionProfile(ctx, repository, processor)
		if err != nil || declaration.Transducer != nil {
			return nil, errors.Join(errors.New("alignment: recurrent decoding has no CTC frame contract"), err)
		}
	}
	resources, err := modelrecipe.CompileDefinitionSessionPlan(ctx, repository, definition)
	if err != nil {
		return nil, err
	}
	if activity.ID.Valid() {
		id, _ := activity.PrimaryDependency(recipe.DependencyProcessorProfile)
		profile, err := speechactivity.RequireProfile(ctx, repository, id)
		if err != nil {
			return nil, err
		}
		contractID, _ := base.PrimaryDependency(recipe.DependencyProfile)
		contract, err := modelrecipe.RequireAudioContract(ctx, repository, contractID)
		if err != nil || profile.Offline == nil || contract.Format.SampleRate != uint64(profile.Frontend.SampleRate) || contract.Format.Channels != 1 || contract.Format.Encoding != "pcm-f32le" {
			return nil, errors.Join(errors.New("transcription session: activity and recognizer formats or offline policy differ"), err)
		}
	}
	director, err := capabilityruntime.NewComponentSessionDirector[*transcriptionComponent](string(definition.Task), "cpu", len(resources.Components))
	if err != nil {
		return nil, err
	}
	session := &Session{repository: repository, definition: definition, base: base, activity: activity, resources: resources, memory: memoryBytes, director: director}
	for _, component := range resources.Components {
		lease, err := director.LeaseComponent(ctx, component, func(ctx context.Context) (*transcriptionComponent, error) { return session.load(ctx, component.Module) })
		if err != nil {
			return nil, errors.Join(err, session.Close(context.WithoutCancel(ctx)))
		}
		if err := lease.Release(); err != nil {
			return nil, errors.Join(err, session.Close(context.WithoutCancel(ctx)))
		}
	}
	return session, nil
}

func (session *Session) load(ctx context.Context, module recipe.ModuleID) (*transcriptionComponent, error) {
	component := &transcriptionComponent{}
	var err error
	switch module {
	case modelrecipe.ModuleTranscribeAudio, modelrecipe.ModuleTranscribeSegments, modelrecipe.ModuleAlignAudio:
		component.transcriber, err = loadTranscriber(ctx, session.repository, session.base, session.memory)
	case modelrecipe.ModuleDetectActivity:
		component.detector, err = speechactivity.LoadDetector(ctx, session.repository, session.activity.ID, session.memory)
	default:
		err = errors.New("transcription session: unknown component")
	}
	return component, err
}

// TranscriptionLease holds one complete recognition or alignment invocation. Its methods
// serialize workspace use and release; a released lease cannot execute again.
type TranscriptionLease struct {
	mu         sync.Mutex
	definition recipe.Definition
	components []*capabilityruntime.SessionLease[*transcriptionComponent]
	released   bool
	err        error
}

// Lease acquires components in compiled order. The serving caller can recheck
// activation after admission; evaluation need not activate a candidate recipe.
func (session *Session) Lease(ctx context.Context) (*TranscriptionLease, error) {
	if session == nil || session.director == nil || ctx == nil {
		return nil, errors.New("transcription session: incomplete execution")
	}
	result := &TranscriptionLease{definition: session.definition}
	for _, component := range session.resources.Components {
		lease, leaseErr := session.director.LeaseComponent(ctx, component, func(ctx context.Context) (*transcriptionComponent, error) { return session.load(ctx, component.Module) })
		if leaseErr != nil {
			return nil, errors.Join(leaseErr, result.Release())
		}
		result.components = append(result.components, lease)
	}
	return result, nil
}

// OpenStream uses the existing component lease for one native recurrent audio
// stream. Checkpoints restore only the same source and recipe; Close and final
// processing release that lease through the shared stream lifecycle owner.
func (session *Session) OpenStream(ctx context.Context, request StreamRequest) (*capabilityruntime.AudioStreamSession, error) {
	if session == nil || session.director == nil || session.activity.ID.Valid() || session.definition.Task != recipe.TaskTranscription {
		return nil, errors.New("transcription stream: standalone recognizer session required")
	}
	return capabilityruntime.OpenAudioStream(ctx, session.repository, func(ctx context.Context) (capabilityruntime.AudioStreamLease, error) {
		lease, err := session.Lease(ctx)
		if err != nil {
			return capabilityruntime.AudioStreamLease{}, err
		}
		component := lease.components[0].Model()
		if component.transcriber == nil || component.transcriber.stream == nil {
			return capabilityruntime.AudioStreamLease{}, errors.Join(errors.New("transcription stream: recipe has no native recurrent stream"), lease.Release())
		}
		return capabilityruntime.AudioStreamLease{
			Processor: func(ctx context.Context, _ int, work workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
				lease.mu.Lock()
				defer lease.mu.Unlock()
				if lease.released {
					return workflowruntime.AudioStreamResult{}, errors.New("transcription stream: lease released")
				}
				return component.transcriber.processStream(ctx, work, &component.stream)
			}, Release: lease.Release,
		}, nil
	}, request.Source, request.Resume)
}

// Transcribe decodes one source once, traverses its declared activity spans and
// publishes output under the complete recipe identity. It never copies corpus
// audio into the store or changes the recipe's boundaries to improve a score.
func (lease *TranscriptionLease) Transcribe(ctx context.Context, data []byte, origin dataset.AudioPayloadOrigin, policy dataset.AudioInspectionPolicy, binding RunBinding) (result recipecontract.Transcription, run runrecord.Run, err error) {
	if lease == nil || ctx == nil {
		return result, run, errors.New("transcription session: incomplete lease execution")
	}
	lease.mu.Lock()
	defer lease.mu.Unlock()
	if lease.released || lease.definition.Task != recipe.TaskTranscription {
		return result, run, errors.New("transcription session: an admitted transcription lease is required")
	}
	var text, activity *transcriptionComponent
	for _, component := range lease.components {
		if component.Model().transcriber != nil {
			text = component.Model()
		} else {
			activity = component.Model()
		}
	}
	if text == nil {
		return result, run, errors.New("transcription session: recognizer absent")
	}
	bound := *text.transcriber
	bound.recipe = lease.definition
	var detector *speechactivity.Detector
	var workspace *speechactivity.DetectionWorkspace
	if activity != nil {
		detector, workspace = activity.detector, &activity.activity
	}
	return bound.transcribe(ctx, data, origin, policy, &text.text, binding, detector, workspace)
}

// Release returns every component exactly once, in reverse acquisition order.
func (lease *TranscriptionLease) Release() error {
	if lease == nil {
		return nil
	}
	lease.mu.Lock()
	defer lease.mu.Unlock()
	if !lease.released {
		lease.released = true
		for i := len(lease.components) - 1; i >= 0; i-- {
			lease.err = errors.Join(lease.err, lease.components[i].Release())
		}
		lease.components = nil
	}
	return lease.err
}

// Snapshot reports actual component residency and admission, without estimating
// process peak memory or claiming unexecuted concurrent throughput.
func (session *Session) Snapshot() capabilityruntime.SessionSnapshot {
	if session == nil || session.director == nil {
		return capabilityruntime.SessionSnapshot{}
	}
	return session.director.Snapshot()
}

// Close drains outstanding component leases and releases all numeric state.
func (session *Session) Close(ctx context.Context) error {
	if session == nil || session.director == nil {
		return nil
	}
	return session.director.Close(ctx)
}
