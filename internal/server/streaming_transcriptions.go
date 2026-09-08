package server

import (
	"bufio"
	"cmp"
	"context"
	"errors"
	"io"
	"net/http"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

type transcriptionStreamRequest struct {
	Model string `json:"model"`
	speechrecognition.StreamRequest
}

type transcriptionStreamEvent struct {
	Transcript speechrecognition.TranscriptionChunk `json:"transcript"`
	Output     artifact.ID                          `json:"output"`
	Checkpoint artifact.ID                          `json:"checkpoint"`
	Admission  artifact.ID                          `json:"admission,omitzero"`
}

type transcriptionHTTPStream struct {
	workspace *TranscriptionWorkspace
	session   *capabilityruntime.AudioStreamSession
}

func (workspace *TranscriptionWorkspace) checkStreamActivation(ctx context.Context) error {
	activation, _, err := modelrecipe.ResolveActiveCapability(ctx, workspace.store, workspace.program.Definition().Model, recipe.TaskTranscription)
	if err != nil {
		return err
	}
	if activation.Definition.ID != workspace.policy.Recipe {
		return errors.New("transcription stream: configured recipe is no longer active")
	}
	return nil
}

func (workspace *TranscriptionWorkspace) openTranscriptionStream(ctx context.Context, request speechrecognition.StreamRequest) (*transcriptionHTTPStream, error) {
	if err := workspace.checkStreamActivation(ctx); err != nil {
		return nil, err
	}
	session, err := workspace.sessions.OpenStream(ctx, request)
	if err != nil {
		return nil, err
	}
	if err := workspace.checkStreamActivation(ctx); err != nil {
		return nil, errors.Join(err, session.Close(context.WithoutCancel(ctx)))
	}
	return &transcriptionHTTPStream{workspace: workspace, session: session}, nil
}

func (stream *transcriptionHTTPStream) process(ctx context.Context, chunk workflowruntime.AudioStreamChunk) (transcriptionStreamEvent, error) {
	var event transcriptionStreamEvent
	w := stream.workspace
	if err := w.checkStreamActivation(ctx); err != nil {
		return event, err
	}
	if chunk.Audio.Valid() {
		descriptor, found, err := w.store.Artifact(ctx, chunk.Audio)
		if err != nil || !found || descriptor.Size > w.policy.Inspection.MaximumEncodedBytes ||
			chunk.Span.Start >= chunk.Span.End || chunk.Span.End-chunk.Span.Start > w.policy.Inspection.MaximumSamples {
			return event, errors.Join(errors.New("transcription stream: chunk absent or exceeds server admission bounds"), err)
		}
		content, found, err := artifact.ReadContent(ctx, w.store, chunk.Audio)
		if err != nil || !found {
			return event, errors.Join(errors.New("transcription stream: input content absent"), err)
		}
		inspection, err := dataset.InspectAudio(ctx, w.store, content.Data, dataset.AudioPayloadOrigin{Container: chunk.Audio}, w.policy.Inspection)
		if err != nil {
			return event, err
		}
		// The inspection owner returns decoded samples only for accepted input.
		if len(inspection.Samples) == 0 {
			return event, speechrecognition.ErrAudioAdmissionRefused
		}
		event.Admission = inspection.DecisionID
	}
	result, err := stream.session.Process(ctx, chunk)
	if err != nil {
		return event, err
	}
	event.Transcript, err = speechrecognition.RequireTranscriptionChunk(ctx, w.store, result.Output)
	if err != nil {
		return event, err
	}
	batch, err := stream.session.CheckpointBatch("transcription-http/" + result.State.String())
	if err != nil {
		return event, err
	}
	if event.Admission.Valid() {
		batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(result.Output, event.Admission)...)
	}
	if _, err := artifact.CommitBatch(ctx, w.store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return event, err
	}
	event.Output, event.Checkpoint = result.Output, batch.Contents[0].Descriptor.ID
	return event, nil
}

// Native live input is an explicit NDJSON extension of the transcription route,
// not the OpenAI multipart wire format. Each subsequent line names an already
// stored waveform chunk; SSE output carries durable resume authority per event.
func (h *Handler) nativeStreamingTranscriptions(response http.ResponseWriter, request *http.Request) {
	workspace, ok := h.generator.(WorkflowWorkspaceAPI)
	if !ok || h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "transcription workspace is unavailable")
		return
	}
	controller := http.NewResponseController(response)
	if request.ProtoMajor == 1 {
		if err := controller.EnableFullDuplex(); err != nil {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "bidirectional transcription transport is unavailable")
			return
		}
	}
	// Request cancellation must interrupt blocked reads and writes, then join
	// the callback before net/http can reuse this connection. No polling worker.
	interrupted := make(chan struct{})
	stopInterrupt := context.AfterFunc(request.Context(), func() {
		_ = controller.SetReadDeadline(time.Now())
		_ = controller.SetWriteDeadline(time.Now())
		close(interrupted)
	})
	defer func() {
		if !stopInterrupt() {
			<-interrupted
		}
	}()
	// A final application chunk need not coincide with request-body EOF. Close
	// the body while this handler still owns the connection, interrupting any
	// attempted drain. Otherwise net/http can begin its next-request read while
	// a late EOF callback starts a background read during finishRequest.
	defer func() {
		_ = controller.SetReadDeadline(time.Now())
		_ = request.Body.Close()
	}()
	scanner := bufio.NewScanner(request.Body)
	scanner.Buffer(nil, maxRequestBytes)
	if !scanner.Scan() {
		writeInvalidRequest(response, errors.Join(errors.New("transcription stream: missing initial request"), scanner.Err()))
		return
	}
	var input transcriptionStreamRequest
	if err := strictjson.DecodeBytes(scanner.Bytes(), &input); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	capabilities, err := workspace.WorkflowCapabilities(request.Context(), WorkflowGeneration)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	capability, err := selectNativeWorkflowCapability(capabilities, recipe.TaskTranscription, input.Model)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	if capability.transcriptionStream == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "selected recipe has no native audio stream")
		return
	}
	session, err := capability.transcriptionStream(request.Context(), input.StreamRequest)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	defer session.session.Close(context.WithoutCancel(request.Context()))
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	emitter := newSSEEmitter(request.Context(), response, flusher)
	if err := emitter.named(streamEventCreated, input); err != nil {
		return
	}
	for scanner.Scan() {
		var chunk workflowruntime.AudioStreamChunk
		err := strictjson.DecodeBytes(scanner.Bytes(), &chunk)
		var event transcriptionStreamEvent
		if err == nil {
			event, err = session.process(request.Context(), chunk)
		}
		if err != nil {
			_ = emitter.named(streamEventError, map[string]string{"message": err.Error()})
			return
		}
		name := streamEventToken
		if chunk.Final {
			name = streamEventDone
		}
		if err := emitter.named(name, event); err != nil || chunk.Final {
			return
		}
	}
	err = cmp.Or(scanner.Err(), io.ErrUnexpectedEOF)
	_ = emitter.named(streamEventError, map[string]string{"message": err.Error()})
}
