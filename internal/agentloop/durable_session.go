package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/runrecord"
)

// ErrReplayDivergence marks a resumed session whose next request differs
// from the entry its log recorded at that step: a determinism defect.
var ErrReplayDivergence = errors.New("agent loop: durable replay diverged from the recorded log")

// durableSessionUnit: the durable unit identity of one agent session.
func durableSessionUnit(id string) (artifact.ID, error) {
	return artifact.IdentifyBytes(artifact.KindEvidence, []byte("overgo/agent-session/"+id))
}

// OpenDurableSession opens a new invocation of the session's durable unit:
// the session replays its tool calls from the log from step one, reads
// recorded calls instead of executing them, and every earlier invocation of
// the same session is stale from here on.
func (c *Coordinator) OpenDurableSession(ctx context.Context, id string) (*Session, error) {
	if c == nil || ctx == nil || id == "" {
		return nil, errors.New("agent loop: nil context or empty session id")
	}
	unit, err := durableSessionUnit(id)
	if err != nil {
		return nil, err
	}
	attempt, err := runrecord.OpenDurableAttempt(ctx, c.store, unit)
	if err != nil {
		return nil, err
	}
	return &Session{ID: id, attempt: &attempt}, nil
}

// childKey: the log key of the session's next step.
func childKey(session *Session) string { return strconv.Itoa(session.Steps + 1) }

// replayChild: the next step's recorded child attempt, if the session is
// durable and the log holds one; a matching request is replayed from the
// recorded interaction, a differing one is recorded as a divergence
// failure and refused, and a stale invocation is refused by the log.
func (c *Coordinator) replayChild(
	ctx context.Context, session *Session, manual agenttool.Manual, arguments json.RawMessage, effect agenttool.InvocationEffect,
) (json.RawMessage, bool, error) {
	if session.attempt == nil {
		return nil, false, nil
	}
	entry, found, err := session.attempt.Lookup(ctx, c.store, runrecord.DurableChild, childKey(session))
	if err != nil || !found {
		return nil, false, err
	}
	interaction, err := runrecord.RequireInteraction(ctx, c.store, entry.Result)
	if err != nil {
		return nil, false, err
	}
	transcript, err := runrecord.RequireInteractionTranscript(ctx, c.store, interaction.Message)
	if err != nil {
		return nil, false, err
	}
	var recorded *runrecord.InteractionToolCall
	var result string
	for _, message := range transcript.Messages {
		if len(message.ToolCalls) != 0 && recorded == nil {
			recorded = &message.ToolCalls[0]
		}
		if message.Role == string(inference.ChatRoleTool) {
			result = message.Content
		}
	}
	if recorded == nil {
		return nil, false, fmt.Errorf("agent loop: recorded step %s of session %s carries no tool call", childKey(session), session.ID)
	}
	if recorded.Manual != manual.ID || recorded.Arguments != string(arguments) {
		message := fmt.Sprintf("session %s step %s requested %s %s but the log recorded %s %s",
			session.ID, childKey(session), manual.Name, arguments, recorded.Name, recorded.Arguments)
		if _, err := runrecord.PublishFailureObservation(ctx, c.store, runrecord.FailureObservation{
			Source: "agent-loop", Message: "durable replay divergence: " + message, ObservedUnixNS: time.Now().UnixNano(),
		}); err != nil {
			return nil, false, err
		}
		return nil, false, fmt.Errorf("%w: %s", ErrReplayDivergence, message)
	}
	session.Interaction = entry.Result
	session.Steps++
	if manual.Effect == agenttool.EffectInspection {
		session.Inspection, session.InspectionEffect = entry.Result, effect.ID
	} else {
		session.Inspection, session.InspectionEffect = artifact.ID{}, artifact.ID{}
	}
	return json.RawMessage(result), true, nil
}

// recordChild: the executed step's interaction becomes the child entry of
// this invocation; a superseded invocation cannot complete the step.
func (c *Coordinator) recordChild(ctx context.Context, session *Session, interaction artifact.ID) error {
	if session.attempt == nil {
		return nil
	}
	_, err := session.attempt.Record(context.WithoutCancel(ctx), c.store, runrecord.DurableChild, childKey(session), interaction)
	return err
}
