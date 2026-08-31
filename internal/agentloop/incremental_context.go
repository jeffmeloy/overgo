package agentloop

import (
	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
)

// sessionContext is what one attempt hands its resumable session: either the
// bounded full context (a session with no held boundary rebuilds
// deterministically) or the stable cursor delta — identity references for
// arcs the session already holds plus the fresh arcs it lacks. Both shapes
// carry every admitted stimulus of the boundary exactly once.
type sessionContext struct {
	full     bool
	boundary artifact.ID
	reused   []artifact.ID
	fresh    []dataset.InteractionSelectionSource
}

// incrementalSessionContext derives the attempt handoff and advances the
// session's held boundary. Continuity is the session's own held state: a
// stateless or retired session receives the bounded full context, while a
// continuous session receives only the verified cursor delta — and a delta
// that cannot prove it preserves every admitted stimulus refuses rather
// than degrading to silence.
func incrementalSessionContext(
	session *Session,
	stimulus runrecord.AttemptStimulusBoundary,
) (sessionContext, error) {
	held := session.held
	if held == nil {
		session.held = &stimulus
		return sessionContext{
			full: true, boundary: stimulus.ID, fresh: stimulus.Selection.Sources,
		}, nil
	}
	delta, err := dataset.SelectIncremental(held.Selection, stimulus.Selection)
	if err != nil {
		return sessionContext{}, err
	}
	if err := runrecord.VerifyIncrementalStimulus(*held, stimulus, delta); err != nil {
		return sessionContext{}, err
	}
	session.held = &stimulus
	return sessionContext{
		boundary: stimulus.ID, reused: delta.Reused, fresh: delta.Fresh,
	}, nil
}
