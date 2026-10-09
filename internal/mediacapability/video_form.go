package mediacapability

import (
	"context"
	"encoding/json"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
)

// editInput serves a LiveEdit request held as a value, recorded as the
// content given (capabilityruntime ModelSessionDirector.ExecuteInput).
type editInput func(
	context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection,
	latentvideo.ReferenceEditRequest, artifact.Content,
) (any, error)

// resolveEditForm completes a clip-form LiveEdit request before the session
// director sees it: the page names a prompt, a seed and a source clip
// artifact, and the runtime's request carries the compiled condition and
// the decoded pixels. A request already in that form passes through as the
// text it arrived as. The resolved request is handed to the director as the
// value it is, and the run keeps the clip form as its request: it names the
// edit exactly, resolves again to the same one, and is small, where the
// resolved request is hundreds of megabytes as text and over the bound a run
// retains whole.
func resolveEditForm(
	resolve func(context.Context, artifact.Reader, string, latentvideo.ReferenceEditRequest) (latentvideo.ReferenceEditRequest, error),
	next capabilityruntime.Executor,
	serve editInput,
) capabilityruntime.Executor {
	return func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		var request latentvideo.ReferenceEditRequest
		if err := json.Unmarshal([]byte(input), &request); err != nil {
			return nil, err
		}
		if !request.ClipForm() {
			return next(ctx, store, path, selection, input)
		}
		resolved, err := resolve(ctx, store, path, request)
		if err != nil {
			return nil, err
		}
		kept, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.video-gen-input.v1"), request)
		if err != nil {
			return nil, err
		}
		return serve(ctx, store, path, selection, resolved, kept)
	}
}
