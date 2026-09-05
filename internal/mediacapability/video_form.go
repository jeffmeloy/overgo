package mediacapability

import (
	"context"
	"encoding/json"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
)

// resolveEditForm completes a clip-form LiveEdit request before the session
// director sees it: the page names a prompt, a seed and a source clip
// artifact, and the runtime's request carries the compiled condition and
// the decoded pixels. A request already in that form passes through.
func resolveEditForm(next capabilityruntime.Executor) capabilityruntime.Executor {
	return func(ctx context.Context, store artifact.Repository, path string, selection modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
		var request latentvideo.ReferenceEditRequest
		if err := json.Unmarshal([]byte(input), &request); err != nil {
			return nil, err
		}
		if !request.ClipForm() {
			return next(ctx, store, path, selection, input)
		}
		resolved, err := latentvideo.ResolveClipRequest(ctx, store, path, request)
		if err != nil {
			return nil, err
		}
		encoded, err := json.Marshal(resolved)
		if err != nil {
			return nil, err
		}
		return next(ctx, store, path, selection, string(encoded))
	}
}
