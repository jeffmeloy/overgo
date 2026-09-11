package plan

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
)

// Dispatch is the current row as data: the harness reads its fields and
// Line is the one-line prose rendered from the same fields.
type Dispatch struct {
	Complete  bool   `json:"complete"`
	Item      string `json:"item,omitzero"`
	Step      string `json:"step,omitzero"`
	ItemTitle string `json:"item_title,omitzero"`
	StepTitle string `json:"step_title,omitzero"`
	Verify    string `json:"verify,omitzero"`
	Line      string `json:"line"`
}

// DispatchCachePath keeps the last resolved dispatch beside the inputs it
// was resolved from, so a turn's several readers resolve the completion
// authority once.
const DispatchCachePath = "docs/.dispatch"

// DispatchOf resolves the current row for role under a resolved authority.
func DispatchOf(document Plan, role string, authority CompletionAuthority) Dispatch {
	item, step, ok := Current(document, role, authority)
	if !ok {
		return Dispatch{Complete: true, Line: dispatchLine(Dispatch{Complete: true})}
	}
	dispatch := Dispatch{Item: item.ID, Step: step.ID, ItemTitle: item.Title, StepTitle: step.Title, Verify: step.Verify}
	dispatch.Line = dispatchLine(dispatch)
	return dispatch
}

func dispatchLine(dispatch Dispatch) string {
	switch {
	case dispatch.Complete:
		return "plan complete: every item is done"
	case dispatch.Step == ".":
		return fmt.Sprintf("%s: %s -- open the rung (define its steps)", dispatch.Item, dispatch.ItemTitle)
	default:
		return fmt.Sprintf("%s / %s: %s -- %s", dispatch.Item, dispatch.Step, dispatch.ItemTitle, dispatch.StepTitle)
	}
}

// dispatchCache binds a dispatch to every input it was resolved from.
type dispatchCache struct {
	Head          string            `json:"head"`
	PlanDigest    string            `json:"plan_digest"`
	StoreHead     artifact.CommitID `json:"store_head"`
	StoreSequence uint64            `json:"store_sequence"`
	Role          string            `json:"role"`
	Dispatch      Dispatch          `json:"dispatch"`
}

// ResolveDispatch resolves the current row for the repository at root and
// role, reusing the cached dispatch when HEAD, the plan bytes, the store
// head and the role are the ones it was resolved from.
func ResolveDispatch(ctx context.Context, root, role string) (Dispatch, error) {
	role, err := AutomationRole(role)
	if err != nil {
		return Dispatch{}, err
	}
	repository, head, err := resolveCompletionRevision(ctx, root, "HEAD")
	if err != nil {
		return Dispatch{}, err
	}
	planPath := filepath.Join(repository, filepath.FromSlash(Path))
	raw, err := os.ReadFile(planPath)
	if err != nil {
		return Dispatch{}, err
	}
	// The cache takes the plan file's own permissions.
	planInfo, err := os.Stat(planPath)
	if err != nil {
		return Dispatch{}, err
	}
	digest := sha256.Sum256(raw)
	store, err := overgodb.OpenReadOnly(filepath.Join(repository, gitauthority.CanonicalOvergoDBDirectory))
	if err != nil {
		return Dispatch{}, err
	}
	defer store.Close()
	storeHead, sequence := store.Head()
	key := dispatchCache{Head: head, PlanDigest: hex.EncodeToString(digest[:]), StoreHead: storeHead, StoreSequence: sequence, Role: role}
	cachePath := filepath.Join(repository, filepath.FromSlash(DispatchCachePath))
	if cached, ok := cachedDispatch(cachePath, key); ok {
		return cached, nil
	}
	document, err := Load(planPath)
	if err != nil {
		return Dispatch{}, err
	}
	authority, err := ResolveCompletionAuthority(ctx, repository, "HEAD", document, store)
	if err != nil {
		return Dispatch{}, err
	}
	key.Dispatch = DispatchOf(document, role, authority)
	// The cache is a convenience: a write that fails leaves the next reader resolving again.
	_ = jsonfile.Write(cachePath, key, planInfo.Mode().Perm())
	return key.Dispatch, nil
}

// cachedDispatch returns the cached dispatch when every input matches key.
func cachedDispatch(path string, key dispatchCache) (Dispatch, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Dispatch{}, false
	}
	var cached dispatchCache
	if err := json.Unmarshal(raw, &cached); err != nil {
		return Dispatch{}, false
	}
	key.Dispatch = cached.Dispatch
	if cached != key || cached.Dispatch.Line == "" {
		return Dispatch{}, false
	}
	return cached.Dispatch, true
}
