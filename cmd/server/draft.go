package main

import (
	"context"
	"errors"

	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

// activeDraft resolves the draft the model's active speculation recipe binds, none when the store declares none: the
// draft a server started without -draft speculates through, so the swap proxy, the desktop launcher and the browser
// lane serve the model's draft as they serve its projector. A model the store cannot resolve gets none here, and its
// open reports why.
func activeDraft(ctx context.Context, repository, path string) (string, error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return "", err
	}
	defer store.Close()
	memo := discovery.LoadMemo(ctx, store)
	loaded, err := modelrecipe.ResolveClaimedGGUF(ctx, store, path, memo.Claim)
	if err != nil {
		return "", nil
	}
	identity, err := loaded.Identity()
	if err = errors.Join(err, loaded.Close()); err != nil {
		return "", err
	}
	draft, _, err := discovery.ActiveDraft(ctx, store, identity.Model, memo)
	return draft, err
}
