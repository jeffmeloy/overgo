package discovery

import (
	"context"
	"errors"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
)

const (
	// IdentityEvidenceMediaType names the persisted digest memo document.
	IdentityEvidenceMediaType = "application/vnd.overgo.file-identity-memo+json"
	// IdentityEvidenceSchema versions the persisted digest memo shape.
	IdentityEvidenceSchema = "overgo/file-identity-memo/v1"
	// IdentityEvidenceAlias is the store's pointer to the latest memo.
	IdentityEvidenceAlias = "discovery/file-identities/latest"
)

// identityRow is one persisted digest claim: this digest held for these
// bytes while the file carried this size and modification time. A row
// whose stat no longer matches simply misses and the file re-hashes,
// so a stale row can never serve a wrong identity.
type identityRow struct {
	Path       string      `json:"path"`
	Kind       string      `json:"kind"`
	Digest     artifact.ID `json:"digest"`
	Size       uint64      `json:"size"`
	StatSize   int64       `json:"stat_size"`
	ModifiedNS int64       `json:"modified_ns"`
}

// IdentityEvidence is the persisted digest memo: the identities the
// interactive surfaces hashed, so a fresh process answers the catalog
// from stat checks instead of re-hashing tens of gigabytes.
type IdentityEvidence struct {
	Rows []identityRow `json:"rows"`
	ID   artifact.ID   `json:"-"`
}

var identityEvidenceCodec = artifact.JSONDocumentCodec(
	"file identity memo", artifact.KindEvidence, IdentityEvidenceMediaType, IdentityEvidenceSchema,
	validateIdentityEvidence, func(value IdentityEvidence) artifact.ID { return value.ID },
	func(value *IdentityEvidence, id artifact.ID) { value.ID = id },
	func(value IdentityEvidence) IdentityEvidence { value.Rows = slices.Clone(value.Rows); return value },
)

func validateIdentityEvidence(value *IdentityEvidence) error {
	if value == nil {
		return errors.New("discovery: nil identity evidence")
	}
	previous := ""
	for _, row := range value.Rows {
		if row.Path == "" || row.Kind == "" || !row.Digest.Valid() || row.Size == 0 {
			return errors.New("discovery: invalid identity row")
		}
		if !filepath.IsAbs(row.Path) && !strings.Contains(row.Path, "/") && !strings.Contains(row.Path, "\\") {
			return errors.New("discovery: identity row path is not a location")
		}
		key := row.Path + "\x00" + row.Kind
		if key <= previous {
			return errors.New("discovery: unordered identity rows")
		}
		previous = key
	}
	return nil
}

// LoadMemo returns a memo seeded from the store's persisted identity
// evidence; an absent or unreadable document yields an empty memo, the
// same cold start as before persistence existed. Every lookup still
// revalidates against the live stat, so loaded rows keep the exact
// claim: digest of these bytes as last read.
func LoadMemo(ctx context.Context, reader artifact.Reader) *Memo {
	memo := NewMemo()
	id, found, err := artifact.ResolveAlias(ctx, reader, IdentityEvidenceAlias)
	if err != nil || !found {
		return memo
	}
	evidence, found, err := identityEvidenceCodec.Read(ctx, reader, id)
	if err != nil || !found {
		return memo
	}
	memo.mu.Lock()
	defer memo.mu.Unlock()
	for _, row := range evidence.Rows {
		memo.entries[row.Path+"\x00"+row.Kind] = memoEntry{
			identity: fileIdentity{id: row.Digest, size: row.Size, present: true},
			size:     row.StatSize,
			modified: time.Unix(0, row.ModifiedNS),
		}
	}
	memo.loaded = len(memo.entries)
	return memo
}

// PublishMemo commits the memo's identities to the store and rebinds
// the alias. Publishing only happens when hashing added rows since the
// memo was loaded; an unchanged memo is a no-op so interactive
// surfaces can call this after every catalog pass.
func PublishMemo(ctx context.Context, repository artifact.Repository, memo *Memo) error {
	if memo == nil {
		return nil
	}
	memo.mu.Lock()
	dirty := memo.dirty
	rows := make([]identityRow, 0, len(memo.entries))
	for key, entry := range memo.entries {
		if !entry.identity.present {
			continue
		}
		path, kind, ok := strings.Cut(key, "\x00")
		if !ok {
			continue
		}
		rows = append(rows, identityRow{
			Path: path, Kind: kind, Digest: entry.identity.id, Size: entry.identity.size,
			StatSize: entry.size, ModifiedNS: entry.modified.UnixNano(),
		})
	}
	memo.mu.Unlock()
	if !dirty || len(rows) == 0 {
		return nil
	}
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Path+"\x00"+rows[i].Kind < rows[j].Path+"\x00"+rows[j].Kind
	})
	evidence, err := identityEvidenceCodec.New(IdentityEvidence{Rows: rows})
	if err != nil {
		return err
	}
	var previous *artifact.ID
	if current, bound, err := artifact.ResolveAlias(ctx, repository, IdentityEvidenceAlias); err != nil {
		return err
	} else if bound {
		if current == evidence.ID {
			return nil
		}
		previous = &current
	}
	batch, err := identityEvidenceCodec.Batch(
		"discovery/file-identities/"+evidence.ID.String(), evidence, nil,
		[]artifact.AliasBinding{{Name: IdentityEvidenceAlias, Target: evidence.ID, Previous: previous}},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	if errors.Is(err, artifact.ErrNoChange) {
		err = nil
	}
	if err == nil {
		memo.mu.Lock()
		memo.dirty = false
		memo.mu.Unlock()
	}
	return err
}
