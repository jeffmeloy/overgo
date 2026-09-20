package closureledger

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// ActiveAliasPrefix scopes current closure bindings.
const ActiveAliasPrefix = "closure/active/"

// ActiveAlias identifies one source declaration across evidence revisions.
func ActiveAlias(binding SourceBinding) (string, error) {
	if !validBinding(binding) {
		return "", errors.New("closure ledger: invalid active binding")
	}
	digest := sha256.Sum256([]byte(bindingDeclarationKey(binding)))
	return ActiveAliasPrefix + hex.EncodeToString(digest[:]), nil
}

// ActiveBindings loads every closure document an active alias names, with
// the aliases that name it: the ledger a store holds, read the same way for
// the store in service and for a candidate before it goes into service.
func ActiveBindings(ctx context.Context, store *overgodb.Store) ([]Document, map[string]artifact.ID, error) {
	var documents []Document
	aliases := map[string]artifact.ID{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: MediaType, Schema: Schema,
		}}, AliasPrefixes: []string{ActiveAliasPrefix}, Order: overgodb.DocumentOldestFirst,
	}, Parse, func(view overgodb.DocumentView, document Document) error {
		if document.ID != view.Content.Descriptor.ID {
			return errors.New("closure ledger: active document identity mismatch")
		}
		documents = append(documents, document)
		for _, alias := range view.Aliases {
			aliases[alias] = document.ID
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return documents, aliases, nil
}

func activeAliasForKey(key string) string {
	digest := sha256.Sum256([]byte(key))
	return ActiveAliasPrefix + hex.EncodeToString(digest[:])
}

// ResolveActiveBinding requires exact current source and value evidence.
func ResolveActiveBinding(
	ctx context.Context,
	reader artifact.Reader,
	binding SourceBinding,
	value json.RawMessage,
) (Document, bool, error) {
	alias, err := ActiveAlias(binding)
	if err != nil {
		return Document{}, false, err
	}
	id, found, err := artifact.ResolveAlias(ctx, reader, alias)
	if err == nil && !found && binding.StructuralID != "" {
		id, found, err = artifact.ResolveAlias(ctx, reader, activeAliasForKey(legacyBindingDeclarationKey(binding)))
	}
	if err != nil || !found {
		return Document{}, found, err
	}
	document, err := documentCodec.Require(ctx, reader, id)
	if err != nil {
		return Document{}, false, err
	}
	canonical, err := canonicalValue(value)
	if err != nil {
		return Document{}, false, err
	}
	matches := 0
	var active SourceBinding
	for _, candidate := range document.Bindings {
		if sameDeclaration(candidate, binding) {
			matches++
			active = candidate
		}
	}
	if matches != 1 {
		return Document{}, false, fmt.Errorf("closure ledger: active alias has %d declaration bindings", matches)
	}
	if !sameEvidence(active, binding) || !bytes.Equal(document.Value, canonical) {
		return Document{}, false, errors.New("closure ledger: active evidence is stale")
	}
	return document, true, nil
}

func bindingDeclarationKey(binding SourceBinding) string {
	if binding.StructuralID != "" {
		return string(binding.Kind) + "\x00" + binding.Package + "\x00" + binding.File + "\x00" +
			binding.Scope + "\x00" + binding.StructuralID
	}
	return legacyBindingDeclarationKey(binding)
}

func legacyBindingDeclarationKey(binding SourceBinding) string {
	return string(binding.Kind) + "\x00" + binding.Package + "\x00" + binding.File + "\x00" +
		binding.Scope + "\x00" + strconv.Itoa(binding.Line) + "\x00" + binding.Name
}

func sameDeclaration(left, right SourceBinding) bool {
	if left.StructuralID != "" && right.StructuralID != "" {
		return bindingDeclarationKey(left) == bindingDeclarationKey(right)
	}
	return left.Kind == right.Kind && left.Package == right.Package && left.File == right.File &&
		left.Scope == right.Scope && left.Name == right.Name
}

func sameEvidence(left, right SourceBinding) bool {
	return sameDeclaration(left, right) && left.Expression == right.Expression &&
		left.SourceID == right.SourceID && left.CallsiteID == right.CallsiteID
}
