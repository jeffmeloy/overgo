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
	"strings"

	"overgo/internal/artifact"
)

const activeAliasPrefix = "closure/active/"

func IsActiveAlias(name string) bool {
	if !strings.HasPrefix(name, activeAliasPrefix) {
		return false
	}
	encoded := strings.TrimPrefix(name, activeAliasPrefix)
	digest, err := hex.DecodeString(encoded)
	return err == nil && len(digest) == sha256.Size && hex.EncodeToString(digest) == encoded
}

// ActiveAlias identifies one source declaration across evidence revisions.
func ActiveAlias(binding SourceBinding) (string, error) {
	if !validBinding(binding) {
		return "", errors.New("closure ledger: invalid active binding")
	}
	digest := sha256.Sum256([]byte(bindingDeclarationKey(binding)))
	return activeAliasPrefix + hex.EncodeToString(digest[:]), nil
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
		if bindingDeclarationKey(candidate) == bindingDeclarationKey(binding) {
			matches++
			active = candidate
		}
	}
	if matches != 1 {
		return Document{}, false, fmt.Errorf("closure ledger: active alias has %d declaration bindings", matches)
	}
	if active != binding || !bytes.Equal(document.Value, canonical) {
		return Document{}, false, errors.New("closure ledger: active evidence is stale")
	}
	return document, true, nil
}

func bindingDeclarationKey(binding SourceBinding) string {
	return string(binding.Kind) + "\x00" + binding.Package + "\x00" + binding.File + "\x00" +
		binding.Scope + "\x00" + strconv.Itoa(binding.Line) + "\x00" + binding.Name
}
