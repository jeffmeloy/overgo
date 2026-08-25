package agenttool

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// ArgvPolicyMediaType identifies argv allowlist policy documents.
	ArgvPolicyMediaType = "application/vnd.overgo.agent-argv-policy+json"
	// ArgvPolicySchema identifies the argv allowlist contract.
	ArgvPolicySchema = "overgo/agent-argv-policy/v1"
	// ArgvPolicyAlias is the store's single active argv allowlist.
	ArgvPolicyAlias = "tool.policy.argv"
)

// ArgvPolicy is the durable allowlist of programs argv manuals may
// name. It is store authority, not a CLI convenience: publication AND
// invocation both check the committed policy, and an ABSENT policy
// refuses every argv manual -- fail closed, never open.
type ArgvPolicy struct {
	Version  uint16      `json:"version"`
	Programs []string    `json:"programs,omitempty"`
	ID       artifact.ID `json:"-"`
}

var argvPolicyCodec = artifact.JSONDocumentCodec(
	"agent argv policy", artifact.KindProfile, ArgvPolicyMediaType, ArgvPolicySchema,
	canonicalizeArgvPolicy,
	func(value ArgvPolicy) artifact.ID { return value.ID },
	func(value *ArgvPolicy, id artifact.ID) { value.ID = id },
	func(value ArgvPolicy) ArgvPolicy { value.Programs = slices.Clone(value.Programs); return value },
)

func canonicalizeArgvPolicy(value *ArgvPolicy) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion {
		return errors.New("agent tool: invalid argv policy version")
	}
	slices.Sort(value.Programs)
	value.Programs = slices.Compact(value.Programs)
	for _, program := range value.Programs {
		if strings.TrimSpace(program) != program || program == "" || strings.ContainsAny(program, `/\`) {
			return errors.New("agent tool: argv policy programs must be bare command words")
		}
	}
	return nil
}

// PublishArgvPolicy commits the allowlist and rebinds the policy alias
// with compare-and-set supersession. An empty program set is a valid
// policy: it durably refuses every argv manual.
func PublishArgvPolicy(ctx context.Context, repository artifact.Repository, programs []string) (ArgvPolicy, error) {
	if ctx == nil || repository == nil {
		return ArgvPolicy{}, errors.New("agent tool: nil policy context or repository")
	}
	policy, err := argvPolicyCodec.New(ArgvPolicy{Version: artifact.InitialDocumentVersion, Programs: slices.Clone(programs)})
	if err != nil {
		return ArgvPolicy{}, err
	}
	content, err := argvPolicyCodec.Content(policy)
	if err != nil {
		return ArgvPolicy{}, err
	}
	batch := artifact.Batch{
		Contents: []artifact.Content{content},
		Aliases:  []artifact.AliasBinding{{Name: ArgvPolicyAlias, Target: policy.ID}},
	}
	supersedes := "initial"
	if current, bound, err := repository.ResolveAlias(ctx, ArgvPolicyAlias); err != nil {
		return ArgvPolicy{}, err
	} else if bound {
		if current == policy.ID {
			batch.Aliases = nil
		} else {
			batch.Aliases[0].Previous = artifact.IDPointer(current)
			supersedes = current.String()
		}
	}
	// The batch key names the TRANSITION, not just the document: the
	// same policy content can rebind the alias from different
	// predecessors, and each such supersession is its own batch.
	batch.Key = "agent-argv-policy/" + policy.ID.String() + "/from/" + supersedes
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return ArgvPolicy{}, err
	}
	return policy, nil
}

// ResolveArgvPolicy reads the store's active argv allowlist.
func ResolveArgvPolicy(ctx context.Context, reader artifact.Reader) (ArgvPolicy, bool, error) {
	if ctx == nil || reader == nil {
		return ArgvPolicy{}, false, errors.New("agent tool: nil policy context or reader")
	}
	target, found, err := reader.ResolveAlias(ctx, ArgvPolicyAlias)
	if err != nil || !found {
		return ArgvPolicy{}, found, err
	}
	policy, err := argvPolicyCodec.Require(ctx, reader, target)
	return policy, err == nil, err
}

// CheckArgvAuthority admits one manual against the committed policy:
// non-argv manuals pass untouched, and an argv manual requires a
// committed policy naming its exact program -- an absent policy or an
// unlisted program refuses.
func CheckArgvAuthority(ctx context.Context, reader artifact.Reader, manual Manual) error {
	if manual.Transport.Kind != TransportArgv {
		return nil
	}
	policy, found, err := ResolveArgvPolicy(ctx, reader)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("agent tool: argv manual %q refused: no argv policy is committed", manual.Name)
	}
	if !slices.Contains(policy.Programs, manual.Transport.Program) {
		return fmt.Errorf("agent tool: argv manual %q names %q outside the committed policy", manual.Name, manual.Transport.Program)
	}
	return nil
}
