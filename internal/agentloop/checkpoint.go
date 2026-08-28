package agentloop

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"unicode/utf8"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/pathidentity"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const restoreToolName = "agent.restore"

// MutationCheckpointRuntime captures and restores files through the common
// repository while serializing writers under one workspace authority.
type MutationCheckpointRuntime struct {
	mu     sync.Mutex
	active bool
	root   string
	store  artifact.Repository
}

// NewMutationCheckpointRuntime binds the runtime to a canonical workspace root
// and the common repository that stores checkpoint manifests and preimages.
func NewMutationCheckpointRuntime(root string, store artifact.Repository) (*MutationCheckpointRuntime, error) {
	canonical, err := pathidentity.Canonical(root)
	if err != nil || store == nil {
		return nil, errors.Join(errors.New("agent loop: checkpoint authority is invalid"), err)
	}
	return &MutationCheckpointRuntime{root: canonical, store: store}, nil
}

// BeginCheckpoint retains the writer claim until SealCheckpoint or Abort.
func (runtime *MutationCheckpointRuntime) BeginCheckpoint(ctx context.Context, operation artifact.ID, effect agenttool.InvocationEffect, epoch uint64) (runrecord.AgentMutationCheckpoint, error) {
	if runtime == nil || ctx == nil || operation.Kind() != artifact.KindEvidence || effect.Class != agenttool.EffectMutation || !effect.Known || effect.OpaqueMutation {
		return runrecord.AgentMutationCheckpoint{}, errors.New("agent loop: invalid mutation checkpoint request")
	}
	runtime.mu.Lock()
	if runtime.active {
		runtime.mu.Unlock()
		return runrecord.AgentMutationCheckpoint{}, errors.New("agent loop: checkpoint writer is active")
	}
	runtime.active = true
	runtime.mu.Unlock()
	checkpoint, contents, err := runtime.capture(operation, effect, epoch)
	if err != nil {
		runtime.AbortCheckpoint()
		return runrecord.AgentMutationCheckpoint{}, err
	}
	content, err := checkpoint.Content()
	if err != nil {
		runtime.AbortCheckpoint()
		return runrecord.AgentMutationCheckpoint{}, err
	}
	contents = append(contents, content)
	batch, err := artifact.NewDocumentBatch("agent-checkpoint/"+checkpoint.ID.String(), contents, checkpoint.Lineage(), nil)
	if err == nil {
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: operation})
	}
	if err == nil {
		_, err = artifact.CommitBatch(ctx, runtime.store, batch)
	}
	if err != nil {
		runtime.AbortCheckpoint()
		return runrecord.AgentMutationCheckpoint{}, err
	}
	return checkpoint, nil
}

func (runtime *MutationCheckpointRuntime) capture(operation artifact.ID, effect agenttool.InvocationEffect, epoch uint64) (runrecord.AgentMutationCheckpoint, []artifact.Content, error) {
	entries := []runrecord.AgentCheckpointEntry{}
	contents := []artifact.Content{}
	for _, target := range effect.Targets {
		if target.Scope != agenttool.EffectScopeWorkspace {
			continue
		}
		path, err := runtime.targetPath(target.Value)
		if err != nil {
			return runrecord.AgentMutationCheckpoint{}, nil, err
		}
		entry := runrecord.AgentCheckpointEntry{Target: filepath.ToSlash(path), Encoding: "none", Mode: "absent"}
		data, readErr := os.ReadFile(path)
		switch {
		case errors.Is(readErr, os.ErrNotExist):
			entry.Absent = true
		case readErr != nil:
			entry.Gap = readErr.Error()
		default:
			info, statErr := os.Stat(path)
			if statErr != nil || !info.Mode().IsRegular() {
				entry.Gap = "target is not a regular file"
				break
			}
			entry.Mode = fmt.Sprintf("%o", info.Mode().Perm())
			if utf8.Valid(data) {
				entry.Encoding = "utf-8"
			} else {
				entry.Encoding = "binary"
			}
			if len(data) == 0 {
				entry.Empty = true
				break
			}
			id, identifyErr := artifact.IdentifyBytes(artifact.KindFile, data)
			if identifyErr != nil {
				return runrecord.AgentMutationCheckpoint{}, nil, identifyErr
			}
			entry.Preimage = artifact.IDPointer(id)
			contents = append(contents, artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data))}, Data: data})
		}
		entries = append(entries, entry)
	}
	if len(entries) == 0 {
		return runrecord.AgentMutationCheckpoint{}, nil, errors.New("agent loop: mutation has no workspace checkpoint target")
	}
	checkpoint, err := runrecord.NewAgentMutationCheckpoint(runrecord.AgentMutationCheckpoint{Operation: operation, MutationEpoch: epoch, Entries: entries})
	return checkpoint, contents, err
}

// SealCheckpoint binds the actual postimage and releases the writer claim.
func (runtime *MutationCheckpointRuntime) SealCheckpoint(ctx context.Context, checkpoint runrecord.AgentMutationCheckpoint) (runrecord.AgentMutationCheckpoint, error) {
	defer runtime.AbortCheckpoint()
	sealed := checkpoint
	sealed.ID = artifact.ID{}
	sealed.Entries = slices.Clone(checkpoint.Entries)
	for index := range sealed.Entries {
		path, err := runtime.targetPath(sealed.Entries[index].Target)
		if err != nil {
			return runrecord.AgentMutationCheckpoint{}, err
		}
		data, readErr := os.ReadFile(path)
		if errors.Is(readErr, os.ErrNotExist) {
			sealed.Entries[index].ExpectedAbsent = true
			continue
		}
		if readErr != nil {
			return runrecord.AgentMutationCheckpoint{}, readErr
		}
		id, err := artifact.IdentifyBytes(artifact.KindFile, data)
		if err != nil {
			return runrecord.AgentMutationCheckpoint{}, err
		}
		sealed.Entries[index].ExpectedPostimage = artifact.IDPointer(id)
	}
	sealed, err := runrecord.NewAgentMutationCheckpoint(sealed)
	if err != nil {
		return runrecord.AgentMutationCheckpoint{}, err
	}
	content, err := sealed.Content()
	if err != nil {
		return runrecord.AgentMutationCheckpoint{}, err
	}
	batch, err := artifact.NewDocumentBatch("agent-checkpoint/seal/"+sealed.ID.String(), []artifact.Content{content}, sealed.Lineage(), nil)
	if err == nil {
		for _, entry := range sealed.Entries {
			if entry.ExpectedPostimage != nil {
				batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: *entry.ExpectedPostimage})
			}
		}
	}
	if err == nil {
		_, err = artifact.CommitBatch(ctx, runtime.store, batch)
	}
	return sealed, err
}

// AbortCheckpoint releases the writer claim without sealing or restoring.
func (runtime *MutationCheckpointRuntime) AbortCheckpoint() {
	if runtime == nil {
		return
	}
	runtime.mu.Lock()
	runtime.active = false
	runtime.mu.Unlock()
}

// RestoreApprovalRequest creates the normal operator decision envelope.
func RestoreApprovalRequest(checkpoint runrecord.AgentMutationCheckpoint, recipeID artifact.ID) (operatoraction.ApprovalRequest, error) {
	operation, err := artifact.JSONID(artifact.KindEvidence, struct {
		Checkpoint artifact.ID `json:"checkpoint"`
		Action     string      `json:"action"`
	}{checkpoint.ID, restoreToolName})
	if err != nil {
		return operatoraction.ApprovalRequest{}, err
	}
	return operatoraction.NewApprovalRequest(operation, recipeID, operatoraction.Action{Code: restoreToolName, Summary: "restore agent mutation checkpoint", Argv: []string{checkpoint.ID.String()}}, artifact.ID{})
}

// RestoreCheckpoint verifies all postimages before changing any target.
func (runtime *MutationCheckpointRuntime) RestoreCheckpoint(ctx context.Context, checkpoint runrecord.AgentMutationCheckpoint, effect agenttool.InvocationEffect, recipeID artifact.ID) (artifact.ID, error) {
	if runtime == nil || checkpoint.Gaps != 0 || effect.Class != agenttool.EffectMutation || !effect.Known || effect.OpaqueMutation {
		return artifact.ID{}, errors.New("agent loop: checkpoint restore is not admissible")
	}
	if !runtime.effectMatchesCheckpoint(effect, checkpoint) {
		return artifact.ID{}, errors.New("agent loop: restore effect differs from checkpoint targets")
	}
	request, err := RestoreApprovalRequest(checkpoint, recipeID)
	if err != nil {
		return artifact.ID{}, err
	}
	decision, found, err := runrecord.ResolveHumanDecision(ctx, runtime.store, request.Operation)
	if err != nil || !found || decision.Answer != operatoraction.AnswerGrant || !decision.Binds(request) {
		return artifact.ID{}, errors.Join(errors.New("agent loop: exact restore approval is absent"), err)
	}
	runtime.mu.Lock()
	if runtime.active {
		runtime.mu.Unlock()
		return artifact.ID{}, errors.New("agent loop: conflicting checkpoint writer is active")
	}
	runtime.active = true
	runtime.mu.Unlock()
	defer runtime.AbortCheckpoint()
	for _, entry := range checkpoint.Entries {
		if err := runtime.verifyPostimage(entry); err != nil {
			return artifact.ID{}, err
		}
	}
	for _, entry := range checkpoint.Entries {
		if err := runtime.restoreEntry(ctx, entry); err != nil {
			return artifact.ID{}, err
		}
	}
	receipt := runrecord.StageReceipt{Recipe: decision.Recipe, Node: recipe.NodeID(restoreToolName), Operation: request.Operation, Attempt: uint32(artifact.InitialDocumentVersion), State: runrecord.StageCompleted,
		Inputs:  []runrecord.StageBinding{{Port: recipe.PortName("effect"), Artifacts: []artifact.ID{effect.ID}}},
		Outputs: []runrecord.StageBinding{{Port: recipe.PortName("checkpoint"), Artifacts: []artifact.ID{checkpoint.ID}}}}
	published, err := runrecord.PublishStageReceipt(context.WithoutCancel(ctx), runtime.store, receipt, nil,
		[]artifact.Descriptor{{ID: decision.Recipe}, {ID: request.Operation}, {ID: effect.ID}})
	return published.ID, err
}

func (runtime *MutationCheckpointRuntime) verifyPostimage(entry runrecord.AgentCheckpointEntry) error {
	path, err := runtime.targetPath(entry.Target)
	if err != nil {
		return err
	}
	data, readErr := os.ReadFile(path)
	if entry.ExpectedAbsent {
		if errors.Is(readErr, os.ErrNotExist) {
			return nil
		}
		return errors.New("agent loop: checkpoint postimage drifted")
	}
	if readErr != nil || entry.ExpectedPostimage == nil {
		return errors.New("agent loop: checkpoint postimage is unavailable")
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil || id != *entry.ExpectedPostimage {
		return errors.New("agent loop: checkpoint postimage drifted")
	}
	return nil
}

func (runtime *MutationCheckpointRuntime) restoreEntry(ctx context.Context, entry runrecord.AgentCheckpointEntry) error {
	path, err := runtime.targetPath(entry.Target)
	if err != nil {
		return err
	}
	if entry.Absent {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	data := []byte{}
	if entry.Preimage != nil {
		content, found, readErr := artifact.ReadContent(ctx, runtime.store, *entry.Preimage)
		if readErr != nil || !found {
			return errors.Join(errors.New("agent loop: checkpoint preimage is absent"), readErr)
		}
		data = content.Data
	}
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		return err
	}
	var mode os.FileMode
	if _, err := fmt.Sscanf(entry.Mode, "%o", &mode); err != nil {
		return err
	}
	if err := os.WriteFile(path, data, mode); err != nil {
		return err
	}
	return os.Chmod(path, mode)
}

func (runtime *MutationCheckpointRuntime) effectMatchesCheckpoint(effect agenttool.InvocationEffect, checkpoint runrecord.AgentMutationCheckpoint) bool {
	targets := map[string]bool{}
	for _, target := range effect.Targets {
		if target.Scope != agenttool.EffectScopeWorkspace {
			continue
		}
		path, err := runtime.targetPath(target.Value)
		if err != nil {
			return false
		}
		targets[filepath.ToSlash(path)] = true
	}
	if len(targets) != len(checkpoint.Entries) {
		return false
	}
	for _, entry := range checkpoint.Entries {
		if !targets[filepath.ToSlash(filepath.Clean(entry.Target))] {
			return false
		}
	}
	return true
}

func (runtime *MutationCheckpointRuntime) targetPath(value string) (string, error) {
	path := filepath.FromSlash(value)
	if !filepath.IsAbs(path) {
		path = filepath.Join(runtime.root, path)
	}
	contained, err := pathidentity.Contains(runtime.root, path)
	if err != nil || !contained {
		return "", errors.Join(errors.New("agent loop: checkpoint target escapes workspace"), err)
	}
	return filepath.Clean(path), nil
}
