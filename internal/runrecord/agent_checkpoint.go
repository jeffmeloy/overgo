package runrecord

import (
	"errors"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	AgentMutationCheckpointMediaType = "application/vnd.overgo.agent-mutation-checkpoint+json"
	AgentMutationCheckpointSchema    = "overgo/agent-mutation-checkpoint/v1"
)

type AgentCheckpointEntry struct {
	Target            string       `json:"target"`
	Mode              string       `json:"mode"`
	Encoding          string       `json:"encoding"`
	Preimage          *artifact.ID `json:"preimage,omitempty"`
	Absent            bool         `json:"absent,omitempty"`
	Empty             bool         `json:"empty,omitempty"`
	ExpectedPostimage *artifact.ID `json:"expected_postimage,omitempty"`
	ExpectedAbsent    bool         `json:"expected_absent,omitempty"`
	Gap               string       `json:"gap,omitempty"`
}

// AgentMutationCheckpoint is a manifest over content-addressed preimages in
// the common object store. It contains no inline file payloads.
type AgentMutationCheckpoint struct {
	Version       uint16                 `json:"version"`
	ID            artifact.ID            `json:"-"`
	Operation     artifact.ID            `json:"operation"`
	MutationEpoch uint64                 `json:"mutation_epoch"`
	Entries       []AgentCheckpointEntry `json:"entries"`
	Captured      uint32                 `json:"captured"`
	Gaps          uint32                 `json:"gaps"`
}

var agentMutationCheckpointCodec = artifact.JSONDocumentCodec(
	"agent mutation checkpoint", artifact.KindCheckpoint, AgentMutationCheckpointMediaType, AgentMutationCheckpointSchema,
	canonicalizeAgentMutationCheckpoint,
	func(v AgentMutationCheckpoint) artifact.ID { return v.ID }, func(v *AgentMutationCheckpoint, id artifact.ID) { v.ID = id },
	func(v AgentMutationCheckpoint) AgentMutationCheckpoint {
		v.Entries = slices.Clone(v.Entries)
		for index := range v.Entries {
			if v.Entries[index].Preimage != nil {
				id := *v.Entries[index].Preimage
				v.Entries[index].Preimage = &id
			}
			if v.Entries[index].ExpectedPostimage != nil {
				id := *v.Entries[index].ExpectedPostimage
				v.Entries[index].ExpectedPostimage = &id
			}
		}
		return v
	},
)

func NewAgentMutationCheckpoint(value AgentMutationCheckpoint) (AgentMutationCheckpoint, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return agentMutationCheckpointCodec.New(value)
}
func (v AgentMutationCheckpoint) Content() (artifact.Content, error) {
	return agentMutationCheckpointCodec.Content(v)
}
func (v AgentMutationCheckpoint) ValidateIdentity() error {
	return agentMutationCheckpointCodec.ValidateIdentity(v)
}
func (v AgentMutationCheckpoint) Lineage() []artifact.Lineage {
	parents := []artifact.ID{v.Operation}
	for _, entry := range v.Entries {
		if entry.Preimage != nil {
			parents = append(parents, *entry.Preimage)
		}
		if entry.ExpectedPostimage != nil {
			parents = append(parents, *entry.ExpectedPostimage)
		}
	}
	return artifact.DependencyLineage(v.ID, parents...)
}

func canonicalizeAgentMutationCheckpoint(v *AgentMutationCheckpoint) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || v.Operation.Kind() != artifact.KindEvidence ||
		len(v.Entries) == 0 || len(v.Entries) > math.MaxUint16 {
		return errors.New("run record: invalid agent mutation checkpoint")
	}
	var captured, gaps uint32
	for _, entry := range v.Entries {
		sources := 0
		if entry.Preimage != nil {
			sources++
		}
		if entry.Absent {
			sources++
		}
		if entry.Empty {
			sources++
		}
		if entry.Gap != "" {
			sources++
		}
		if !checkpointText(entry.Target) || !checkpointText(entry.Mode) || !checkpointText(entry.Encoding) || sources != int(artifact.InitialDocumentVersion) || entry.ExpectedAbsent && entry.ExpectedPostimage != nil {
			return errors.New("run record: invalid agent checkpoint entry")
		}
		if entry.Preimage != nil {
			if entry.Preimage.Kind() != artifact.KindFile {
				return errors.New("run record: checkpoint preimage is not file content")
			}
			captured++
		} else if entry.Absent || entry.Empty {
			captured++
		} else if !checkpointText(entry.Gap) {
			return errors.New("run record: invalid checkpoint capture gap")
		} else {
			gaps++
		}
		if entry.ExpectedPostimage != nil && entry.ExpectedPostimage.Kind() != artifact.KindFile {
			return errors.New("run record: checkpoint postimage is not file content")
		}
	}
	sort.Slice(v.Entries, func(i, j int) bool { return v.Entries[i].Target < v.Entries[j].Target })
	for index := range v.Entries {
		if index != 0 && v.Entries[index-1].Target == v.Entries[index].Target {
			return errors.New("run record: duplicate checkpoint target")
		}
	}
	v.Captured, v.Gaps = captured, gaps
	return nil
}
func checkpointText(value string) bool {
	return value != "" && textcheck.Bounded(value, len(value), "\x00\r\n")
}
