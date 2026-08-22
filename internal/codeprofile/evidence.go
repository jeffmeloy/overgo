package codeprofile

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	EvidenceVersion   uint16 = 4
	EvidenceMediaType        = "application/vnd.overgo.code-profile+json"
	EvidenceSchema           = "overgo/code-profile/v4"
)

type Evidence struct {
	Version    uint16      `json:"version"`
	CodeCommit string      `json:"code_commit"`
	GateResult artifact.ID `json:"gate_result"`
	Profile    Profile     `json:"profile"`
	ID         artifact.ID `json:"-"`
}

var evidenceCodec = artifact.JSONDocumentCodec(
	"code profile evidence", artifact.KindEvidence, EvidenceMediaType, EvidenceSchema,
	func(value *Evidence) error {
		if value == nil || value.Version != EvidenceVersion || !commitIdentity(value.CodeCommit) ||
			value.GateResult.Kind() != artifact.KindEvidence || validateProfile(value.Profile) != nil {
			return errors.New("code profile: invalid evidence")
		}
		return nil
	},
	func(value Evidence) artifact.ID { return value.ID },
	func(value *Evidence, id artifact.ID) { value.ID = id },
	func(value Evidence) Evidence {
		value.Profile.Functions = slices.Clone(value.Profile.Functions)
		value.Profile.Clones = slices.Clone(value.Profile.Clones)
		for index := range value.Profile.Clones {
			value.Profile.Clones[index].Functions = slices.Clone(value.Profile.Clones[index].Functions)
		}
		return value
	},
)

func NewEvidence(codeCommit string, gateResult artifact.ID, profile Profile) (Evidence, error) {
	return evidenceCodec.New(Evidence{Version: EvidenceVersion, CodeCommit: codeCommit, GateResult: gateResult, Profile: profile})
}

func (value Evidence) Content() (artifact.Content, error) { return evidenceCodec.Content(value) }
func (value Evidence) ValidateIdentity() error            { return evidenceCodec.ValidateIdentity(value) }
func (value Evidence) Lineage() []artifact.Lineage {
	return []artifact.Lineage{{Child: value.ID, Parent: value.GateResult, Relation: artifact.RelationDependsOn}}
}

func commitIdentity(value string) bool {
	if len(value) != hex.EncodedLen(sha1.Size) && len(value) != hex.EncodedLen(sha256.Size) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func validateProfile(profile Profile) error {
	for _, partition := range []Partition{profile.Runtime, profile.Automation, profile.Generated, profile.Test} {
		if partition.Files < 0 || partition.Nodes < 0 {
			return errors.New("negative profile metric")
		}
	}
	if profile.DuplicateExcessNodes < 0 || profile.ExportedDeclarations < 0 || profile.PackageImportEdges < 0 ||
		profile.Consumers.Production < 0 || profile.Consumers.TestOnly < 0 || profile.Consumers.Boundary < 0 || profile.Consumers.Zero < 0 ||
		profile.Impact.Owned < 0 || profile.Impact.Triggered < 0 || profile.Impact.Excluded < 0 || profile.Impact.Unresolved < 0 {
		return errors.New("negative profile metric")
	}
	if profile.Impact.Triggered+profile.Impact.Excluded+profile.Impact.Unresolved != profile.Impact.Owned ||
		profile.Impact.Owned > 0 && profile.Impact.Identity == "" {
		return errors.New("invalid impact selection metric")
	}
	excess := 0
	for _, function := range profile.Functions {
		if function.File == "" || function.Name == "" || function.Nodes <= 0 || function.Branches < 0 || !validAdvisoryClass(function.AdvisoryClass) {
			return errors.New("invalid function profile")
		}
	}
	for _, clone := range profile.Clones {
		if len(clone.Fingerprint) != hex.EncodedLen(sha256.Size) || clone.Nodes <= 0 || !hasFunctionPair(clone.Functions) || !validAdvisoryClass(clone.AdvisoryClass) {
			return errors.New("invalid clone profile")
		}
		if _, err := hex.DecodeString(clone.Fingerprint); err != nil {
			return errors.New("invalid clone fingerprint")
		}
		excessCopies := len(clone.Functions)
		excessCopies--
		excess += clone.Nodes * excessCopies
	}
	if excess != profile.DuplicateExcessNodes {
		return errors.New("duplicate excess does not match clones")
	}
	return nil
}

func hasFunctionPair(functions []string) bool {
	found := false
	for range functions {
		if found {
			return true
		}
		found = true
	}
	return false
}

func validAdvisoryClass(class string) bool {
	return class == "" || class == "validator" || class == "test"
}
