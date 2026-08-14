package codeprofile

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	EvidenceVersion   uint16 = 1
	EvidenceMediaType        = "application/vnd.overgo.code-profile+json"
	EvidenceSchema           = "overgo/code-profile/v1"
)

type Evidence struct {
	Version    uint16      `json:"version"`
	CodeCommit string      `json:"code_commit"`
	GateResult artifact.ID `json:"gate_result"`
	Profile    Profile     `json:"profile"`
	ID         artifact.ID `json:"-"`
}

var evidenceCodec = artifact.DocumentCodec[Evidence]{
	Name: "code profile evidence",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: EvidenceMediaType, Schema: EvidenceSchema,
	},
	Decode: func(data []byte, value *Evidence) error { return strictjson.DecodeBytes(data, value) },
	Encode: func(value Evidence) ([]byte, error) {
		value.ID = artifact.ID{}
		return json.Marshal(value)
	},
	Canonicalize: func(value *Evidence) error {
		if value == nil || value.Version != EvidenceVersion || !commitIdentity(value.CodeCommit) ||
			value.GateResult.Kind() != artifact.KindEvidence || validateProfile(value.Profile) != nil {
			return errors.New("code profile: invalid evidence")
		}
		return nil
	},
	Clone: func(value Evidence) Evidence {
		value.Profile.Functions = slices.Clone(value.Profile.Functions)
		value.Profile.Clones = slices.Clone(value.Profile.Clones)
		for index := range value.Profile.Clones {
			value.Profile.Clones[index].Functions = slices.Clone(value.Profile.Clones[index].Functions)
		}
		return value
	},
	Identity:    func(value Evidence) artifact.ID { return value.ID },
	SetIdentity: func(value *Evidence, id artifact.ID) { value.ID = id },
}

func NewEvidence(codeCommit string, gateResult artifact.ID, profile Profile) (Evidence, error) {
	return evidenceCodec.New(Evidence{Version: EvidenceVersion, CodeCommit: codeCommit, GateResult: gateResult, Profile: profile})
}

func ParseEvidence(data []byte) (Evidence, error)         { return evidenceCodec.Parse(data) }
func (value Evidence) Content() (artifact.Content, error) { return evidenceCodec.Content(value) }
func (value Evidence) ValidateIdentity() error            { return evidenceCodec.ValidateIdentity(value) }
func (value Evidence) Lineage() []artifact.Lineage {
	return []artifact.Lineage{{Child: value.ID, Parent: value.GateResult, Relation: artifact.RelationDependsOn}}
}

func commitIdentity(value string) bool {
	if len(value) != 40 && len(value) != 64 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func validateProfile(profile Profile) error {
	if profile.Production.Files < 0 || profile.Production.Nodes < 0 || profile.Test.Files < 0 || profile.Test.Nodes < 0 ||
		profile.DuplicateExcessNodes < 0 || profile.ExportedDeclarations < 0 || profile.PackageImportEdges < 0 {
		return errors.New("negative profile metric")
	}
	excess := 0
	for _, function := range profile.Functions {
		if function.File == "" || function.Name == "" || function.Nodes <= 0 || function.Branches < 0 {
			return errors.New("invalid function profile")
		}
	}
	for _, clone := range profile.Clones {
		if len(clone.Fingerprint) != sha256HexLength || clone.Nodes <= 0 || len(clone.Functions) < 2 {
			return errors.New("invalid clone profile")
		}
		if _, err := hex.DecodeString(clone.Fingerprint); err != nil {
			return errors.New("invalid clone fingerprint")
		}
		excess += clone.Nodes * (len(clone.Functions) - 1)
	}
	if excess != profile.DuplicateExcessNodes {
		return errors.New("duplicate excess does not match clones")
	}
	return nil
}

const sha256HexLength = 64
