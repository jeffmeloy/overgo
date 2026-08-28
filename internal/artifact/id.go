package artifact

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const (
	idAlgorithm         = "sha256"
	idParts             = 3
	digestBytes         = sha256.Size
	digestHex           = digestBytes * 2
	identifyBufferBytes = 1 << 20
)

// Kind defines durable artifact class.
type Kind uint8

const (
	KindInvalid Kind = iota
	KindModel
	KindTensorSet
	KindTokenizer
	KindProjector
	KindAdapter
	KindDataset
	KindDatasetShard
	KindCheckpoint
	KindRecipe
	KindOutput
	KindFile
	KindEvidence
	KindProfile
	KindRun
	KindEvaluation
	KindTensorInventory
	KindModelDefinition
)

var kindNames = [...]string{
	KindInvalid:         "invalid",
	KindModel:           "model",
	KindTensorSet:       "tensor-set",
	KindTokenizer:       "tokenizer",
	KindProjector:       "projector",
	KindAdapter:         "adapter",
	KindDataset:         "dataset",
	KindDatasetShard:    "dataset-shard",
	KindCheckpoint:      "checkpoint",
	KindRecipe:          "recipe",
	KindOutput:          "output",
	KindFile:            "file",
	KindEvidence:        "evidence",
	KindProfile:         "profile",
	KindRun:             "run",
	KindEvaluation:      "evaluation",
	KindTensorInventory: "tensor-inventory",
	KindModelDefinition: "model-definition",
}

func (k Kind) String() string {
	if int(k) >= len(kindNames) {
		return kindNames[KindInvalid]
	}
	return kindNames[k]
}

func ParseKind(value string) (Kind, error) {
	for kind, name := range kindNames {
		if kind > 0 && value == name {
			return Kind(kind), nil
		}
	}
	return KindInvalid, fmt.Errorf("artifact: unknown kind %q", value)
}

// ID defines kind-qualified SHA-256 content identity.
type ID struct {
	kind   Kind
	digest [digestBytes]byte
}

// CompareID orders by kind, then digest.
func CompareID(left, right ID) int {
	if order := cmp.Compare(left.kind, right.kind); order != 0 {
		return order
	}
	return bytes.Compare(left.digest[:], right.digest[:])
}

func NewID(kind Kind, digest [digestBytes]byte) (ID, error) {
	if kind == KindInvalid || int(kind) >= len(kindNames) {
		return ID{}, errors.New("artifact: invalid kind")
	}
	return ID{kind: kind, digest: digest}, nil
}

func IdentifyBytes(kind Kind, data []byte) (ID, error) {
	return NewID(kind, sha256.Sum256(data))
}

// JSONID hashes canonical encoding/json output.
func JSONID(kind Kind, value any) (ID, error) {
	data, err := json.Marshal(value)
	if err == nil {
		return IdentifyBytes(kind, data)
	}
	return ID{}, err
}

func Identify(kind Kind, reader io.Reader) (ID, uint64, error) {
	if reader == nil {
		return ID{}, 0, errors.New("artifact: nil content reader")
	}
	if kind == KindInvalid || int(kind) >= len(kindNames) {
		return ID{}, 0, errors.New("artifact: invalid kind")
	}
	hasher := sha256.New()
	count, err := io.CopyBuffer(hasher, reader, make([]byte, identifyBufferBytes))
	if err != nil {
		return ID{}, 0, fmt.Errorf("artifact: hash content: %w", err)
	}
	var digest [digestBytes]byte
	copy(digest[:], hasher.Sum(nil))
	return ID{kind: kind, digest: digest}, uint64(count), nil
}

func ParseID(value string) (ID, error) {
	parts := strings.Split(value, ":")
	if len(parts) != idParts || parts[1] != idAlgorithm || len(parts[2]) != digestHex {
		return ID{}, fmt.Errorf("artifact: invalid ID %q", value)
	}
	kind, err := ParseKind(parts[0])
	if err != nil {
		return ID{}, err
	}
	decoded, err := hex.DecodeString(parts[2])
	if err != nil {
		return ID{}, fmt.Errorf("artifact: invalid ID digest: %w", err)
	}
	var digest [digestBytes]byte
	copy(digest[:], decoded)
	return ID{kind: kind, digest: digest}, nil
}

func (id ID) Valid() bool {
	return id.kind != KindInvalid && int(id.kind) < len(kindNames)
}

func (id ID) Kind() Kind {
	return id.kind
}

// DigestHex returns the filesystem-safe digest portion of a valid identity.
func (id ID) DigestHex() string {
	if !id.Valid() {
		return ""
	}
	return hex.EncodeToString(id.digest[:])
}

func (id ID) String() string {
	if !id.Valid() {
		return ""
	}
	return id.kind.String() + ":" + idAlgorithm + ":" + id.DigestHex()
}

func (id ID) MarshalText() ([]byte, error) {
	if !id.Valid() {
		return nil, errors.New("artifact: invalid ID")
	}
	return []byte(id.String()), nil
}

func (id *ID) UnmarshalText(data []byte) error {
	return unmarshalParsedText(id, data, ParseID, "ID")
}

func (id ID) MarshalJSON() ([]byte, error) {
	text, err := id.MarshalText()
	if err != nil {
		return nil, err
	}
	return json.Marshal(string(text))
}

func (id *ID) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("artifact: decode ID: %w", err)
	}
	return id.UnmarshalText([]byte(value))
}
