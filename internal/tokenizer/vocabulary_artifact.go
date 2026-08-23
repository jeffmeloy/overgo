package tokenizer

import (
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// VocabularyArtifactVersion is the immutable semantic vocabulary version.
	VocabularyArtifactVersion = artifact.InitialDocumentVersion
	// VocabularyArtifactMediaType identifies semantic vocabulary documents.
	VocabularyArtifactMediaType = "application/vnd.overgo.vocabulary-artifact+json"
	// VocabularyArtifactSchema identifies the semantic vocabulary wire schema.
	VocabularyArtifactSchema = "overgo/vocabulary-artifact/v1"
	// VocabularyMappingMediaType identifies exact vocabulary row mappings.
	VocabularyMappingMediaType = "application/vnd.overgo.vocabulary-mapping+json"
	// VocabularyMappingSchema identifies the vocabulary mapping wire schema.
	VocabularyMappingSchema = "overgo/vocabulary-mapping/v1"
)

// VocabularyRole identifies a tokenizer-declared semantic token role.
type VocabularyRole string

const (
	// VocabularyRoleBOS marks a beginning-of-sequence token.
	VocabularyRoleBOS VocabularyRole = "bos"
	// VocabularyRoleEOS marks an end-of-sequence token.
	VocabularyRoleEOS VocabularyRole = "eos"
	// VocabularyRoleEOT marks an end-of-turn token.
	VocabularyRoleEOT VocabularyRole = "eot"
	// VocabularyRoleEOM marks an end-of-message token.
	VocabularyRoleEOM VocabularyRole = "eom"
	// VocabularyRoleUNK marks the unknown token.
	VocabularyRoleUNK VocabularyRole = "unknown"
	// VocabularyRoleSEP marks a sequence separator token.
	VocabularyRoleSEP VocabularyRole = "separator"
	// VocabularyRolePAD marks a padding token.
	VocabularyRolePAD VocabularyRole = "padding"
	// VocabularyRoleMask marks a masked-input token.
	VocabularyRoleMask VocabularyRole = "mask"
	// VocabularyRoleFIMPre marks a fill-in-the-middle prefix token.
	VocabularyRoleFIMPre VocabularyRole = "fim-prefix"
	// VocabularyRoleFIMSuf marks a fill-in-the-middle suffix token.
	VocabularyRoleFIMSuf VocabularyRole = "fim-suffix"
	// VocabularyRoleFIMMid marks a fill-in-the-middle insertion token.
	VocabularyRoleFIMMid VocabularyRole = "fim-middle"
	// VocabularyRoleFIMPad marks fill-in-the-middle padding.
	VocabularyRoleFIMPad VocabularyRole = "fim-padding"
	// VocabularyRoleFIMRep marks a fill-in-the-middle repository token.
	VocabularyRoleFIMRep VocabularyRole = "fim-repository"
	// VocabularyRoleFIMSep marks a fill-in-the-middle separator token.
	VocabularyRoleFIMSep VocabularyRole = "fim-separator"
)

// VocabularySemantics contains tokenizer policy that changes token meaning.
type VocabularySemantics struct {
	Model        string `json:"model"`
	Pre          string `json:"pre,omitzero"`
	AddBOS       bool   `json:"add_bos,omitzero"`
	AddEOS       bool   `json:"add_eos,omitzero"`
	AddSEP       bool   `json:"add_separator,omitzero"`
	AddPrefix    bool   `json:"add_prefix,omitzero"`
	IgnoreMerges bool   `json:"ignore_merges,omitzero"`
	Lowercase    bool   `json:"lowercase,omitzero"`
	StripAccents bool   `json:"strip_accents,omitzero"`
}

// VocabularyToken is one ordered token and its exact semantic classification.
type VocabularyToken struct {
	Text  string           `json:"text"`
	Type  TokenType        `json:"type"`
	Roles []VocabularyRole `json:"roles,omitzero"`
}

// VocabularyArtifact is an immutable semantic view of a tokenizer artifact.
type VocabularyArtifact struct {
	Version   uint16              `json:"version"`
	Tokenizer artifact.ID         `json:"tokenizer"`
	Semantics VocabularySemantics `json:"semantics"`
	Tokens    []VocabularyToken   `json:"tokens"`
	ID        artifact.ID         `json:"-"`
}

// VocabularyMapping maps target rows to exact source-token rows.
type VocabularyMapping struct {
	Version uint16      `json:"version"`
	Source  artifact.ID `json:"source"`
	Target  artifact.ID `json:"target"`
	Rows    []TokenID   `json:"rows"`
	ID      artifact.ID `json:"-"`
}

var vocabularyArtifactCodec = artifact.JSONDocumentCodec(
	"vocabulary artifact", artifact.KindProfile,
	VocabularyArtifactMediaType, VocabularyArtifactSchema,
	canonicalizeVocabularyArtifact,
	func(value VocabularyArtifact) artifact.ID { return value.ID },
	func(value *VocabularyArtifact, id artifact.ID) { value.ID = id },
	func(value VocabularyArtifact) VocabularyArtifact {
		value.Tokens = cloneVocabularyTokens(value.Tokens)
		return value
	},
)

var vocabularyMappingCodec = artifact.JSONDocumentCodec(
	"vocabulary mapping", artifact.KindProfile,
	VocabularyMappingMediaType, VocabularyMappingSchema,
	canonicalizeVocabularyMapping,
	func(value VocabularyMapping) artifact.ID { return value.ID },
	func(value *VocabularyMapping, id artifact.ID) { value.ID = id },
	func(value VocabularyMapping) VocabularyMapping {
		value.Rows = slices.Clone(value.Rows)
		return value
	},
)

// NewVocabularyArtifact seals explicit vocabulary semantics. It is useful for
// non-GGUF importers; GGUF callers should use Vocab.Artifact.
func NewVocabularyArtifact(
	tokenizerID artifact.ID,
	semantics VocabularySemantics,
	tokens []VocabularyToken,
) (VocabularyArtifact, error) {
	return vocabularyArtifactCodec.New(VocabularyArtifact{
		Version: VocabularyArtifactVersion, Tokenizer: tokenizerID,
		Semantics: semantics, Tokens: cloneVocabularyTokens(tokens),
	})
}

// Artifact compiles the vocabulary policy loaded from an exact tokenizer.
func (v *Vocab) Artifact(tokenizerID artifact.ID) (VocabularyArtifact, error) {
	if v == nil {
		return VocabularyArtifact{}, errors.New("tokenizer: vocabulary artifact source is absent")
	}
	tokens := make([]VocabularyToken, len(v.Tokens))
	for index, token := range v.Tokens {
		tokens[index] = VocabularyToken{
			Text: token.Text, Type: token.Type,
			Roles: v.roles(TokenID(index)),
		}
	}
	return NewVocabularyArtifact(tokenizerID, VocabularySemantics{
		Model: v.Model, Pre: v.Pre, AddBOS: v.AddBOS, AddEOS: v.AddEOS,
		AddSEP: v.AddSEP, AddPrefix: v.AddPrefix, IgnoreMerges: v.IgnoreMerges,
		Lowercase: v.Lowercase, StripAccents: v.StripAccents,
	}, tokens)
}

// ValidateIdentity verifies the complete semantic vocabulary identity.
func (value VocabularyArtifact) ValidateIdentity() error {
	return vocabularyArtifactCodec.ValidateIdentity(value)
}

// Content returns the immutable vocabulary document.
func (value VocabularyArtifact) Content() (artifact.Content, error) {
	return vocabularyArtifactCodec.Content(value)
}

// Lineage binds semantic vocabulary facts to the tokenizer artifact.
func (value VocabularyArtifact) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Tokenizer)
}

// Batch prepares atomic publication of semantic vocabulary facts.
func (value VocabularyArtifact) Batch(key string) (artifact.Batch, error) {
	return vocabularyArtifactCodec.Batch(key, value, value.Lineage(), nil)
}

// ExactVocabularyMapping compiles target-row to source-row correspondence.
// Missing tokens and any semantic mismatch refuse; no embedding is invented.
func ExactVocabularyMapping(source, target VocabularyArtifact) (VocabularyMapping, error) {
	if err := source.ValidateIdentity(); err != nil {
		return VocabularyMapping{}, err
	}
	if err := target.ValidateIdentity(); err != nil {
		return VocabularyMapping{}, err
	}
	if source.Semantics != target.Semantics {
		return VocabularyMapping{}, errors.New("tokenizer: vocabulary semantics differ")
	}
	byText := make(map[string]int, len(source.Tokens))
	for index, token := range source.Tokens {
		byText[token.Text] = index
	}
	rows := make([]TokenID, len(target.Tokens))
	for index, token := range target.Tokens {
		sourceIndex, found := byText[token.Text]
		if !found {
			return VocabularyMapping{}, fmt.Errorf("tokenizer: target token %q has no exact source", token.Text)
		}
		candidate := source.Tokens[sourceIndex]
		if candidate.Type != token.Type || !slices.Equal(candidate.Roles, token.Roles) {
			return VocabularyMapping{}, fmt.Errorf("tokenizer: target token %q semantics differ", token.Text)
		}
		rows[index] = TokenID(sourceIndex)
	}
	return vocabularyMappingCodec.New(VocabularyMapping{
		Version: VocabularyArtifactVersion, Source: source.ID, Target: target.ID, Rows: rows,
	})
}

// ValidateIdentity verifies mapping bounds and content identity.
func (value VocabularyMapping) ValidateIdentity() error {
	return vocabularyMappingCodec.ValidateIdentity(value)
}

// Content returns the immutable exact row mapping.
func (value VocabularyMapping) Content() (artifact.Content, error) {
	return vocabularyMappingCodec.Content(value)
}

// Lineage binds the mapping to both semantic vocabularies.
func (value VocabularyMapping) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Source, value.Target)
}

// Batch prepares atomic publication of an exact vocabulary mapping.
func (value VocabularyMapping) Batch(key string) (artifact.Batch, error) {
	return vocabularyMappingCodec.Batch(key, value, value.Lineage(), nil)
}

func (v *Vocab) roles(id TokenID) []VocabularyRole {
	bindings := [...]struct {
		role VocabularyRole
		id   TokenID
	}{
		{VocabularyRoleBOS, v.BOS}, {VocabularyRoleEOS, v.EOS},
		{VocabularyRoleEOT, v.EOT}, {VocabularyRoleEOM, v.EOM},
		{VocabularyRoleUNK, v.UNK}, {VocabularyRoleSEP, v.SEP},
		{VocabularyRolePAD, v.PAD}, {VocabularyRoleMask, v.Mask},
		{VocabularyRoleFIMPre, v.FIMPre}, {VocabularyRoleFIMSuf, v.FIMSuf},
		{VocabularyRoleFIMMid, v.FIMMid}, {VocabularyRoleFIMPad, v.FIMPad},
		{VocabularyRoleFIMRep, v.FIMRep}, {VocabularyRoleFIMSep, v.FIMSep},
	}
	var roles []VocabularyRole
	for _, binding := range bindings {
		if binding.id == id {
			roles = append(roles, binding.role)
		}
	}
	return roles
}

func canonicalizeVocabularyArtifact(value *VocabularyArtifact) error {
	if value == nil || value.Version != VocabularyArtifactVersion ||
		value.Tokenizer.Kind() != artifact.KindTokenizer ||
		strings.TrimSpace(value.Semantics.Model) == "" || !checked.Nonempty(value.Tokens) {
		return errors.New("tokenizer: invalid vocabulary artifact")
	}
	seen := make(map[string]bool, len(value.Tokens))
	for index := range value.Tokens {
		token := &value.Tokens[index]
		if token.Text == "" || token.Type < TokenUndefined || token.Type > TokenByte || seen[token.Text] {
			return errors.New("tokenizer: vocabulary token is invalid or duplicated")
		}
		seen[token.Text] = true
		sort.Slice(token.Roles, func(left, right int) bool { return token.Roles[left] < token.Roles[right] })
		for roleIndex, role := range token.Roles {
			if !validVocabularyRole(role) || roleIndex > 0 && role == token.Roles[roleIndex-1] {
				return errors.New("tokenizer: vocabulary token role is invalid or duplicated")
			}
		}
	}
	return nil
}

func canonicalizeVocabularyMapping(value *VocabularyMapping) error {
	if value == nil || value.Version != VocabularyArtifactVersion ||
		value.Source.Kind() != artifact.KindProfile || value.Target.Kind() != artifact.KindProfile ||
		!checked.Nonempty(value.Rows) {
		return errors.New("tokenizer: invalid vocabulary mapping")
	}
	for _, row := range value.Rows {
		if row < 0 {
			return errors.New("tokenizer: vocabulary mapping contains a negative row")
		}
	}
	return nil
}

func validVocabularyRole(role VocabularyRole) bool {
	switch role {
	case VocabularyRoleBOS, VocabularyRoleEOS, VocabularyRoleEOT, VocabularyRoleEOM,
		VocabularyRoleUNK, VocabularyRoleSEP, VocabularyRolePAD, VocabularyRoleMask,
		VocabularyRoleFIMPre, VocabularyRoleFIMSuf, VocabularyRoleFIMMid,
		VocabularyRoleFIMPad, VocabularyRoleFIMRep, VocabularyRoleFIMSep:
		return true
	default:
		return false
	}
}

func cloneVocabularyTokens(tokens []VocabularyToken) []VocabularyToken {
	result := make([]VocabularyToken, len(tokens))
	for index, token := range tokens {
		result[index] = token
		result[index].Roles = slices.Clone(token.Roles)
	}
	return result
}
