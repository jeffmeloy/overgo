package tokenizer

import (
	"errors"
	"slices"
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

var vocabularyArtifactCodec = artifact.JSONDocumentCodec(
	"vocabulary artifact", artifact.KindProfile,
	VocabularyArtifactMediaType, VocabularyArtifactSchema,
	canonicalizeVocabularyArtifact,
	func(value VocabularyArtifact) artifact.ID { return value.ID },
	func(value *VocabularyArtifact, id artifact.ID) { value.ID = id },
).WithClone(func(value VocabularyArtifact) VocabularyArtifact {
	value.Tokens = cloneVocabularyTokens(value.Tokens)
	return value
})

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
	if id != v.EOS && v.eog[id].declaredEOS {
		roles = append(roles, VocabularyRoleEOS)
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
		slices.Sort(token.Roles)
		for roleIndex, role := range token.Roles {
			if !validVocabularyRole(role) || roleIndex > 0 && role == token.Roles[roleIndex-1] {
				return errors.New("tokenizer: vocabulary token role is invalid or duplicated")
			}
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
