package evaluation

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

const (
	promptTemplateMediaType   = "application/vnd.overgo.prompt-template+json"
	promptTemplateSchema      = "overgo/prompt-template/v1"
	promptTemplateAliasPrefix = "evaluation/prompt-templates/"
)

// PromptTemplate is one model's input-shaping declaration in the
// store, sourced from the model's own metadata: a chat template where
// the GGUF or tokenizer config declares one, and the scoring prefix a
// hybrid DNA tokenizer's begin tag defines for pure-sequence input.
// The store is the authority every eval consumer reads.
type PromptTemplate struct {
	ID            artifact.ID `json:"-"`
	Version       uint16      `json:"version"`
	Model         artifact.ID `json:"model"`
	Source        string      `json:"source"`
	ScoringPrefix string      `json:"scoring_prefix,omitempty"`
	ChatTemplate  string      `json:"chat_template,omitempty"`
}

var promptTemplateCodec = artifact.JSONDocumentCodec(
	"prompt template", artifact.KindEvidence, promptTemplateMediaType, promptTemplateSchema,
	func(value *PromptTemplate) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion ||
			value.Model.Kind() != artifact.KindModel || value.Source == "" {
			return errors.New("evaluation: invalid prompt template")
		}
		return nil
	},
	func(value PromptTemplate) artifact.ID { return value.ID },
	func(value *PromptTemplate, id artifact.ID) { value.ID = id },
	func(value PromptTemplate) PromptTemplate { return value },
)

// PublishPromptTemplate commits one model's template and binds its
// alias, replacing any previous declaration.
func PublishPromptTemplate(
	ctx context.Context,
	repository artifact.Repository,
	template PromptTemplate,
) (PromptTemplate, error) {
	template.Version = artifact.InitialDocumentVersion
	published, err := promptTemplateCodec.New(template)
	if err != nil {
		return PromptTemplate{}, err
	}
	alias := artifact.AliasBinding{Name: promptTemplateAliasPrefix + template.Model.String(), Target: published.ID}
	current, bound, err := artifact.ResolveAlias(ctx, repository, alias.Name)
	if err != nil {
		return PromptTemplate{}, err
	}
	if bound {
		if current == published.ID {
			return published, nil
		}
		alias.Previous = &current
	}
	batch, err := promptTemplateCodec.Batch(
		"evaluation/prompt-template/"+published.ID.String(), published, nil, []artifact.AliasBinding{alias},
	)
	if err != nil {
		return PromptTemplate{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return PromptTemplate{}, err
	}
	return published, nil
}

// LoadPromptTemplate reads one model's declared template; absent means
// no declaration stands yet.
func LoadPromptTemplate(
	ctx context.Context,
	reader artifact.Reader,
	model artifact.ID,
) (PromptTemplate, bool, error) {
	id, bound, err := artifact.ResolveAlias(ctx, reader, promptTemplateAliasPrefix+model.String())
	if err != nil || !bound {
		return PromptTemplate{}, false, err
	}
	return promptTemplateCodec.Read(ctx, reader, id)
}

// scoringPrefixer is the optional runtime surface a model-declared
// scoring prefix arrives through: the runner derives it from the same
// metadata the store template records.
type scoringPrefixer interface {
	ScoringPrefix() string
}

// runtimeScoringPrefix reports the runtime's declared prefix, empty
// for runtimes without one.
func runtimeScoringPrefix(runtime any) string {
	if prefixer, declared := runtime.(scoringPrefixer); declared {
		return prefixer.ScoringPrefix()
	}
	return ""
}
