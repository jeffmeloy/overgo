package evaluation

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	evalDomainsMediaType   = "application/vnd.overgo.evaluation-domains+json"
	evalDomainsSchema      = "overgo/evaluation-domains/v1"
	evalDomainsAliasPrefix = "evaluation/domains/"

	// DomainText is the domain every lm_eval-convention suite carries:
	// natural-language benchmarks bind to natural-language models.
	DomainText = "text"
)

// EvalDomainDeclaration binds one model to the evaluation domains its
// scores are meaningful in. A DNA model with its own tokenizer scores
// near zero on English multiple choice -- the number is true and
// useless -- so the declaration routes suites instead of letting every
// model meet every benchmark.
type EvalDomainDeclaration struct {
	ID      artifact.ID `json:"-"`
	Version uint16      `json:"version"`
	Model   artifact.ID `json:"model"`
	Domains []string    `json:"domains"`
}

var evalDomainCodec = artifact.JSONDocumentCodec(
	"evaluation domains", artifact.KindEvidence, evalDomainsMediaType, evalDomainsSchema,
	canonicalizeEvalDomains, func(value EvalDomainDeclaration) artifact.ID { return value.ID },
	func(value *EvalDomainDeclaration, id artifact.ID) { value.ID = id },
	func(value EvalDomainDeclaration) EvalDomainDeclaration {
		value.Domains = slices.Clone(value.Domains)
		return value
	},
)

func canonicalizeEvalDomains(value *EvalDomainDeclaration) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Model.Kind() != artifact.KindModel || len(value.Domains) == 0 {
		return errors.New("evaluation: invalid domain declaration")
	}
	value.Domains = slices.Clone(value.Domains)
	slices.Sort(value.Domains)
	previous := ""
	for _, domain := range value.Domains {
		if strings.TrimSpace(domain) != domain || domain == "" || domain == previous {
			return errors.New("evaluation: invalid domain")
		}
		previous = domain
	}
	return nil
}

// DeclareEvalDomains commits one model's domain declaration and binds
// its alias, replacing any previous declaration.
func DeclareEvalDomains(ctx context.Context, repository artifact.Repository, model artifact.ID, domains []string) (artifact.ID, error) {
	declaration, err := evalDomainCodec.New(EvalDomainDeclaration{
		Version: artifact.InitialDocumentVersion, Model: model, Domains: domains,
	})
	if err != nil {
		return artifact.ID{}, err
	}
	alias := artifact.AliasBinding{Name: evalDomainsAliasPrefix + model.String(), Target: declaration.ID}
	current, bound, err := artifact.ResolveAlias(ctx, repository, alias.Name)
	if err != nil {
		return artifact.ID{}, err
	}
	if bound {
		if current == declaration.ID {
			return declaration.ID, nil
		}
		alias.Previous = &current
	}
	batch, err := evalDomainCodec.Batch(
		"evaluation/domains/"+declaration.ID.String(), declaration, nil, []artifact.AliasBinding{alias},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return artifact.ID{}, err
	}
	return declaration.ID, nil
}

// EvalDomains reads one model's declared domains; absent means
// undeclared, and an undeclared model keeps full suite coverage so
// nothing silently stops being evaluated.
func EvalDomains(ctx context.Context, reader artifact.Reader, model artifact.ID) ([]string, bool, error) {
	id, bound, err := artifact.ResolveAlias(ctx, reader, evalDomainsAliasPrefix+model.String())
	if err != nil || !bound {
		return nil, false, err
	}
	declaration, found, err := evalDomainCodec.Read(ctx, reader, id)
	if err != nil || !found {
		return nil, false, err
	}
	return declaration.Domains, true, nil
}

// SuiteDomain names the domain a derived suite belongs to; every
// lm_eval family is text today, and new families declare theirs in the
// cache table.
func SuiteDomain(source string) string {
	suffix := strings.TrimPrefix(source, "store/")
	for _, family := range hfCacheFamilies {
		if family.family == suffix {
			return family.domain
		}
	}
	return DomainText
}

// FilterSuitesForDomains keeps the suites whose domain the model
// declares. An undeclared model (declared=false) keeps everything.
func FilterSuitesForDomains(suites []CompiledSuite, domains []string, declared bool) []CompiledSuite {
	if !declared {
		return suites
	}
	kept := make([]CompiledSuite, 0, len(suites))
	for _, suite := range suites {
		if slices.Contains(domains, SuiteDomain(suite.Descriptor().Source)) {
			kept = append(kept, suite)
		}
	}
	return kept
}
