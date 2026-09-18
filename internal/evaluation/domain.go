package evaluation

import (
	"slices"
	"strings"
)

// DomainText is the domain every lm_eval-convention suite carries.
const DomainText = "text"

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
