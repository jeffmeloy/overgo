package repoanalysis

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

type modernGoExceptionPolicy struct {
	reason     string
	retirement string
}

var modernGoExceptionPolicies = map[string]modernGoExceptionPolicy{
	"sync_waitgroup_go": {
		reason:     "The owner relies on explicit goroutine launch, accounting, recovery, or ordering semantics that WaitGroup.Go does not preserve.",
		retirement: "Retire when the owning concurrency protocol no longer requires explicit Add, Done, recovery, or launch-order control.",
	},
	"json_omitzero": {
		reason:     "The field's established wire contract can distinguish omitempty from omitzero through composite emptiness, a named type, or custom marshaling.",
		retirement: "Retire after byte-level compatibility evidence proves omitzero preserves every supported payload for this owner.",
	},
	"strings_split_seq": {
		reason:     "The owner materializes, indexes, counts, or reuses the complete split result, so an iterator is not an equivalent replacement.",
		retirement: "Retire when the consumer becomes a single-pass range with no retained split slice or indexing contract.",
	},
	"maps_keys_values_iter": {
		reason:     "The owner requires an explicitly materialized key or value slice for capacity, mutation, ordering, or downstream API ownership.",
		retirement: "Retire when the downstream contract accepts an iterator directly and no concrete slice identity or mutation remains.",
	},
	"slices_sorted": {
		reason:     "The owner retains explicit materialization or in-place ordering semantics that slices.Sorted would hide or change.",
		retirement: "Retire when ordering can consume an iterator directly without preserving the current buffer identity or stable staging.",
	},
	"time_tick_gc": {
		reason:     "The ticker lifecycle requires Stop, Reset, or explicit ownership beyond the garbage-collectable channel returned by time.Tick.",
		retirement: "Retire when the ticker is channel-only for its complete lifetime and no explicit lifecycle operation remains.",
	},
	"cmp_or": {
		reason:     "The fallback is intentionally lazy, side-effecting, non-comparable, or coupled to an assignment shape that cmp.Or cannot preserve exactly.",
		retirement: "Retire when the fallback becomes eagerly safe and comparable with a single assignment-equivalent destination.",
	},
	"slices_sort_func": {
		reason:     "The comparator depends on indexes, stability, side effects, or strict-less behavior without a proven three-way comparison equivalent.",
		retirement: "Retire when the owner exposes a pure element comparator with a proven three-way ordering and equivalent stability policy.",
	},
	"sync_once_func": {
		reason:     "The once body participates in reset, error, reentrancy, or lifecycle semantics not represented by sync.OnceFunc.",
		retirement: "Retire when the once operation is an immutable no-result closure with standard panic replay and no reset contract.",
	},
	"sync_once_value": {
		reason:     "The cached value has error, reset, multi-result, address, or retry semantics not represented by sync.OnceValue.",
		retirement: "Retire when the cache becomes an immutable single-value computation with standard panic replay and no retry contract.",
	},
	"context_timeout_deadline_cause": {
		reason:     "The owner preserves the established cancellation error identity or CancelFunc type and has no stable cause to publish.",
		retirement: "Retire when the API can expose a stable cancellation cause without changing its cancel function or observed error contract.",
	},
	"context_cancel_cause": {
		reason:     "The owner requires the existing CancelFunc surface or nil-cause cancellation identity across an API or lifecycle boundary.",
		retirement: "Retire when the complete caller chain accepts CancelCauseFunc and tests pin the replacement cause identity.",
	},
}

type modernGoExceptionGroup struct {
	guideline string
	path      string
	symbol    string
	owner     string
	count     int
}

// CloseModernGoExceptions converts every retained candidate into an exact,
// package-owned exception under the reviewed policy for its guideline.
func CloseModernGoExceptions(
	baseline ModernGoBaseline,
	census ModernGoCensus,
	expires string,
	today time.Time,
) (ModernGoBaseline, error) {
	if err := AdmitModernGoRatchet(baseline, census, time.Time{}); err != nil {
		return ModernGoBaseline{}, err
	}
	expiry, err := time.Parse(time.DateOnly, expires)
	if err != nil {
		return ModernGoBaseline{}, fmt.Errorf("modern-Go exception expiry %q: %w", expires, err)
	}
	if !today.IsZero() && !expiry.After(today) {
		return ModernGoBaseline{}, fmt.Errorf("modern-Go exception expiry %q is not after %s", expires, today.Format(time.DateOnly))
	}
	groups := map[string]*modernGoExceptionGroup{}
	guidelineOrder := map[string]int{}
	for index, finding := range census.Findings {
		guidelineOrder[finding.ID] = index
		if len(finding.Candidates) != 0 {
			if _, ok := modernGoExceptionPolicies[finding.ID]; !ok {
				return ModernGoBaseline{}, fmt.Errorf("modern-Go retained guideline %s has no reviewed exception policy", finding.ID)
			}
		}
		for _, site := range finding.Candidates {
			key := finding.ID + "\x00" + site.Path + "\x00" + site.Symbol
			group := groups[key]
			if group == nil {
				group = &modernGoExceptionGroup{
					guideline: finding.ID,
					path:      site.Path,
					symbol:    site.Symbol,
					owner:     site.Package,
				}
				groups[key] = group
			}
			if group.owner != site.Package {
				return ModernGoBaseline{}, fmt.Errorf("modern-Go exception group %s crosses owners %s and %s", key, group.owner, site.Package)
			}
			group.count++
		}
	}
	measured, err := BuildModernGoBaseline(census)
	if err != nil {
		return ModernGoBaseline{}, err
	}
	measured.Doc = baseline.Doc
	for _, group := range groups {
		packagePath := filepath.ToSlash(filepath.Dir(group.path))
		if packagePath == "." || packagePath == "" {
			return ModernGoBaseline{}, fmt.Errorf("modern-Go exception %s has no package directory", group.path)
		}
		policy := modernGoExceptionPolicies[group.guideline]
		measured.Exceptions = append(measured.Exceptions, ModernGoException{
			Guideline:        group.guideline,
			Path:             group.path,
			Symbol:           group.symbol,
			Owner:            group.owner,
			Reason:           policy.reason,
			Oracle:           "go test ./" + packagePath + " -count=1",
			Retirement:       policy.retirement,
			Expires:          expires,
			CandidateCeiling: group.count,
		})
	}
	slices.SortFunc(measured.Exceptions, func(left, right ModernGoException) int {
		if order := guidelineOrder[left.Guideline] - guidelineOrder[right.Guideline]; order != 0 {
			return order
		}
		if order := strings.Compare(left.Path, right.Path); order != 0 {
			return order
		}
		return strings.Compare(left.Symbol, right.Symbol)
	})
	measured.ExceptionSHA256, err = ModernGoExceptionIdentity(measured.Exceptions)
	if err != nil {
		return ModernGoBaseline{}, err
	}
	for index := range measured.Guidelines {
		excepted, countErr := modernGoExceptionCount(measured.Exceptions, census.Findings[index])
		if countErr != nil {
			return ModernGoBaseline{}, countErr
		}
		if excepted != len(census.Findings[index].Candidates) {
			return ModernGoBaseline{}, fmt.Errorf("modern-Go exception closure incomplete for %s: %d of %d", census.Findings[index].ID, excepted, len(census.Findings[index].Candidates))
		}
		measured.Guidelines[index].CandidateCeiling = len(census.Findings[index].Candidates) - excepted
	}
	if err := validateModernGoBaseline(measured, today); err != nil {
		return ModernGoBaseline{}, err
	}
	return measured, nil
}
