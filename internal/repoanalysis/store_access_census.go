package repoanalysis

import (
	"cmp"
	"path"
	"slices"
)

// The store access census computes, from syntax alone, which packages talk
// to the store themselves and how: it is what a consolidation of database
// interaction is cut and ranked from. Like the ownership census beside it,
// nothing here is a hand-kept list of files; the kinds are observable syntax
// and a package's domains are the record packages it imports.

// storeAccessKinds are the ways a package reaches the store directly.
func storeAccessKinds() []CensusFamily {
	const records, store = "overgo/internal/artifact", "overgo/internal/overgodb"
	return []CensusFamily{
		{Name: "open", Calls: []CensusCall{{store, "Open"}, {store, "OpenReadOnly"}, {store, "OpenContext"}}},
		// Building the batch is the storage mechanic itself, in domain code.
		{Name: "batch", Literals: []CensusCall{{records, "Batch"}}, Calls: []CensusCall{{records, "NewDocumentBatch"}}},
		{Name: "commit", Calls: []CensusCall{{records, "CommitBatch"}}, ScopedCalls: []string{"Commit"}, ScopedImport: records},
		{Name: "alias", Calls: []CensusCall{{records, "ResolveAlias"}}},
		{Name: "read", Calls: []CensusCall{{records, "ReadContent"}, {records, "ReadContents"}, {records, "RequireTypedContent"}}},
		{Name: "query", ScopedCalls: []string{"Query", "Visit"}, ScopedImport: store},
	}
}

// storeDomains names each domain by the record packages that define it; a
// package belongs to every domain it is or imports, and one that belongs to
// several is where isolating them will be hardest.
var storeDomains = map[string][]string{
	"models":   {"overgo/internal/recipe", "overgo/internal/modelrecipe", "overgo/internal/modelartifact"},
	"datasets": {"overgo/internal/dataset"},
	"evidence": {"overgo/internal/runrecord", "overgo/internal/evaluation"},
	"control":  {"overgo/internal/plan", "overgo/internal/worklease"},
}

// StoreAccess is one package's direct use of the store, by kind.
type StoreAccess struct {
	Package string         `json:"package"`
	Domains []string       `json:"domains,omitempty"`
	Sites   map[string]int `json:"sites"`
	Total   int            `json:"total"`
}

// StoreAccessCensus reports every production package that reaches the store
// directly, most sites first.
func StoreAccessCensus(snapshot SourceSnapshot) ([]StoreAccess, error) {
	kinds := storeAccessKinds()
	packages := map[string]*StoreAccess{}
	imported := map[string]map[string]bool{}
	for _, file := range snapshot.Files {
		if file.Test {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			return nil, err
		}
		pkg := path.Dir(file.Path)
		imports := importBases(syntax)
		if imported[pkg] == nil {
			imported[pkg] = map[string]bool{"overgo/" + pkg: true}
		}
		for _, importPath := range imports {
			imported[pkg][importPath] = true
		}
		for _, kind := range kinds {
			count := countFamilySites(kind, syntax, imports)
			if count == 0 {
				continue
			}
			if packages[pkg] == nil {
				packages[pkg] = &StoreAccess{Package: pkg, Sites: map[string]int{}}
			}
			packages[pkg].Sites[kind.Name] += count
			packages[pkg].Total += count
		}
	}
	census := make([]StoreAccess, 0, len(packages))
	for pkg, access := range packages {
		for domain, records := range storeDomains {
			if slices.ContainsFunc(records, func(record string) bool { return imported[pkg][record] }) {
				access.Domains = append(access.Domains, domain)
			}
		}
		slices.Sort(access.Domains)
		census = append(census, *access)
	}
	slices.SortFunc(census, func(left, right StoreAccess) int {
		return cmp.Or(cmp.Compare(right.Total, left.Total), cmp.Compare(left.Package, right.Package))
	})
	return census, nil
}
