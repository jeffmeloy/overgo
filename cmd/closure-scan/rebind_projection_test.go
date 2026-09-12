package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

func TestClosureRebindProjectionAcceptance(t *testing.T) {
	t.Run("ambiguous successors preserve the original decision", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "internal", "sample", "policy.go")
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("package sample\nfunc value() int { return -1 }\n"), 0600); err != nil {
			t.Fatal(err)
		}
		candidates, err := closurescan.ScanRoot(root, closurescan.CandidateAll)
		if err != nil || len(candidates) != 1 {
			t.Fatalf("original fixture candidates: %d %v", len(candidates), err)
		}
		binding, err := candidates[0].Binding()
		if err != nil {
			t.Fatal(err)
		}
		document, err := closureledger.New(candidates[0].Name, candidates[0].ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed, "One reviewed sentinel.", []closureledger.SourceBinding{binding}, "Retain its owner.", "Sentinel contract changes.", binding.Owner)
		if err != nil {
			t.Fatal(err)
		}
		storePath := filepath.Join(root, "store")
		if _, _, err := commitClosureDocuments(root, storePath, closurePublishOperation, []closureledger.Document{document}, nil, nil); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("package sample\nfunc value() int { x:=0; if x==0 { return -1 }; return -1 }\n"), 0600); err != nil {
			t.Fatal(err)
		}
		snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
		if err != nil {
			t.Fatal(err)
		}
		count, unmatched, _, _, err := importClosureDocuments(root, "store", storePath, snapshot, false, false)
		if err != nil || count != 0 || unmatched != 1 {
			t.Fatalf("ambiguous review moved: count=%d unmatched=%d err=%v", count, unmatched, err)
		}
		store, err := overgodb.OpenReadOnly(storePath)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, sequence := store.Head(); sequence != 1 {
			t.Fatal("ambiguous projection wrote partial state")
		}
		alias, err := closureledger.ActiveAlias(binding)
		if err != nil {
			t.Fatal(err)
		}
		active, bound, err := store.ResolveAlias(t.Context(), alias)
		if err != nil || !bound || active != document.ID {
			t.Fatal("ambiguous review was retired or replaced")
		}
	})
	for _, mode := range []string{"import", "catalog", "cached", "changed-source", "changed-head"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.24\n"), 0600); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(root, "internal", "sample", "rules.go")
			if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
				t.Fatal(err)
			}
			before := `package sample
import("regexp";"strings")
func count(kind, response string, patterns []*regexp.Regexp, values []string) int {
 count := 0
 switch kind {
 case "matches":
  for _, pattern := range patterns { count += len(pattern.FindAllStringIndex(response, -1)) }
 case "highlights":
  for _, pattern := range patterns {
   for _, match := range pattern.FindAllString(response, -1) { if strings.TrimSpace(match) != "" { count++ } }
  }
 case "paragraphs": count = len(strings.Split(response, "***"))
 case "paired":
  parts := strings.Split(response, "******")
  if len(parts) == 2 && parts[0] != parts[1] { count++ }
 case "title":
  for _, match := range patterns[0].FindAllString(response, -1) { if strings.TrimSpace(match) != "" { count++ } }
 }
 return count
}
`
			write := func(source string) repoanalysis.SourceSnapshot {
				t.Helper()
				if err := os.WriteFile(path, []byte(source), 0600); err != nil {
					t.Fatal(err)
				}
				snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
				if err != nil {
					t.Fatal(err)
				}
				return snapshot
			}
			if err := os.WriteFile(filepath.Join(filepath.Dir(path), "unrelated.go"), []byte("package sample\nconst UnrelatedPolicy = 7\n"), 0600); err != nil {
				t.Fatal(err)
			}
			write(before)
			candidates, err := closurescan.ScanRoot(root, closurescan.CandidateAll)
			if err != nil {
				t.Fatal(err)
			}
			var documents []closureledger.Document
			for _, candidate := range candidates {
				if string(candidate.ValueJSON()) != "-1" && string(candidate.ValueJSON()) != "2" && candidate.Name != "UnrelatedPolicy" {
					continue
				}
				binding, err := candidate.Binding()
				if err != nil {
					t.Fatal(err)
				}
				document, err := closureledger.New(candidate.Name, candidate.ValueJSON(), closureledger.TierImplementation, closureledger.StatusClosed, "All regex matches are required by the fixture protocol.", []closureledger.SourceBinding{binding}, "Keep complete enumeration.", "Regex API or fixture protocol changes.", binding.Owner)
				if err != nil {
					t.Fatal(err)
				}
				documents = append(documents, document)
			}
			if len(documents) != 5 {
				t.Fatalf("fixture has %d reviewed sites", len(documents))
			}
			storePath := filepath.Join(root, "store")
			if _, _, err := commitClosureDocuments(root, storePath, closurePublishOperation, documents, nil, nil); err != nil {
				t.Fatal(err)
			}
			after := strings.Replace(before, " case \"highlights\":", " case \"letters\": count = strings.Count(response, values[0])\n case \"highlights\":", 1)
			snapshot := write(after)
			var count, unmatched int
			var first string
			expectedSequence := uint64(2)
			if mode == "import" {
				count, unmatched, first, _, err = importClosureDocuments(root, "store", storePath, snapshot, false, false)
			} else {
				store, err := overgodb.OpenReadOnly(storePath)
				if err != nil {
					t.Fatal(err)
				}
				report, err := buildUnclassifiedPolicyReport(t.Context(), snapshot, store)
				if err != nil {
					t.Fatal(err)
				}
				if err := store.Close(); err != nil {
					t.Fatal(err)
				}
				var names []string
				for _, candidate := range report.Candidates {
					if string(candidate.Current.ValueJSON()) != "-1" {
						continue
					}
					if !candidate.ActiveRebind || candidate.Category != unclassifiedRecoverable {
						t.Fatalf("restorable site classified as new: %+v", candidate)
					}
					names = append(names, candidate.Current.Name)
				}
				if len(names) != 2 {
					t.Fatalf("inventory omitted moved reviews: %d", len(names))
				}
				if mode == "catalog" {
					var added int
					added, count, err = catalogCandidates(root, "store", snapshot, names, catalogText{tier: string(closureledger.TierImplementation), understanding: "This text must not replace a prior active review.", closurePath: "No replacement.", rerankTrigger: "No replacement."})
					if added != 0 {
						t.Fatal("existing review was catalogued as new")
					}
				} else {
					if mode == "changed-source" {
						snapshot = write(after + "\n// New snapshot.\n")
					}
					if mode == "changed-head" {
						writer, err := overgodb.Open(storePath)
						if err != nil {
							t.Fatal(err)
						}
						content, err := (artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/json", Schema: "test/unrelated/v1"}).ContentBytes([]byte(`{"observation":"unrelated"}`))
						if err != nil {
							t.Fatal(err)
						}
						batch, err := artifact.NewDocumentBatch("unrelated", []artifact.Content{content}, nil, nil)
						if err != nil {
							t.Fatal(err)
						}
						if _, err := writer.Commit(t.Context(), batch); err != nil {
							t.Fatal(err)
						}
						if err := writer.Close(); err != nil {
							t.Fatal(err)
						}
						expectedSequence++
					}
					var review closureAliasReview
					count, unmatched, first, review, err = importClosureDocumentsWith(root, "store", storePath, snapshot, false, false, nil, report.review)
					if review.HistoryReused != (mode == "cached") {
						t.Fatal("review reuse ignored its source/store identity")
					}
				}
			}

			if err != nil || unmatched != 0 || count != len(documents)-1 {
				t.Fatalf("one import did not settle shifted sites: rebound=%d unmatched=%d first=%s error=%v", count, unmatched, first, err)
			}
			store, err := overgodb.OpenReadOnly(storePath)
			if err != nil {
				t.Fatal(err)
			}
			_, sequence := store.Head()
			if sequence != expectedSequence {
				t.Fatalf("rebind used %d transactions after initial publication", sequence-1)
			}
			current, err := closurescan.ScanRoot(root, closurescan.CandidateAll)
			if err != nil {
				t.Fatal(err)
			}
			for _, candidate := range current {
				if string(candidate.ValueJSON()) != "-1" {
					continue
				}
				binding, err := candidate.Binding()
				if err != nil {
					t.Fatal(err)
				}
				document, bound, err := closureledger.ResolveActiveBinding(t.Context(), store, binding, candidate.ValueJSON())
				if err != nil || !bound || document.Understanding != "All regex matches are required by the fixture protocol." {
					t.Fatalf("review changed or absent: %t %v", bound, err)
				}
			}
			for _, original := range documents {
				if original.Name != "UnrelatedPolicy" {
					continue
				}
				document, bound, err := closureledger.ResolveActiveBinding(t.Context(), store, original.Bindings[0], original.Value)
				if err != nil || !bound || document.ID != original.ID {
					t.Fatal("unrelated review changed")
				}
			}
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}
			count, unmatched, first, _, err = importClosureDocuments(root, "store", storePath, snapshot, false, false)
			if err != nil || unmatched != 0 || count != 0 {
				t.Fatalf("repeat was not exact reuse: count=%d unmatched=%d first=%s err=%v", count, unmatched, first, err)
			}
			store, err = overgodb.OpenReadOnly(storePath)
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			if _, after := store.Head(); after != sequence {
				t.Fatal("repeat changed durable state")
			}

		})
	}
}
