package automationcheck

import (
	"overgo/internal/runrecord"
)

const publishedImpact Fact = "authority:published-documents"

// PublishedCheck proves published OvergoDB documents still load through the
// production readers whenever a document-schema-owning package changes. It is
// the data half of the source-data contract: every other gate check compares
// source against source, which is exactly why an unversioned schema change
// once shipped green while every stored profile became unreadable.
func PublishedCheck(root string, command Command) Check {
	return Check{
		Descriptor: Descriptor{
			Name: "published", Phase: runrecord.PhaseValidate,
			Triggers: []Fact{publishedImpact}, Inapplicable: "no document schema owner changed",
			Requirements: Requirements{Process: ProcessToolchain},
			Ownership: Ownership{Fact: publishedImpact, Packages: []string{
				"internal/modelrecipe", "internal/recipe", "internal/artifact",
				"internal/modelartifact", "internal/runrecord", "cmd/store-check",
			}},
		},
		Run: commandRunner(root, command, "go", "run", "./cmd/store-check"),
	}
}
