// closure-scan: the native magic baseline tool (floor component 5, Automation
// Doctrine Layer 6). Report mode ranks overgo's numeric constants as closure
// candidates; -triage mode emits selected constants as closure-ledger
// documents in the store, owned and pinned by their declaring source file.
// The scan core lives in internal/closurescan, shared with the gate's magic
// step. Full catalog is ongoing triage; this tool is the mechanism.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/repoanalysis"
	"overgo/internal/repodb"
	"overgo/internal/strictjson"
)

type triageRow struct {
	Name          string `json:"name"`
	File          string `json:"file"`
	Tier          string `json:"tier"`
	Status        string `json:"status"`
	ClosurePath   string `json:"closure_path"`
	RerankTrigger string `json:"rerank_trigger"`
}

type triageFile struct {
	Rows []triageRow `json:"rows"`
}

func main() {
	triagePath := flag.String("triage", "", "triage JSON ({rows:[{name,file,tier,status,closure_path,rerank_trigger}]}); emits closure documents to the store")
	storePath := flag.String("store", "repodb-store", "RepoDB store directory (emit mode)")
	limit := flag.Int("limit", 40, "report mode: top-N candidates to print")
	raw := flag.Bool("raw", false, "rank repeated raw policy literals instead of declared constants")
	flag.Parse()
	root, err := os.Getwd()
	if err != nil {
		fatal(err)
	}
	if *raw {
		ranked, err := closurescan.RankRawPolicyLiterals(mustSnapshot(root))
		if err != nil {
			fatal(err)
		}
		reportRaw(ranked, *limit)
		return
	}
	candidates, err := closurescan.ScanRoot(root)
	if err != nil {
		fatal(err)
	}
	if *triagePath == "" {
		report(candidates, *limit)
		return
	}
	if err := emit(root, *storePath, *triagePath, candidates); err != nil {
		fatal(err)
	}
}

func mustSnapshot(root string) repoanalysis.SourceSnapshot {
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		fatal(err)
	}
	return snapshot
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "closure-scan: %v\n", err)
	os.Exit(1)
}

func report(candidates []closurescan.Candidate, limit int) {
	fmt.Printf("closure-scan: %d numeric constants (iota enums, tests, generated excluded)\n", len(candidates))
	fmt.Printf("%-6s %-44s %-16s %s\n", "score", "const", "value", "file")
	for index, row := range candidates {
		if index >= limit {
			fmt.Printf("... %d more (raise -limit)\n", len(candidates)-limit)
			break
		}
		fmt.Printf("%-6d %-44s %-16s %s\n", row.Score, row.Name, row.Value, row.File)
	}
}

func reportRaw(candidates []closurescan.RawPolicyLiteral, limit int) {
	fmt.Printf("closure-scan: %d repeated raw policy candidates (tests, generated, structural math excluded)\n", len(candidates))
	fmt.Printf("%-6s %-8s %-24s %-28s %s\n", "score", "count", "value", "package", "functions")
	for index, row := range candidates {
		if index >= limit {
			fmt.Printf("... %d more (raise -limit)\n", len(candidates)-limit)
			break
		}
		owners := row.Functions
		if len(owners) > 5 {
			owners = append(append([]string(nil), owners[:5]...), fmt.Sprintf("+%d", len(row.Functions)-5))
		}
		fmt.Printf("%-6d %-8d %-24s %-28s %s\n", row.Score, row.Count, row.Value, row.Package, strings.Join(owners, ","))
	}
}

func emit(root, storePath, triagePath string, candidates []closurescan.Candidate) error {
	raw, err := os.ReadFile(triagePath)
	if err != nil {
		return err
	}
	var triage triageFile
	if err := strictjson.DecodeBytes(raw, &triage); err != nil {
		return fmt.Errorf("triage file: %w", err)
	}
	if len(triage.Rows) == 0 {
		return fmt.Errorf("triage file has no rows")
	}
	byKey := map[string]closurescan.Candidate{}
	for _, row := range candidates {
		byKey[row.File+"#"+row.Name] = row
	}
	// Content-derived batch key: identical triage re-emits idempotently,
	// different triage gets its own key (a fixed key collided on the second
	// ever emit).
	digest := sha256.Sum256(raw)
	batch := artifact.Batch{Key: "closure-scan/" + hex.EncodeToString(digest[:8])}
	fileIDs := map[string]artifact.ID{}
	for _, row := range triage.Rows {
		found, ok := byKey[row.File+"#"+row.Name]
		if !ok {
			return fmt.Errorf("triage row %s not found by scan in %s (stale triage?)", row.Name, row.File)
		}
		fileID, ok := fileIDs[row.File]
		if !ok {
			id, descriptor, location, err := fileArtifact(root, row.File)
			if err != nil {
				return err
			}
			batch.Artifacts = append(batch.Artifacts, descriptor)
			batch.Locations = append(batch.Locations, location)
			fileIDs[row.File] = id
			fileID = id
		}
		valueJSON, err := json.Marshal(json.Number(found.Value))
		if err != nil || !json.Valid(valueJSON) {
			valueJSON, _ = json.Marshal(found.Value) // composite expressions pin as strings
		}
		document, err := closureledger.New(
			row.Name, valueJSON,
			closureledger.Tier(row.Tier), closureledger.Status(row.Status),
			[]artifact.ID{fileID}, row.ClosurePath, row.RerankTrigger, fileID,
		)
		if err != nil {
			return fmt.Errorf("row %s: %w", row.Name, err)
		}
		content, err := document.Content()
		if err != nil {
			return err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, document.Lineage()...)
	}
	store, err := repodb.Open(filepath.Join(root, storePath))
	if err != nil {
		return err
	}
	defer store.Close()
	commit, err := store.Commit(context.Background(), batch)
	if err != nil {
		return err
	}
	fmt.Printf("emitted %d closure documents (%d owner files) commit %x\n",
		len(triage.Rows), len(fileIDs), commit[:8])
	return nil
}

func fileArtifact(root, relative string) (artifact.ID, artifact.Descriptor, artifact.LocationEvent, error) {
	resolved := filepath.Join(root, filepath.FromSlash(relative))
	file, err := os.Open(resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	defer file.Close()
	id, size, err := artifact.Identify(artifact.KindFile, file)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	absolute, err := filepath.Abs(resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	return id, artifact.Descriptor{ID: id, Size: size}, artifact.LocationEvent{
		Location: artifact.Location{Artifact: id, Kind: artifact.LocationFile, Value: absolute},
		Action:   artifact.LocationAdd,
	}, nil
}
