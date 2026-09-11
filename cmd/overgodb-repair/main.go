//overgo:runtime-inputs caller

// overgodb-repair truncates a torn or corrupt journal tail. The journal
// is an append-only chain of checksummed frames; corruption that enters
// through a dying or memory-corrupted writer can only live in the final
// frames, so the repair walks the chain, finds the last frame whose
// transaction still decodes, and swaps in a copy cut at that boundary.
// The corrupt original is renamed alongside, never deleted.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

const (
	storeHeaderBytes = 16
	frameHeaderBytes = 84
	frameMagic       = "RDB1"
	crcBytes         = 4
)

func main() {
	clioptions.MainNamed("overgodb-repair", run)
}

func run() error {
	repo := flag.String("repo", "overgodb-store", "OvergoDB store directory")
	apply := flag.Bool("apply", false, "perform the repair; default reports only")
	lineage := flag.Bool("lineage", false, "commit the lineage edges committed lifecycle events declare but the store lacks (additive typed-document repair)")
	flag.Parse()
	if *lineage {
		return reconcileLineage(*repo)
	}
	journal := filepath.Join(*repo, "overgodb.log")
	validEnd, total, frames, badFrames, err := scanJournal(journal)
	if err != nil {
		return err
	}
	fmt.Printf("journal=%s size=%d frames_valid=%d valid_end=%d corrupt_tail_bytes=%d corrupt_frames=%d\n",
		journal, total, frames, validEnd, total-validEnd, badFrames)
	if validEnd == total {
		fmt.Println("journal is clean; nothing to repair")
		return nil
	}
	if !*apply {
		fmt.Println("re-run with -apply to truncate the corrupt tail (the original is kept alongside)")
		return nil
	}
	repaired := journal + ".repaired"
	if err := copyPrefix(journal, repaired, validEnd); err != nil {
		return err
	}
	corrupt := journal + ".corrupt"
	for suffix := 0; ; suffix++ {
		candidate := corrupt
		if suffix > 0 {
			candidate = fmt.Sprintf("%s.%d", corrupt, suffix)
		}
		if _, err := os.Stat(candidate); errors.Is(err, os.ErrNotExist) {
			corrupt = candidate
			break
		}
	}
	if err := os.Rename(journal, corrupt); err != nil {
		return err
	}
	if err := os.Rename(repaired, journal); err != nil {
		return err
	}
	fmt.Printf("repaired: corrupt original kept at %s\n", corrupt)
	return nil
}

// scanJournal walks the frame chain and returns the byte offset after
// the last frame whose envelope and transaction document both decode.
func scanJournal(path string) (validEnd, total int64, frames, badFrames int, err error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return 0, 0, 0, 0, err
	}
	total = info.Size()
	reader := bufio.NewReaderSize(file, 1<<20)
	if _, err := io.CopyN(io.Discard, reader, storeHeaderBytes); err != nil {
		return 0, total, 0, 0, fmt.Errorf("overgodb-repair: store header: %w", err)
	}
	validEnd = storeHeaderBytes
	header := make([]byte, frameHeaderBytes)
	var payload []byte
	for {
		if _, err := io.ReadFull(reader, header); err != nil {
			return validEnd, total, frames, badFrames, nil
		}
		if string(header[:4]) != frameMagic {
			badFrames++
			return validEnd, total, frames, badFrames, nil
		}
		payloadSize := int64(binary.LittleEndian.Uint32(header[80:84]))
		body := payloadSize + crcBytes
		if int64(cap(payload)) < body {
			payload = make([]byte, body)
		}
		payload = payload[:body]
		if _, err := io.ReadFull(reader, payload); err != nil {
			badFrames++
			return validEnd, total, frames, badFrames, nil
		}
		if !transactionDecodes(payload[:payloadSize]) {
			badFrames++
			return validEnd, total, frames, badFrames, nil
		}
		frames++
		validEnd += frameHeaderBytes + body
	}
}

// transactionDecodes applies the same framing the store applies: a
// little-endian u32 metadata size, then the transaction document as
// JSON, then raw content bytes.
func transactionDecodes(payload []byte) bool {
	if len(payload) < 4 {
		return false
	}
	metadataSize := int(binary.LittleEndian.Uint32(payload))
	if metadataSize < 2 || metadataSize > len(payload)-4 {
		return false
	}
	return json.Valid(payload[4 : 4+metadataSize])
}

func copyPrefix(source, destination string, bytesToCopy int64) error {
	in, err := os.Open(source)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(destination)
	if err != nil {
		return err
	}
	if _, err := io.CopyN(out, in, bytesToCopy); err != nil {
		_ = out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		_ = out.Close()
		return err
	}
	return out.Close()
}

// reconcileLineage repairs the typed-document layer: lifecycle events
// published without their declared lineage edges leave evidence invisible
// to lineage-closure consumers such as compaction's retained set.
func reconcileLineage(repo string) error {
	store, err := overgodb.Open(repo)
	if err != nil {
		return err
	}
	defer store.Close()
	report, err := modelrecipe.ReconcileLifecycleLineage(context.Background(), store)
	if err != nil {
		return err
	}
	fmt.Printf("lineage reconciled: events=%d missing_edges_committed=%d skipped_events=%d\n",
		report.Events, report.MissingEdges, report.SkippedEvents)
	return nil
}
