package processcontrol

import (
	"bytes"
	"cmp"
	"io"
	"sync"
)

// ListeningAnnouncement prefixes the line a served child writes once its
// listener is bound; a launcher connects when it reads the line.
const ListeningAnnouncement = "listening on http://"

// LineWatch forwards a child's output to a sink and reports the first line
// carrying a prefix, so a launcher waits on the child's own announcement.
type LineWatch struct {
	prefix []byte
	sink   io.Writer

	mu      sync.Mutex
	partial []byte
	found   bool
	line    chan string
}

// WatchLine builds the watch; the sink may be nil.
func WatchLine(sink io.Writer, prefix string) *LineWatch {
	return &LineWatch{prefix: []byte(prefix), sink: cmp.Or(sink, io.Writer(io.Discard)), line: make(chan string, 1)}
}

// Line delivers the remainder of the first announced line after its prefix.
func (w *LineWatch) Line() <-chan string { return w.line }

// Write forwards the data and scans it line by line for the prefix.
func (w *LineWatch) Write(data []byte) (int, error) {
	w.mu.Lock()
	if !w.found {
		w.partial = append(w.partial, data...)
		for {
			line, rest, found := bytes.Cut(w.partial, []byte{'\n'})
			if !found {
				break
			}
			w.partial = rest
			line = bytes.TrimRight(line, "\r")
			if _, remainder, announced := bytes.Cut(line, w.prefix); announced {
				w.found = true
				w.partial = nil
				w.line <- string(remainder)
				break
			}
		}
	}
	w.mu.Unlock()
	return w.sink.Write(data)
}
