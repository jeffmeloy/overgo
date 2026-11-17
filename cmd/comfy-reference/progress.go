package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"slices"
	"sync"
	"time"
)

// ComfyUI tells the client that posted a prompt, over its websocket (/ws?clientId=), each event of the prompt's run as
// a JSON text message ({"type": ..., "data": {...}}); its sampler's previews go as binary messages. A progress event
// says a node finished value of max steps; the recorder stamps each as it arrives, from the prompt's execution_start.
const (
	// clientID names the recorder to ComfyUI: the prompts it posts and the websocket that hears their events.
	clientID = "overgo-comfy-reference"
	// socketAccept is what RFC 6455 appends to the client's key before hashing it into the server's accept.
	socketAccept = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
	// largestMessage bounds a message the recorder reads; a preview is ComfyUI's largest, a few hundred kilobytes.
	largestMessage = 64 << 20
	// socketWait is how long after ComfyUI's history says a run ended its websocket may take to tell the same.
	socketWait = 30 * time.Second
)

// websocket frame opcodes (RFC 6455 5.2).
const (
	opContinuation, opText, opBinary = 0x0, 0x1, 0x2
	opClose, opPing, opPong          = 0x8, 0x9, 0xA
)

// progressStamp is one progress event of a node: value of max steps done, at microseconds after the prompt's
// execution_start event arrived. ComfyUI sends a node's first and last step and, between them, a step that brings a
// preview or comes 100 ms or more after the last it sent (comfy/utils.py ProgressBar), so where steps take less, an
// event tells the steps done since the one before.
type progressStamp struct {
	Value        int   `json:"value"`
	Max          int   `json:"max"`
	Microseconds int64 `json:"us"`
}

// heard is one event the socket heard, with the time it arrived.
type heard struct {
	kind, prompt, node string
	value, max         int
	at                 time.Time
}

// progressWatch hears ComfyUI's events for the recorder's prompts until it is closed.
type progressWatch struct {
	conn   net.Conn
	reader *bufio.Reader
	// writing orders the frames the reader's pong and close write.
	writing sync.Mutex
	mu      sync.Mutex
	events  []heard
	// ended is set when the socket stops hearing; changed is signalled at each event kept and at the end.
	ended   bool
	changed chan struct{}
	done    chan error
}

// watchProgress opens ComfyUI's websocket at server as the recorder's client and hears its events until closed.
func watchProgress(ctx context.Context, server string) (*progressWatch, error) {
	address, err := url.Parse(server)
	if err != nil {
		return nil, err
	}
	var dialer net.Dialer
	conn, err := dialer.DialContext(ctx, "tcp", address.Host)
	if err != nil {
		return nil, err
	}
	key := make([]byte, 16)
	_, _ = rand.Read(key)
	encoded := base64.StdEncoding.EncodeToString(key)
	handshake := "GET /ws?clientId=" + url.QueryEscape(clientID) + " HTTP/1.1\r\nHost: " + address.Host +
		"\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: " + encoded + "\r\nSec-WebSocket-Version: 13\r\n\r\n"
	reader := bufio.NewReader(conn)
	if _, err := io.WriteString(conn, handshake); err != nil {
		return nil, errors.Join(err, conn.Close())
	}
	response, err := http.ReadResponse(reader, &http.Request{Method: http.MethodGet})
	if err != nil {
		return nil, errors.Join(err, conn.Close())
	}
	_ = response.Body.Close()
	sum := sha1.Sum([]byte(encoded + socketAccept))
	if response.StatusCode != http.StatusSwitchingProtocols || response.Header.Get("Sec-WebSocket-Accept") != base64.StdEncoding.EncodeToString(sum[:]) {
		return nil, errors.Join(fmt.Errorf("comfy-reference: ComfyUI's websocket answered %s, not the upgrade", response.Status), conn.Close())
	}
	w := &progressWatch{conn: conn, reader: reader, changed: make(chan struct{}, 1), done: make(chan error, 1)}
	go func() {
		err := w.hear()
		w.mu.Lock()
		w.ended = true
		w.mu.Unlock()
		w.signal()
		w.done <- err
	}()
	return w, nil
}

// signal tells a waiter the events changed, without waiting for one.
func (w *progressWatch) signal() {
	select {
	case w.changed <- struct{}{}:
	default:
	}
}

// hear reads events until the socket closes, keeping each execution_start, progress and execution_success event by
// its prompt.
func (w *progressWatch) hear() error {
	for {
		message, err := w.next()
		if err != nil {
			return err
		}
		var event struct {
			Type string `json:"type"`
			Data struct {
				Prompt string `json:"prompt_id"`
				Node   string `json:"node"`
				Value  int    `json:"value"`
				Max    int    `json:"max"`
			} `json:"data"`
		}
		if json.Unmarshal(message, &event) != nil || (event.Type != "execution_start" && event.Type != "progress" && event.Type != "execution_success") {
			continue
		}
		w.mu.Lock()
		w.events = append(w.events, heard{kind: event.Type, prompt: event.Data.Prompt, node: event.Data.Node, value: event.Data.Value, max: event.Data.Max, at: time.Now()})
		w.mu.Unlock()
		w.signal()
	}
}

// next is the socket's next whole text message: a binary message is passed over, a ping answered, and a close ends it.
func (w *progressWatch) next() ([]byte, error) {
	var message []byte
	var kind byte
	for {
		var header [2]byte
		if _, err := io.ReadFull(w.reader, header[:]); err != nil {
			return nil, err
		}
		final, opcode, masked := header[0]&0x80 != 0, header[0]&0x0F, header[1]&0x80 != 0
		length := uint64(header[1] & 0x7F)
		switch length {
		case 126:
			var extended [2]byte
			if _, err := io.ReadFull(w.reader, extended[:]); err != nil {
				return nil, err
			}
			length = uint64(binary.BigEndian.Uint16(extended[:]))
		case 127:
			var extended [8]byte
			if _, err := io.ReadFull(w.reader, extended[:]); err != nil {
				return nil, err
			}
			length = binary.BigEndian.Uint64(extended[:])
		}
		var mask [4]byte
		if masked {
			if _, err := io.ReadFull(w.reader, mask[:]); err != nil {
				return nil, err
			}
		}
		if length > largestMessage || uint64(len(message))+length > largestMessage {
			return nil, fmt.Errorf("comfy-reference: ComfyUI's websocket sent a message of more than %d bytes", largestMessage)
		}
		payload := make([]byte, length)
		if _, err := io.ReadFull(w.reader, payload); err != nil {
			return nil, err
		}
		if masked {
			for index := range payload {
				payload[index] ^= mask[index%4]
			}
		}
		switch opcode {
		case opClose:
			return nil, io.EOF
		case opPing:
			if err := w.send(opPong, payload); err != nil {
				return nil, err
			}
			continue
		case opPong:
			continue
		case opText, opBinary:
			kind, message = opcode, payload
		case opContinuation:
			message = append(message, payload...)
		default:
			return nil, fmt.Errorf("comfy-reference: ComfyUI's websocket sent a frame of opcode %d", opcode)
		}
		if final {
			if kind == opText {
				return message, nil
			}
			message = nil
		}
	}
}

// send writes one frame of a control message, masked as a client's frames are.
func (w *progressWatch) send(opcode byte, payload []byte) error {
	if len(payload) > 125 {
		return errors.New("comfy-reference: a control frame holds at most 125 bytes")
	}
	var mask [4]byte
	_, _ = rand.Read(mask[:])
	frame := append([]byte{0x80 | opcode, 0x80 | byte(len(payload))}, mask[:]...)
	for index, value := range payload {
		frame = append(frame, value^mask[index%4])
	}
	w.writing.Lock()
	defer w.writing.Unlock()
	_, err := w.conn.Write(frame)
	return err
}

// close ends the watch: it tells ComfyUI the socket closes and waits for hear to end.
func (w *progressWatch) close() {
	_ = w.send(opClose, nil)
	_ = w.conn.Close()
	<-w.done
}

// stamps are prompt's progress events by node, each at microseconds after the prompt's execution_start arrived, read
// once the socket has told the run's success: ComfyUI's history can say the run ended before the socket has passed on
// its last events. A socket that ends, or tells no success within wait, is refused.
func (w *progressWatch) stamps(ctx context.Context, prompt string, wait time.Duration) (map[string][]progressStamp, error) {
	deadline := time.NewTimer(wait)
	defer deadline.Stop()
	for {
		w.mu.Lock()
		succeeded := slices.ContainsFunc(w.events, func(event heard) bool { return event.prompt == prompt && event.kind == "execution_success" })
		ended := w.ended
		w.mu.Unlock()
		if succeeded {
			break
		}
		if ended {
			return nil, errors.New("comfy-reference: ComfyUI's websocket closed before it told the run's success")
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-deadline.C:
			return nil, fmt.Errorf("comfy-reference: ComfyUI's websocket told no success of the run within %s of its history's", wait)
		case <-w.changed:
		}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	var start time.Time
	for _, event := range w.events {
		if event.prompt == prompt && event.kind == "execution_start" {
			start = event.at
			break
		}
	}
	if start.IsZero() {
		return nil, errors.New("comfy-reference: ComfyUI's websocket told the run's success but not its start")
	}
	stamped := map[string][]progressStamp{}
	for _, event := range w.events {
		if event.prompt == prompt && event.kind == "progress" {
			stamped[event.node] = append(stamped[event.node], progressStamp{Value: event.value, Max: event.max, Microseconds: event.at.Sub(start).Microseconds()})
		}
	}
	return stamped, nil
}
