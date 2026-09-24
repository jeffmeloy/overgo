package webuilane

import (
	"bufio"
	"encoding/json"
	"io"
	"net"
	"strings"
	"testing"
)

// pageSide plays the page end of a DevTools socket: it drains what the lane
// sends and answers with the frames given, unmasked as a server sends them.
func pageSide(t *testing.T) (*webSocket, func(string)) {
	t.Helper()
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	go func() { _, _ = io.Copy(io.Discard, server) }()
	send := func(message string) {
		frame := []byte{webSocketFinalBit | webSocketText, byte(len(message))}
		go func() { _, _ = server.Write(append(frame, message...)) }()
	}
	return &webSocket{connection: client, reader: bufio.NewReader(client)}, send
}

// TestBrowserLegFailsWhenTheBrowserStopsAnswering holds every wait on the
// page to the page's own report that it has ended: a renderer crash or a
// detach while a call waits for its reply fails that call at once, naming
// the call and the event, and an event wait fails the same way; an unrelated
// event does not end the wait, and the call's reply still answers it.
func TestBrowserLegFailsWhenTheBrowserStopsAnswering(t *testing.T) {
	t.Parallel()
	for _, event := range []string{"Inspector.targetCrashed", "Inspector.detached"} {
		socket, send := pageSide(t)
		send(`{"method":"` + event + `","params":{}}`)
		err := socket.call(t.Context(), "Runtime.evaluate", nil, nil)
		if err == nil || !strings.Contains(err.Error(), "Runtime.evaluate") || !strings.Contains(err.Error(), event) {
			t.Fatalf("%s: call = %v", event, err)
		}
		socket, send = pageSide(t)
		send(`{"method":"` + event + `","params":{}}`)
		if err := socket.awaitEvent(t.Context(), func(string, json.RawMessage) bool { return false }); err == nil || !strings.Contains(err.Error(), event) {
			t.Fatalf("%s: event wait = %v", event, err)
		}
	}
	socket, send := pageSide(t)
	send(`{"method":"Page.frameNavigated","params":{}}`)
	send(`{"id":1,"result":{}}`)
	if err := socket.call(t.Context(), "Runtime.evaluate", nil, nil); err != nil {
		t.Fatalf("an unrelated event ended the call: %v", err)
	}
}
