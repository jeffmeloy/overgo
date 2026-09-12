// Package webuilane owns the real-browser acceptance transport used by Overgo's GUI lane.
package webuilane

import (
	"bufio"
	"bytes"
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
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
)

const (
	webSocketGUID                 = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
	webSocketProtocolVersion      = "13"
	webSocketKeyBytes             = 16
	webSocketMaskBytes            = 4
	webSocketBaseHeaderBytes      = 2
	webSocketLength64Bytes        = 8
	webSocketFinalBit        byte = 0x80
	webSocketOpcodeMask      byte = 0x0f
	webSocketLengthMask      byte = 0x7f
	webSocketContinuation    byte = 0x0
	webSocketText            byte = 0x1
	webSocketClose           byte = 0x8
	webSocketPing            byte = 0x9
	webSocketPong            byte = 0xA
	webSocketLength16             = 126
	webSocketLength64             = 127
)

// Browser is one isolated Chromium target controlled through its DevTools socket.
type Browser struct {
	supervised *processcontrol.Supervised
	wait       context.Context
	profile    string
	socket     *webSocket
	output     *bytes.Buffer
}

// FindBrowser resolves an explicit path or an installed Chromium browser.
func FindBrowser(explicit string) (string, error) {
	if explicit != "" {
		path, err := filepath.Abs(explicit)
		if err == nil {
			if info, statErr := os.Stat(path); statErr == nil && !info.IsDir() {
				return path, nil
			}
		}
		return "", errors.New("webui lane: explicit browser executable is unavailable")
	}
	for _, name := range []string{"chrome", "chrome.exe", "chromium", "chromium.exe", "msedge", "msedge.exe"} {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}
	if runtime.GOOS == "windows" {
		for _, candidate := range windowsBrowserCandidates() {
			if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
				return candidate, nil
			}
		}
	}
	return "", errors.New("webui lane: Chrome, Chromium, or Edge is required")
}

func windowsBrowserCandidates() []string {
	var result []string
	for _, root := range []string{os.Getenv("ProgramFiles"), os.Getenv("ProgramFiles(x86)"), os.Getenv("LocalAppData")} {
		if root == "" {
			continue
		}
		result = append(result,
			filepath.Join(root, "Google", "Chrome", "Application", "chrome.exe"),
			filepath.Join(root, "Microsoft", "Edge", "Application", "msedge.exe"),
		)
	}
	return result
}

// Open launches an isolated headless browser and opens one real page target.
func Open(ctx context.Context, executable, pageURL string) (*Browser, error) {
	if ctx == nil || executable == "" || pageURL == "" {
		return nil, errors.New("webui lane: browser, URL, and context are required")
	}
	profile, err := os.MkdirTemp("", "overgo-webui-lane-")
	if err != nil {
		return nil, err
	}
	// Chromium also uses its default directory for intermediate downloads.
	// CDP's destination override alone does not isolate those files.
	downloads := filepath.Join(profile, "downloads")
	preferences, err := json.Marshal(map[string]any{
		"download": map[string]any{"default_directory": downloads, "prompt_for_download": false},
		"savefile": map[string]any{"default_directory": downloads},
	})
	if err != nil {
		return nil, errors.Join(err, removeProfile(profile))
	}
	if err := os.MkdirAll(filepath.Join(profile, "Default"), clioptions.OutputDirectoryMode); err != nil {
		return nil, errors.Join(err, removeProfile(profile))
	}
	if err := os.WriteFile(filepath.Join(profile, "Default", "Preferences"), preferences, clioptions.PrivateFileMode); err != nil {
		return nil, errors.Join(err, removeProfile(profile))
	}
	output := &bytes.Buffer{}
	supervised, err := processcontrol.Start(ctx, processcontrol.Command{
		Path: executable,
		Args: []string{
			"--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
			"--disable-background-networking", "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
			"--user-data-dir=" + profile, "about:blank",
		},
		Stdout: output,
		Stderr: output,
	})
	if err != nil {
		return nil, errors.Join(err, removeProfile(profile))
	}
	browser := &Browser{supervised: supervised, wait: ctx, profile: profile, output: output}
	port, err := waitDevToolsPort(ctx, profile, supervised)
	if err != nil {
		closeErr := browser.Close()
		return nil, errors.Join(fmt.Errorf("webui lane: browser debugging endpoint: %w: %s", err, output.String()), closeErr)
	}
	target, err := createTarget(ctx, port, "about:blank")
	if err != nil {
		return nil, errors.Join(err, browser.Close())
	}
	browser.socket, err = dialWebSocket(ctx, target)
	if err != nil {
		return nil, errors.Join(err, browser.Close())
	}
	if err := browser.Call(ctx, "Runtime.enable", nil, nil); err != nil {
		return nil, errors.Join(err, browser.Close())
	}
	// Downloads belong to this run and leave with its temporary profile.
	if err := browser.Call(ctx, "Browser.setDownloadBehavior", map[string]any{
		"behavior": "allow", "downloadPath": downloads,
	}, nil); err != nil {
		return nil, errors.Join(err, browser.Close())
	}
	if err := browser.Call(ctx, "Page.navigate", map[string]any{"url": pageURL}, nil); err != nil {
		return nil, errors.Join(err, browser.Close())
	}
	if err := browser.Eventually(ctx, `document.readyState === "complete"`); err != nil {
		return nil, errors.Join(err, browser.Close())
	}
	return browser, nil
}

// Close terminates the browser and removes only its verified temporary profile.
func (browser *Browser) Close() error {
	if browser == nil {
		return nil
	}
	if browser.socket != nil {
		_ = browser.socket.close()
	}
	var stopErr, waitErr error
	if browser.supervised != nil {
		if !browser.supervised.Exited() {
			stopErr = browser.supervised.Terminate()
		}
		_, waitErr = browser.supervised.Wait(context.WithoutCancel(browser.wait))
	}
	return errors.Join(stopErr, waitErr, removeProfile(browser.profile))
}

// Call invokes one DevTools method.
func (browser *Browser) Call(ctx context.Context, method string, parameters any, result any) error {
	if browser == nil || browser.socket == nil {
		return errors.New("webui lane: browser socket is unavailable")
	}
	return browser.socket.call(ctx, method, parameters, result)
}

// Evaluate runs JavaScript in the page and decodes its returned value.
func (browser *Browser) Evaluate(ctx context.Context, expression string, result any) error {
	var response struct {
		Result struct {
			Value json.RawMessage `json:"value"`
		} `json:"result"`
		Exception json.RawMessage `json:"exceptionDetails"`
	}
	if err := browser.Call(ctx, "Runtime.evaluate", map[string]any{
		"expression": expression, "returnByValue": true, "awaitPromise": true,
	}, &response); err != nil {
		return err
	}
	if len(response.Exception) != 0 {
		return fmt.Errorf("webui lane: JavaScript exception: %s", response.Exception)
	}
	if result == nil {
		return nil
	}
	if len(response.Result.Value) == 0 {
		return errors.New("webui lane: JavaScript result has no value")
	}
	return json.Unmarshal(response.Result.Value, result)
}

// Eventually waits for a JavaScript predicate while the enclosing context owns the deadline.
func (browser *Browser) Eventually(ctx context.Context, expression string) error {
	ticker := time.Tick(time.Millisecond)
	for {
		var ready bool
		if err := browser.Evaluate(ctx, expression, &ready); err == nil && ready {
			return nil
		}
		select {
		case <-ctx.Done():
			return errors.Join(errors.New("webui lane: browser predicate did not become true"), ctx.Err())
		case <-ticker:
		}
	}
}

// SetViewport drives the responsive layout through the browser's emulation owner.
func (browser *Browser) SetViewport(ctx context.Context, width, height int) error {
	if width <= 0 || height <= 0 {
		return errors.New("webui lane: viewport dimensions must be positive")
	}
	return browser.Call(ctx, "Emulation.setDeviceMetricsOverride", map[string]any{
		"width": width, "height": height, "deviceScaleFactor": float64(width) / float64(width), "mobile": false,
	}, nil)
}

func waitDevToolsPort(ctx context.Context, profile string, supervised *processcontrol.Supervised) (int, error) {
	ticker := time.Tick(time.Millisecond)
	path := filepath.Join(profile, "DevToolsActivePort")
	for {
		data, err := os.ReadFile(path)
		if err == nil {
			line, _, _ := strings.Cut(string(data), "\n")
			port, parseErr := strconv.Atoi(strings.TrimSpace(line))
			if parseErr == nil && port > 0 {
				return port, nil
			}
		}
		if supervised.Exited() {
			return 0, errors.New("browser exited before publishing DevTools port")
		}
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-ticker:
		}
	}
}

func createTarget(ctx context.Context, port int, pageURL string) (string, error) {
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/json/new?%s", port, url.QueryEscape(pageURL))
	request, err := http.NewRequestWithContext(ctx, http.MethodPut, endpoint, nil)
	if err != nil {
		return "", err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	var target struct {
		WebSocket string `json:"webSocketDebuggerUrl"`
	}
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("webui lane: create browser target returned %s", response.Status)
	}
	if err := json.NewDecoder(response.Body).Decode(&target); err != nil || target.WebSocket == "" {
		return "", errors.Join(errors.New("webui lane: browser target omitted DevTools socket"), err)
	}
	return target.WebSocket, nil
}

func removeProfile(profile string) error {
	if profile == "" {
		return nil
	}
	temporary, err := filepath.Abs(os.TempDir())
	resolved, resolveErr := filepath.Abs(profile)
	if err != nil || resolveErr != nil || filepath.Dir(resolved) != temporary ||
		!strings.HasPrefix(filepath.Base(resolved), "overgo-webui-lane-") {
		return errors.New("webui lane: cleanup path is not an owned temporary profile")
	}
	return os.RemoveAll(resolved)
}

type webSocket struct {
	connection net.Conn
	reader     *bufio.Reader
	mu         sync.Mutex
	nextID     uint64
}

func dialWebSocket(ctx context.Context, rawURL string) (*webSocket, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme != "ws" || parsed.Host == "" {
		return nil, errors.Join(errors.New("webui lane: invalid DevTools socket URL"), err)
	}
	connection, err := (&net.Dialer{}).DialContext(ctx, "tcp", parsed.Host)
	if err != nil {
		return nil, err
	}
	keyBytes := make([]byte, webSocketKeyBytes)
	if _, err := rand.Read(keyBytes); err != nil {
		connection.Close()
		return nil, err
	}
	key := base64.StdEncoding.EncodeToString(keyBytes)
	request := &http.Request{
		Method: http.MethodGet, URL: parsed, Host: parsed.Host,
		Header: http.Header{
			"Connection": {"Upgrade"}, "Upgrade": {"websocket"}, "Sec-Websocket-Version": {webSocketProtocolVersion},
			"Sec-Websocket-Key": {key},
		},
	}
	if err := request.Write(connection); err != nil {
		connection.Close()
		return nil, err
	}
	reader := bufio.NewReader(connection)
	response, err := http.ReadResponse(reader, request)
	if err != nil {
		connection.Close()
		return nil, err
	}
	expectedHash := sha1.Sum([]byte(key + webSocketGUID))
	expected := base64.StdEncoding.EncodeToString(expectedHash[:])
	if response.StatusCode != http.StatusSwitchingProtocols || response.Header.Get("Sec-Websocket-Accept") != expected {
		connection.Close()
		return nil, fmt.Errorf("webui lane: DevTools socket upgrade returned %s", response.Status)
	}
	return &webSocket{connection: connection, reader: reader}, nil
}

func (socket *webSocket) close() error {
	if socket == nil || socket.connection == nil {
		return nil
	}
	return socket.connection.Close()
}

func (socket *webSocket) call(ctx context.Context, method string, parameters any, result any) error {
	socket.mu.Lock()
	defer socket.mu.Unlock()
	socket.nextID++
	id := socket.nextID
	payload, err := json.Marshal(struct {
		ID     uint64 `json:"id"`
		Method string `json:"method"`
		Params any    `json:"params,omitempty"`
	}{ID: id, Method: method, Params: parameters})
	if err != nil {
		return err
	}
	if deadline, found := ctx.Deadline(); found {
		_ = socket.connection.SetDeadline(deadline)
	}
	if err := socket.writeText(payload); err != nil {
		return err
	}
	for {
		message, err := socket.readText()
		if err != nil {
			return err
		}
		var envelope struct {
			ID     uint64          `json:"id"`
			Result json.RawMessage `json:"result"`
			Error  *struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(message, &envelope); err != nil || envelope.ID != id {
			continue
		}
		if envelope.Error != nil {
			return fmt.Errorf("webui lane: DevTools %s failed (%d): %s", method, envelope.Error.Code, envelope.Error.Message)
		}
		if result == nil || len(envelope.Result) == 0 {
			return nil
		}
		return json.Unmarshal(envelope.Result, result)
	}
}

func (socket *webSocket) writeText(payload []byte) error {
	mask := make([]byte, webSocketMaskBytes)
	if _, err := rand.Read(mask); err != nil {
		return err
	}
	header := []byte{webSocketFinalBit | webSocketText}
	switch {
	case len(payload) < webSocketLength16:
		header = append(header, webSocketFinalBit|byte(len(payload)))
	case uint64(len(payload)) <= uint64(^uint16(0)):
		header = append(header, webSocketFinalBit|webSocketLength16)
		header = append(header, make([]byte, webSocketBaseHeaderBytes)...)
		binary.BigEndian.PutUint16(header[len(header)-webSocketBaseHeaderBytes:], uint16(len(payload)))
	default:
		header = append(header, webSocketFinalBit|webSocketLength64)
		header = append(header, make([]byte, webSocketLength64Bytes)...)
		binary.BigEndian.PutUint64(header[len(header)-webSocketLength64Bytes:], uint64(len(payload)))
	}
	header = append(header, mask...)
	masked := make([]byte, len(payload))
	for index := range payload {
		masked[index] = payload[index] ^ mask[index%len(mask)]
	}
	if _, err := socket.connection.Write(header); err != nil {
		return err
	}
	_, err := socket.connection.Write(masked)
	return err
}

func (socket *webSocket) readText() ([]byte, error) {
	var assembled []byte
	for {
		header := make([]byte, webSocketBaseHeaderBytes)
		if _, err := io.ReadFull(socket.reader, header); err != nil {
			return nil, err
		}
		final, opcode, masked := header[0]&webSocketFinalBit != 0, header[0]&webSocketOpcodeMask, header[1]&webSocketFinalBit != 0
		length := uint64(header[1] & webSocketLengthMask)
		switch length {
		case webSocketLength16:
			var extended uint16
			if err := binary.Read(socket.reader, binary.BigEndian, &extended); err != nil {
				return nil, err
			}
			length = uint64(extended)
		case webSocketLength64:
			if err := binary.Read(socket.reader, binary.BigEndian, &length); err != nil {
				return nil, err
			}
		}
		var mask []byte
		if masked {
			mask = make([]byte, webSocketMaskBytes)
			if _, err := io.ReadFull(socket.reader, mask); err != nil {
				return nil, err
			}
		}
		if length > uint64(^uint(0)>>1) {
			return nil, errors.New("webui lane: DevTools frame exceeds host bounds")
		}
		payload := make([]byte, int(length))
		if _, err := io.ReadFull(socket.reader, payload); err != nil {
			return nil, err
		}
		for index := range payload {
			if masked {
				payload[index] ^= mask[index%len(mask)]
			}
		}
		switch opcode {
		case webSocketClose:
			return nil, io.EOF
		case webSocketPing:
			if err := socket.writeControl(webSocketPong, payload); err != nil {
				return nil, err
			}
			continue
		case webSocketContinuation, webSocketText:
			assembled = append(assembled, payload...)
			if final {
				return assembled, nil
			}
		}
	}
}

func (socket *webSocket) writeControl(opcode byte, payload []byte) error {
	if len(payload) >= webSocketLength16 {
		return errors.New("webui lane: invalid control frame")
	}
	mask := make([]byte, webSocketMaskBytes)
	if _, err := rand.Read(mask); err != nil {
		return err
	}
	frame := []byte{webSocketFinalBit | opcode, webSocketFinalBit | byte(len(payload))}
	frame = append(frame, mask...)
	for index, value := range payload {
		frame = append(frame, value^mask[index%len(mask)])
	}
	_, err := socket.connection.Write(frame)
	return err
}
