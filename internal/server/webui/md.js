/* overgo_gui markdown: a tiny, dependency-free markdown→DOM renderer for chat
   assistant messages. It builds real DOM nodes with textContent only — never
   innerHTML — so model output can never inject markup, and link hrefs are
   scheme-checked (http/https/mailto only). Subset: fenced code blocks (with a
   copy button + language label), inline code, bold, italic, links, headings,
   bullet/numbered lists, paragraphs, and hard line breaks. */
(function () {
  "use strict";

  function copyButton(text, label) {
    label = label || "copy";
    const btn = document.createElement("button");
    btn.className = "md-copy";
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "copied";
      } catch (_) { btn.textContent = "copy failed"; }
      setTimeout(() => { btn.textContent = label; }, 1200);
    });
    return btn;
  }

  function appendText(parent, text) {
    // Render hard line breaks as <br>; everything else is a plain text node.
    text.split("\n").forEach((part, index) => {
      if (index) parent.appendChild(document.createElement("br"));
      if (part) parent.appendChild(document.createTextNode(part));
    });
  }

  function linkNode(text, href) {
    // Only well-known safe schemes become links; anything else stays literal.
    if (!/^(https?:|mailto:)/i.test(href)) {
      const span = document.createElement("span");
      span.textContent = "[" + text + "](" + href + ")";
      return span;
    }
    const a = document.createElement("a");
    a.href = href;
    a.textContent = text;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    return a;
  }

  function wrap(tag, inner) {
    const node = document.createElement(tag);
    renderInline(node, inner); // recurse so **_nested_** formatting works
    return node;
  }

  const INLINE = [
    { re: /^`([^`]+)`/, make: (m) => { const c = document.createElement("code"); c.className = "md-code"; c.textContent = m[1]; return c; } },
    { re: /^\*\*([^*]+)\*\*/, make: (m) => wrap("strong", m[1]) },
    { re: /^__([^_]+)__/, make: (m) => wrap("strong", m[1]) },
    { re: /^\*([^*]+)\*/, make: (m) => wrap("em", m[1]) },
    { re: /^_([^_]+)_/, make: (m) => wrap("em", m[1]) },
    { re: /^\[([^\]]+)\]\(([^)\s]+)\)/, make: (m) => linkNode(m[1], m[2]) },
  ];

  function renderInline(parent, text) {
    let rest = String(text == null ? "" : text);
    while (rest.length) {
      let matched = false;
      for (const pattern of INLINE) {
        const m = rest.match(pattern.re);
        if (m) { parent.appendChild(pattern.make(m)); rest = rest.slice(m[0].length); matched = true; break; }
      }
      if (matched) continue;
      // Consume plain text up to the next token-starting character.
      const next = rest.search(/[`*_[]/);
      if (next === -1) { appendText(parent, rest); break; }
      const take = next === 0 ? 1 : next;
      appendText(parent, rest.slice(0, take));
      rest = rest.slice(take);
    }
  }

  function codeBlock(code, lang) {
    const block = document.createElement("div");
    block.className = "md-codeblock";
    const bar = document.createElement("div");
    bar.className = "md-codebar";
    const label = document.createElement("span");
    label.className = "md-lang";
    label.textContent = lang || "code";
    bar.append(label, copyButton(code));
    const pre = document.createElement("pre");
    pre.className = "md-pre";
    const codeEl = document.createElement("code");
    codeEl.textContent = code;
    pre.appendChild(codeEl);
    block.append(bar, pre);
    return block;
  }

  const isListLine = (line) => /^\s*([-*]|\d+\.)\s+/.test(line);

  function md(text) {
    const root = document.createElement("div");
    root.className = "md";
    const lines = String(text == null ? "" : text).split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^```(\w*)\s*$/);
      if (fence) {
        const buf = [];
        i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++; // consume closing fence (or end of input)
        root.appendChild(codeBlock(buf.join("\n"), fence[1]));
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        const node = document.createElement("h" + Math.min(6, heading[1].length));
        node.className = "md-h";
        renderInline(node, heading[2]);
        root.appendChild(node);
        i++;
        continue;
      }
      if (isListLine(line)) {
        const ordered = /^\s*\d+\.\s+/.test(line);
        const list = document.createElement(ordered ? "ol" : "ul");
        list.className = "md-list";
        while (i < lines.length && isListLine(lines[i])) {
          const item = document.createElement("li");
          renderInline(item, lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ""));
          list.appendChild(item);
          i++;
        }
        root.appendChild(list);
        continue;
      }
      if (/^\s*$/.test(line)) { i++; continue; }
      const para = [];
      while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^```/.test(lines[i]) &&
        !/^#{1,6}\s+/.test(lines[i]) && !isListLine(lines[i])) {
        para.push(lines[i]);
        i++;
      }
      const p = document.createElement("p");
      p.className = "md-p";
      renderInline(p, para.join("\n"));
      root.appendChild(p);
    }
    return root;
  }

  window.overgo = window.overgo || {};
  window.overgo.md = md;
  window.overgo.copyButton = copyButton;
})();
