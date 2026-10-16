// Builds the ΔContext-R presentation (paper/presentation/DeltaContext-R_presentation.pptx).
// Numbers come from paper/results/memory_summary.json and memory_paper_stats.json (test set).
//
//   NODE_PATH=<dir with pptxgenjs> node paper/presentation/build_deck.js <path to apply_theme.js>
"use strict";
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = path.join(__dirname, "DeltaContext-R_presentation.pptx");
const FIG = path.join(__dirname, "..", "figures");
const THEME = {
  name: "DeltaContext",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F3B", lt1: "FFFFFF", dk2: "3A4060", lt2: "F2F4F8",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "E87BA4", accent6: "6B7280", hlink: "2A78D6", folHlink: "6B7280",
  },
};
// Entity colours shared with the paper's figures (hex: chart options are hex-only).
const HEX = { ours: "2A78D6", secom: "EB6834", turn: "1BAF7A", exch: "EDA100", ctrl: "E87BA4",
  gray: "8C8C8C", grid: "E3E6ED", label: "3A4060", border: "D5D9E2", tint: "EAF2FC" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "ΔContext-R: Budget-Aware Retrieval over an Online Topic Store";
pres.author = "Pravin Raj, Narendra R";
pres.company = "School of Artificial Intelligence, Amrita Vishwa Vidyapeetham";
const C = pres.SchemeColor;
const W = 13.333;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.7, w: 11.7, h: 1.3,
      fontSize: 54, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.1, w: 11.0, h: 1.1,
      fontSize: 24, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 2.4, y: 2.9, w: 10.0, h: 1.0,
      fontSize: 40, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 2.4, y: 3.9, w: 10.0, h: 0.8,
      fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85,
      fontSize: 30, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "ΔContext-R  |  School of Artificial Intelligence, Amrita Vishwa Vidyapeetham",
      options: { x: 0.6, y: 7.02, w: 9.0, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: "6B7280", align: "right" },
});

// ---------------------------------------------------------------- helpers
let sectionName = "Opening";
function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sectionName });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function section(num, title, sub, notes) {
  sectionName = title;
  pres.addSection({ title });
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: title });
  s.addShape(pres.shapes.OVAL, { x: 0.8, y: 2.9, w: 1.2, h: 1.2, fill: { color: C.accent1 },
    line: { color: C.accent1 }, objectName: "section-badge" });
  s.addText(String(num), { x: 0.8, y: 2.9, w: 1.2, h: 1.2, fontSize: 36, bold: true, align: "center",
    valign: "middle", color: C.background1, margin: 0, isTextBox: true, fontFace: THEME.headFontFace });
  s.addText(title, { placeholder: "title" });
  s.addText(sub, { placeholder: "body" });
  if (notes) s.addNotes(notes);
  return s;
}
// A number in a coloured circle: the deck's recurring motif.
function badge(s, x, y, label, color, size = 0.55, fontSize = 18) {
  s.addShape(pres.shapes.OVAL, { x, y, w: size, h: size, fill: { color }, line: { color },
    objectName: `badge-${label}` });
  s.addText(String(label), { x, y, w: size, h: size, fontSize, bold: true, align: "center",
    valign: "middle", color: C.background1, margin: 0, isTextBox: true });
}
function card(s, x, y, w, h, fill = C.background2, name = "card") {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: fill }, objectName: name });
}
function txt(s, text, opts) {
  s.addText(text, { isTextBox: true, margin: 0, color: C.text1, fontSize: 16, valign: "top", ...opts });
}
function stat(s, x, y, w, big, small, color = C.accent1) {
  txt(s, big, { x, y, w, h: 0.75, fontSize: 40, bold: true, color, fontFace: THEME.headFontFace });
  txt(s, small, { x, y: y + 0.78, w, h: 0.75, fontSize: 15, color: C.text2 });
}
function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipV: y2 < y1,
    line: { color: "6B7280", width: 1.75, endArrowType: "triangle" } });
}
function hbar(s, { x, y, w, h, labels, values, colors, title, fmt = "0.000", max = 1 }) {
  s.addChart(pres.charts.BAR, [{ name: title, labels, values }], {
    x, y, w, h, barDir: "bar", chartColors: colors, catAxisOrientation: "maxMin",
    showTitle: true, title, titleFontSize: 14, titleColor: HEX.label, titleFontFace: "+mn-lt",
    catAxisLabelColor: HEX.label, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: max,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLineShow: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: fmt,
    dataLabelColor: HEX.label, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
    showLegend: false, barGapWidthPct: 45,
  });
}
const tableHead = (cells) => cells.map((t) => ({ text: t, options: { bold: true, color: C.background1,
  fill: { color: C.text1 }, valign: "middle" } }));
const tableOpts = (x, y, w, colW, fontSize = 14) => ({ x, y, w, colW, fontSize, color: C.text1, valign: "middle",
  border: { type: "solid", pt: 0.75, color: HEX.border }, margin: [0.05, 0.08, 0.05, 0.08] });

// ================================================================ 1. title
{
  pres.addSection({ title: "Opening" });
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sectionName });
  s.addText("ΔContext-R", { placeholder: "title" });
  s.addText("Budget-aware retrieval over an online topic store for long-term conversational memory",
    { placeholder: "body" });
  s.addShape(pres.shapes.OVAL, { x: 0.8, y: 4.75, w: 0.18, h: 0.18, fill: { color: C.accent1 },
    line: { color: C.accent1 }, objectName: "title-dot" });
  txt(s, "Pravin Raj   ·   Narendra R", { x: 1.15, y: 4.6, w: 10, h: 0.5, fontSize: 22, bold: true,
    color: C.background1 });
  txt(s, "School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, India",
    { x: 1.15, y: 5.1, w: 10, h: 0.45, fontSize: 18, color: C.background2 });
  txt(s, "Evaluated on LongMemEval-S against SeCom (ICLR 2025)", { x: 1.15, y: 6.3, w: 10, h: 0.4,
    fontSize: 14, color: "A9B4D0", italic: true });
  s.addNotes("Title. One line: we store a growing chat as topics online with no language model, and we "
    + "retrieve from that store so that a small token budget carries as much evidence as possible.");
}

// ================================================================ 2. problem
{
  const s = content("A chat assistant's memory outgrows its context window",
    "Every LongMemEval question comes with roughly 115 thousand tokens of past chat. Long-context models "
    + "lose about 30% accuracy on such histories, so practical systems pick a few past turns for each "
    + "question. We give the reader 2,048 tokens, about 2% of the history. The question for the rest of "
    + "the talk: which turns should go into that small window?");
  stat(s, 0.6, 1.55, 5.2, "≈115K tokens", "of chat history behind each LongMemEval question (≈490 turns, ≈48 sessions)");
  stat(s, 0.6, 3.25, 5.2, "≈30% drop", "in accuracy for long-context LLMs on such histories (LongMemEval, ICLR 2025)", C.accent2);
  stat(s, 0.6, 4.95, 5.2, "2,048 tokens", "is the context budget the reader gets here: about 2% of the history", C.accent3);
  // diagram: history stack -> select -> small context -> LLM
  for (let i = 0; i < 6; i++) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.4 + i * 0.07, y: 1.75 + i * 0.6, w: 2.3, h: 0.5,
      rectRadius: 0.05, fill: { color: C.background2 }, line: { color: HEX.border }, objectName: `session-${i}` });
    txt(s, i === 5 ? "Session 48" : (i === 4 ? "…" : `Session ${i + 1}`), { x: 6.5 + i * 0.07,
      y: 1.75 + i * 0.6, w: 2.1, h: 0.5, fontSize: 14, valign: "middle", color: C.text2 });
  }
  txt(s, "Chat history (grows every day)", { x: 6.3, y: 5.45, w: 3.0, h: 0.4, fontSize: 13, italic: true, color: C.accent6 });
  arrow(s, 9.05, 3.25, 9.85, 3.25);
  txt(s, "select", { x: 9.0, y: 2.8, w: 0.9, h: 0.35, fontSize: 13, align: "center", color: C.accent6 });
  card(s, 9.95, 2.75, 1.45, 1.0, HEX.tint, "context-box");
  txt(s, "Context\n2K tokens", { x: 9.95, y: 2.75, w: 1.45, h: 1.0, fontSize: 14, bold: true, align: "center",
    valign: "middle", color: C.accent1 });
  arrow(s, 11.45, 3.25, 11.95, 3.25);
  s.addShape(pres.shapes.OVAL, { x: 12.0, y: 2.8, w: 0.9, h: 0.9, fill: { color: C.text1 },
    line: { color: C.text1 }, objectName: "llm" });
  txt(s, "LLM", { x: 12.0, y: 2.8, w: 0.9, h: 0.9, fontSize: 15, bold: true, align: "center",
    valign: "middle", color: C.background1 });
  card(s, 6.3, 6.0, 6.6, 0.7, C.background2, "question-box");
  txt(s, "Which turns should the model see?", { x: 6.5, y: 6.0, w: 6.3, h: 0.7, fontSize: 18, bold: true,
    valign: "middle", color: C.text1 });
}

// ================================================================ 3. two questions
{
  const s = content("Two questions, two answers",
    "The work splits into storage and retrieval. Storage: build topic units online, without an LLM and "
    + "without ever moving a boundary. Retrieval: decide, under a fixed budget, which turns of which topics "
    + "to read. The baseline is SeCom from ICLR 2025, the closest published system: it segments with GPT-4, "
    + "compresses with LLMLingua-2 and retrieves whole segments.");
  const items = [
    [C.accent1, "1", "How do we store a growing chat?", "ΔContext: an online topic segmenter. Each boundary is decided a few turns late and never revised. No LLM, no training."],
    [C.accent3, "2", "What do we read under a fixed budget?", "ΔContext-R: rank turns with lexical, dense and topic evidence, then spend the budget on the user's own statements inside the most relevant topics."],
  ];
  items.forEach(([col, n, head, body], i) => {
    const x = 0.6 + i * 6.2;
    card(s, x, 1.55, 5.9, 3.35, C.background2, `q-card-${n}`);
    badge(s, x + 0.35, 1.85, n, col, 0.7, 24);
    txt(s, head, { x: x + 1.25, y: 1.85, w: 4.4, h: 0.75, fontSize: 22, bold: true, valign: "middle",
      fontFace: THEME.headFontFace });
    txt(s, body, { x: x + 0.35, y: 2.8, w: 5.2, h: 1.95, fontSize: 20, color: C.text2 });
  });
  card(s, 0.6, 5.25, 12.1, 1.3, HEX.tint, "baseline-box");
  txt(s, [{ text: "Baseline to beat: ", options: { bold: true, color: C.accent2 } },
    { text: "SeCom (ICLR 2025, Microsoft) — LLM-segmented topics, LLMLingua-2 denoising, whole-segment retrieval. It beat turn- and session-level memory in its paper." }],
    { x: 0.9, y: 5.25, w: 11.6, h: 1.3, fontSize: 18, valign: "middle" });
}

// ================================================================ 4. contributions
{
  const s = content("Contributions",
    "Four contributions. One and two are the method. Three is the controlled comparison: every system shares "
    + "the budget, the prompt and the reader, and every setting was chosen on dev questions and tested once. "
    + "Four is the honest part: we test what actually causes the gain.");
  const rows = [
    ["Online topic store", "Delayed-commitment segmentation: accurate on long chats, zero boundary revisions, no LLM calls."],
    ["Budget-aware retrieval (ΔContext-R)", "Fuses BM25, MiniLM and topic ranks; reads user turns in the top topics; reserves 25% of the budget for them."],
    ["Controlled re-implementation of SeCom", "Plus turn-, exchange- and session-level memory, all under one budget and one reader, tuned on dev, tested once."],
    ["An honest ablation", "The gain comes from budget-aware reading of user turns; online topics match true sessions and beat fixed chunks."],
  ];
  rows.forEach(([head, body], i) => {
    const y = 1.55 + i * 1.3;
    badge(s, 0.7, y + 0.1, i + 1, [C.accent1, C.accent1, C.accent2, C.accent3][i], 0.7, 24);
    txt(s, head, { x: 1.7, y, w: 10.8, h: 0.5, fontSize: 21, bold: true, fontFace: THEME.headFontFace });
    txt(s, body, { x: 1.7, y: y + 0.5, w: 10.8, h: 0.6, fontSize: 17, color: C.text2 });
  });
}

// ================================================================ 5-7. literature review
section(1, "Literature review", "Ten recent works and the gap they leave",
  "Ten works from 2023 to 2026: five memory systems, then segmentation, compression and chunking work.");
const LIT1 = [
  ["MemoryBank", "2024", "Memory store with summaries, forgetting curve, dense retrieval", "Persistent, user-adaptive memory", "Flat units; ignores the context budget"],
  ["MemGPT", "2023", "LLM pages information between prompt and storage", "Unbounded virtual context", "An LLM call for every memory operation"],
  ["Recursive summaries", "2025", "LLM rewrites memory from old memory and new turns", "Compact memory", "Lossy: verbatim evidence is lost"],
  ["LongMemEval", "2025", "Round-level values, fact-augmented keys, time-aware queries", "Systematic memory study", "Key expansion needs an LLM per round"],
  ["SeCom", "2025", "LLM segmentation + LLMLingua-2 denoising + segment retrieval", "Beats turn and session memory", "LLM call per session; whole segments fill small budgets"],
];
const LIT2 = [
  ["LLMLingua-2", "2024", "Token classifier keeps the informative tokens", "Fast, faithful compression", "A 560M-parameter pass over all stored text"],
  ["UMLF", "2025", "Joint discourse parsing and topic segmentation", "Best unsupervised segmentation", "Offline; several large encoders"],
  ["DialSTART", "2023", "Topic-aware contrastive utterance learning", "Strong unsupervised accuracy", "Needs in-domain training; offline"],
  ["Topic-aware comprehension", "2024", "Unsupervised topic blocks + dual-attention matching", "Topics help response selection", "Segments finished passages; no budget"],
  ["Chunking for RAG", "2026", "Fixed, recursive, semantic and LLM chunking compared", "Chunking changes RAG quality", "Needs document structure that chats lack"],
];
function litSlide(title, rows, cite, notes) {
  const s = content(title, notes);
  const body = rows.map((r) => r.map((t, j) => ({ text: t, options: j === 0 ? { bold: true } : {} })));
  const highlight = rows.findIndex((r) => r[0] === "SeCom");
  if (highlight >= 0) body[highlight].forEach((c) => { c.options.fill = { color: HEX.tint }; });
  s.addTable([tableHead(["Work", "Year", "Method", "Advantage", "Drawback for our setting"]), ...body],
    { ...tableOpts(0.6, 1.5, 12.1, [2.2, 0.75, 3.55, 2.55, 3.05], 14), rowH: [0.5, 0.86, 0.86, 0.86, 0.86, 0.86] });
  txt(s, cite, { x: 0.6, y: 6.5, w: 12.1, h: 0.35, fontSize: 11, color: C.accent6, italic: true });
  return s;
}
litSlide("Literature review (1/2): memory systems", LIT1,
  "Zhong et al., AAAI 2024 · Packer et al., 2023 · Wang et al., Neurocomputing 2025 · Wu et al., ICLR 2025 · Pan et al., ICLR 2025",
  "Memory systems. MemoryBank and MemGPT manage memory but ignore how much the budget can hold. Recursive "
  + "summaries lose verbatim evidence. LongMemEval studies memory design and needs an LLM to expand keys. "
  + "SeCom, highlighted, is our baseline: LLM topics, compression, whole-segment retrieval.");
litSlide("Literature review (2/2): segmentation, compression, chunking", LIT2,
  "Pan et al., Findings ACL 2024 · Xu et al., IEEE/ACM TASLP 2025 · Gao et al., SIGIR 2023 · Ma et al., Neurocomputing 2024 · Lavarec & Du, IJ-AI 2026",
  "The building blocks. LLMLingua-2 is SeCom's compressor. UMLF and DialSTART are the best unsupervised "
  + "dialogue segmenters but run offline on finished dialogues. Chunking studies show the unit choice "
  + "matters, but they rely on document structure that chat logs do not have.");
{
  const s = content("Identified gap",
    "Three gaps. Building topic memory needs an LLM call per session. Retrieving whole topics wastes a small "
    + "budget, because one LongMemEval session is larger than the whole 2K window. And published topic-memory "
    + "gains are not checked against topic-free controls, so nobody knows what actually drives them.");
  const gaps = [
    ["Building topics needs an LLM", "SeCom segments every session with GPT-4: 19,195 calls for LongMemEval-S, plus a ~2 GPU-hour compression pass."],
    ["Whole units waste small budgets", "A LongMemEval session averages ~2,300 tokens, more than a 2,048-token budget can hold."],
    ["No one tests what drives the gain", "Topic-memory gains are reported without topic-free controls, so their cause is unknown."],
  ];
  gaps.forEach(([head, body], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.6, 3.8, 4.6, C.background2, `gap-${i}`);
    badge(s, x + 0.35, 1.9, i + 1, C.accent2, 0.7, 24);
    txt(s, head, { x: x + 0.35, y: 2.85, w: 3.1, h: 1.0, fontSize: 21, bold: true, fontFace: THEME.headFontFace });
    txt(s, body, { x: x + 0.35, y: 3.95, w: 3.1, h: 2.1, fontSize: 17, color: C.text2 });
  });
  txt(s, "ΔContext-R targets all three: LLM-free topics, budget-aware reading, and controlled ablations.",
    { x: 0.6, y: 6.45, w: 12.1, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
}

// ================================================================ 8-11. proposed work
section(2, "Proposed work", "ΔContext storage + ΔContext-R retrieval",
  "Now the method: storage first, then retrieval.");
{
  const s = content("ΔContext-R in four phases",
    "Two stages. Storage runs once per incoming turn: encode the turn, decide topic boundaries with delayed "
    + "commitment, append to the topic store. Retrieval runs once per question: rank turns and topics, fuse "
    + "the ranks and boost the user's turns in the top topics, then fill the budget with a reserve for those "
    + "user turns. No language model is called anywhere before the reader.");
  txt(s, "STORAGE  ·  once per incoming turn", { x: 0.6, y: 1.45, w: 6, h: 0.35, fontSize: 14, bold: true, color: C.accent1 });
  txt(s, "RETRIEVAL  ·  once per question", { x: 0.6, y: 4.05, w: 6, h: 0.35, fontSize: 14, bold: true, color: C.accent3 });
  const top = [["1", "Encode the turn", "MiniLM vector, computed once and cached"],
    ["2", "Delayed-commitment topics", "Decide each boundary L+1 turns late; never revise it"],
    ["3", "Topic store", "Turns, roles, token costs, topic text, centroids, BM25 index"]];
  top.forEach(([n, head, body], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.85, 3.75, 1.75, C.background2, `store-${n}`);
    badge(s, x + 0.25, 2.05, n, C.accent1, 0.6, 20);
    txt(s, head, { x: x + 1.0, y: 2.0, w: 2.6, h: 0.7, fontSize: 18, bold: true, valign: "middle" });
    txt(s, body, { x: x + 0.25, y: 2.75, w: 3.3, h: 0.8, fontSize: 15, color: C.text2 });
    if (i < 2) arrow(s, x + 3.78, 2.72, x + 4.12, 2.72);
  });
  const bottom = [["4a", "Rank", "Turns by BM25 + MiniLM (RRF); topics by BM25 + centroid"],
    ["4b", "Fuse + read the user", "Turn rank + topic rank; boost user turns in top-3 topics"],
    ["4c", "Fill the budget", "75% by score, 25% for user turns of topics already read"]];
  bottom.forEach(([n, head, body], i) => {
    const x = 0.6 + i * 3.6;
    card(s, x, 4.45, 3.25, 1.95, HEX.tint, `retr-${n}`);
    badge(s, x + 0.25, 4.65, n, C.accent3, 0.6, 16);
    txt(s, head, { x: x + 1.0, y: 4.6, w: 2.15, h: 0.7, fontSize: 18, bold: true, valign: "middle" });
    txt(s, body, { x: x + 0.25, y: 5.35, w: 2.85, h: 0.95, fontSize: 15, color: C.text2 });
    arrow(s, x + 3.28, 5.42, x + 3.57, 5.42);
  });
  s.addShape(pres.shapes.OVAL, { x: 11.5, y: 4.85, w: 1.15, h: 1.15, fill: { color: C.text1 },
    line: { color: C.text1 }, objectName: "reader" });
  txt(s, "LLM\nreader", { x: 11.5, y: 4.85, w: 1.15, h: 1.15, fontSize: 14, bold: true, align: "center",
    valign: "middle", color: C.background1 });
  arrow(s, 10.93, 3.62, 10.93, 4.4);
  txt(s, "read by 4a–4c at query time", { x: 11.05, y: 3.72, w: 2.1, h: 0.5, fontSize: 12, italic: true, color: C.accent6 });
}
{
  const s = content("Phase 2: decide each boundary a few turns late, never revise it",
    "How the boundaries are found. To judge the gap between turns g and g+1, we wait until L more turns have "
    + "arrived. Then we compare the last c turns before the gap with the L turns after it. A gap becomes a "
    + "boundary only if its score is a local maximum and above a threshold. Once committed it never moves, so "
    + "the store is append-only. The delay is a constant L+1 turns: 4 turns on LongMemEval.");
  const x0 = 0.9, y0 = 2.15, sq = 0.62, gap = 0.12;
  const fills = ["D9DDE6", "D9DDE6", "D9DDE6", "D9DDE6", HEX.ours, HEX.ours, HEX.ours, HEX.secom, HEX.secom, HEX.secom, "3A4060"];
  fills.forEach((f, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0 + i * (sq + gap), y: y0, w: sq, h: sq, rectRadius: 0.05,
      fill: { color: f }, line: { color: f }, objectName: `turn-${i}` });
  });
  const gx = x0 + 7 * (sq + gap) - gap / 2;
  s.addShape(pres.shapes.LINE, { x: gx, y: y0 - 0.45, w: 0.001, h: sq + 0.9, line: { color: "C0392B", width: 3 } });
  txt(s, "gap g", { x: gx - 0.6, y: y0 - 0.85, w: 1.2, h: 0.35, fontSize: 15, bold: true, align: "center", color: "C0392B" });
  const lab = [[0, 4, "open topic (from turn a)", C.accent6], [4, 3, "last c turns", C.accent1],
    [7, 3, "next L turns", C.accent2], [10, 1, "turn t", C.text2]];
  lab.forEach(([i0, n, t, col]) => txt(s, t, { x: x0 + i0 * (sq + gap) - 0.1, y: y0 + sq + 0.12,
    w: n * (sq + gap) + 0.2, h: 0.4, fontSize: 14, bold: true, align: "center", color: col }));
  card(s, 9.4, 1.5, 3.3, 2.4, C.background2, "props");
  [["No revisions", "the store is append-only"], ["Constant delay", "L + 1 = 4 turns"], ["No LLM, no training", "4 parameters"]]
    .forEach(([h, b], i) => {
      txt(s, [{ text: h + ": ", options: { bold: true, color: C.accent1 } }, { text: b, options: { color: C.text2 } }],
        { x: 9.65, y: 1.7 + i * 0.7, w: 2.9, h: 0.6, fontSize: 15, valign: "middle" });
    });
  card(s, 0.6, 3.95, 12.1, 2.45, HEX.tint, "rule-box");
  txt(s, "Score of gap g", { x: 0.9, y: 4.15, w: 5, h: 0.4, fontSize: 18, bold: true });
  txt(s, "s(g) = 1 − cos( mean of last c turns ,  mean of next L turns )", { x: 0.9, y: 4.6, w: 11.5, h: 0.5,
    fontSize: 20, color: C.accent1, fontFace: "Cambria", italic: true });
  txt(s, "Commit a boundary at g when", { x: 0.9, y: 5.25, w: 5, h: 0.4, fontSize: 18, bold: true });
  txt(s, "s(g) ≥ s(g−1),   s(g) ≥ s(g+1)   and   s(g) ≥ τ          LongMemEval setting: L = 3, c = 3, τ = 0.75 (tuned on dev)",
    { x: 0.9, y: 5.7, w: 11.6, h: 0.5, fontSize: 17, color: C.text2 });
}
{
  const s = content("Phase 4: one score per turn, three ideas",
    "Retrieval gives every turn one score with three parts. First, how well the turn itself matches the "
    + "question, from BM25 and MiniLM fused by reciprocal rank. Second, a smaller bonus if its topic matches. "
    + "Third, a bonus for the user's own turns inside the three best topics. Turns fill 75% of the budget by "
    + "this score; the last 25% takes the remaining user turns of every topic already read. All four "
    + "parameters were set on dev questions only.");
  card(s, 0.6, 1.5, 12.1, 1.1, C.text1, "formula");
  txt(s, "f(turn)  =  rank score of the turn   +   w · rank score of its topic   +   β · [user turn in a top-k topic]",
    { x: 0.9, y: 1.5, w: 11.6, h: 1.1, fontSize: 20, color: C.background1, valign: "middle", fontFace: "Cambria" });
  const parts = [[C.accent3, "1", "Rank the turn", "BM25 + MiniLM, fused by reciprocal rank (k = 60). Catches exact words and paraphrases."],
    [C.accent1, "2", "Prefer relevant topics", "Topic rank from BM25 on topic text + MiniLM centroid. Weight w = 0.25."],
    [C.accent2, "3", "Read the user", "User turns inside the k = 3 best topics get β = 0.5. Personal facts are what the user said."]];
  parts.forEach(([col, n, head, body], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 2.9, 3.8, 2.55, C.background2, `part-${n}`);
    badge(s, x + 0.3, 3.15, n, col, 0.6, 20);
    txt(s, head, { x: x + 1.05, y: 3.1, w: 2.6, h: 0.7, fontSize: 19, bold: true, valign: "middle" });
    txt(s, body, { x: x + 0.3, y: 3.95, w: 3.25, h: 1.4, fontSize: 16, color: C.text2 });
  });
  card(s, 0.6, 5.7, 12.1, 0.95, HEX.tint, "fill-box");
  txt(s, [{ text: "Budget fill: ", options: { bold: true, color: C.accent1 } },
    { text: "75% of the budget by f, then a 25% reserve for the remaining user turns of every topic already read. Context shown in time order with session dates." }],
    { x: 0.9, y: 5.7, w: 11.6, h: 0.95, fontSize: 16, valign: "middle" });
}
{
  const s = content("Why read the user? User turns are five times shorter",
    "The mechanism. A user turn averages 69 tokens; an assistant reply 379; a whole topic or session over "
    + "2,300, more than the entire budget. So ΔContext-R fits 25 turns where a plain turn ranker fits 12. "
    + "And a diagnosis on dev questions showed that two thirds of the evidence a turn ranker misses sits "
    + "inside topics it already read from: it was a user statement ranked too low.");
  hbar(s, { x: 0.5, y: 1.45, w: 7.3, h: 4.9, title: "Mean tokens per memory unit (test histories)",
    labels: ["User turn", "Assistant turn", "Exchange", "ΔContext topic", "Session"],
    values: [69, 379, 447, 2322, 2331], colors: [HEX.ours, HEX.gray, HEX.gray, HEX.gray, HEX.gray],
    fmt: "#,##0", max: 2800 });
  stat(s, 8.3, 1.6, 4.4, "25.3 vs 11.6", "turns fit in the same 2,048 tokens: ΔContext-R vs turn-level retrieval");
  stat(s, 8.3, 3.35, 4.4, "67%", "of the evidence a turn ranker misses lies inside topics it already read (dev)", C.accent2);
  stat(s, 8.3, 5.1, 4.4, "> 2K tokens", "a single topic or session: larger than the whole budget", C.accent3);
}

// ================================================================ 12-19. results and comparisons
section(3, "Results", "Storage quality, evidence recall and answer quality",
  "Results. Everything here is on the 237 answerable test questions, scored once after tuning on dev.");
{
  const s = content("Experimental setup",
    "LongMemEval-S gives every question its own long history with turn-level evidence labels. We split the "
    + "histories 250 dev and 250 test with a fixed seed; the 237 answerable test questions are scored. Every "
    + "method shares the reader, the prompt and the token counter. Settings were chosen on dev only, tested "
    + "once, and differences use a paired bootstrap with Holm correction.");
  const cards = [["Dataset", C.accent1, ["LongMemEval-S (ICLR 2025)", "500 questions, each with a ~492-turn, ~48-session history", "Turn-level evidence labels", "Seeded split: 250 dev / 250 test histories", "237 answerable test questions"]],
    ["Protocol", C.accent3, ["Reader: Qwen2.5-3B-Instruct, greedy decoding", "Budgets: 2,048 and 4,096 context tokens", "Same prompt, token counter and reader for all", "Settings chosen on dev, scored once on test", "Paired bootstrap (10,000) + Holm correction"]]];
  cards.forEach(([head, col, lines], i) => {
    const x = 0.6 + i * 6.2;
    card(s, x, 1.5, 5.9, 3.55, C.background2, `setup-${i}`);
    txt(s, head, { x: x + 0.35, y: 1.7, w: 5.2, h: 0.55, fontSize: 22, bold: true, color: col, fontFace: THEME.headFontFace });
    txt(s, lines.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < lines.length - 1 } })),
      { x: x + 0.35, y: 2.35, w: 5.3, h: 2.6, fontSize: 17, color: C.text2, paraSpaceAfter: 8 });
  });
  txt(s, "Metrics", { x: 0.6, y: 5.3, w: 4, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  const metrics = ["Evidence recall", "All-evidence rate", "Answer token F1", "Exact match"];
  metrics.forEach((m, i) => {
    const x = 0.6 + i * 3.05;
    card(s, x, 5.9, 2.85, 0.75, HEX.tint, `metric-${i}`);
    txt(s, m, { x, y: 5.9, w: 2.85, h: 0.75, fontSize: 16, bold: true, align: "center", valign: "middle", color: C.accent1 });
  });
}
{
  const s = content("Storage: online topics are accurate and never revised",
    "Storage quality. Against true session boundaries on 250 test histories, ΔContext has the lowest error "
    + "and the best exact boundary F1, and it is the only method that never changes a boundary once made. "
    + "TextTiling and semantic chunking must be re-run as the chat grows and move about 17 boundaries per "
    + "100 new turns. SeCom needs a GPT-4 call per session.");
  const rows = [["ΔContext (ours)", "yes", "0.058", "0.914", "0", "0"],
    ["TextTiling", "no", "0.069", "0.894", "17.0", "0"],
    ["Semantic chunking", "no", "0.141", "0.756", "17.6", "0"],
    ["SeCom (GPT-4)", "no", "–", "–", "–", "1 per session"]];
  const body = rows.map((r, i) => r.map((t) => ({ text: t, options: i === 0 ? { bold: true, fill: { color: HEX.tint } } : {} })));
  s.addTable([tableHead(["Segmenter", "Online", "Pk ↓", "Boundary F1 ↑", "Revisions / 100 turns", "LLM calls"]), ...body],
    { ...tableOpts(0.6, 1.55, 8.4, [2.6, 0.9, 0.9, 1.3, 1.4, 1.3], 15), rowH: 0.62 });
  txt(s, "250 LongMemEval-S test histories; reference = session boundaries; parameters tuned on dev.",
    { x: 0.6, y: 4.75, w: 8.4, h: 0.4, fontSize: 12, italic: true, color: C.accent6 });
  card(s, 9.4, 1.55, 3.3, 4.6, C.text1, "rev-card");
  txt(s, "0", { x: 9.4, y: 1.85, w: 3.3, h: 1.3, fontSize: 80, bold: true, align: "center", color: C.background1, fontFace: THEME.headFontFace });
  txt(s, "boundary revisions per 100 new turns", { x: 9.6, y: 3.2, w: 2.9, h: 0.8, fontSize: 16, align: "center", color: C.background2 });
  txt(s, "vs 17 for the offline segmenters", { x: 9.6, y: 4.3, w: 2.9, h: 0.9, fontSize: 18, bold: true, align: "center", color: "F5A97F" });
  txt(s, "Every revision forces the store to re-index topics: ΔContext never does.",
    { x: 0.6, y: 5.5, w: 8.4, h: 0.8, fontSize: 18, bold: true, color: C.accent1 });
}
const RECALL_LABELS = ["SeCom (MPNet)", "SeCom (faithful)", "Turn-level dense", "Exchange-level", "Turn-level hybrid", "ΔContext-R (ours)"];
const RECALL_COLORS = [HEX.secom, HEX.secom, HEX.gray, HEX.exch, HEX.turn, HEX.ours];
{
  const s = content("Evidence recall: ΔContext-R retrieves the most evidence",
    "Evidence recall: the share of labelled evidence turns that reach the context. At 2,048 tokens "
    + "ΔContext-R reaches 0.908 against 0.641 for faithful SeCom. At 4,096 it reaches 0.969, significantly "
    + "above the strongest baseline, turn-level hybrid retrieval at 0.912.");
  hbar(s, { x: 0.5, y: 1.45, w: 6.1, h: 4.4, title: "2,048-token budget", labels: RECALL_LABELS,
    values: [0.610, 0.641, 0.773, 0.814, 0.859, 0.908], colors: RECALL_COLORS, max: 1.1 });
  hbar(s, { x: 6.75, y: 1.45, w: 6.1, h: 4.4, title: "4,096-token budget", labels: RECALL_LABELS,
    values: [0.741, 0.782, 0.880, 0.891, 0.912, 0.969], colors: RECALL_COLORS, max: 1.1 });
  card(s, 0.6, 6.0, 12.1, 0.75, HEX.tint, "recall-takeaway");
  txt(s, [{ text: "+0.27 ", options: { bold: true, color: C.accent1 } }, { text: "recall over SeCom at 2K (p < 0.001)      " },
    { text: "+0.057 ", options: { bold: true, color: C.accent1 } }, { text: "over turn-level hybrid at 4K (p < 0.001)" }],
    { x: 0.9, y: 6.0, w: 11.6, h: 0.75, fontSize: 18, valign: "middle" });
}
{
  const s = content("Answer quality: best F1 at 2K, smallest drop at 4K",
    "Answer token F1 with the 3B reader. At 2,048 tokens ΔContext-R has the best F1, 0.369, against 0.270 "
    + "for SeCom: a 37% relative gain. A surprise: with 4,096 tokens every system answers worse, because a "
    + "3B reader handles longer contexts poorly. ΔContext-R drops least, so at 4K it beats turn-level "
    + "hybrid retrieval significantly, 0.310 vs 0.258.");
  hbar(s, { x: 0.5, y: 1.45, w: 6.6, h: 4.4, title: "Answer token F1, 2,048 tokens", labels: RECALL_LABELS,
    values: [0.253, 0.270, 0.339, 0.329, 0.351, 0.369], colors: RECALL_COLORS, max: 0.45 });
  s.addChart(pres.charts.BAR, [
    { name: "2,048 tokens", labels: ["SeCom", "Turn-level hybrid", "ΔContext-R"], values: [0.270, 0.351, 0.369] },
    { name: "4,096 tokens", labels: ["SeCom", "Turn-level hybrid", "ΔContext-R"], values: [0.260, 0.258, 0.310] },
  ], {
    x: 7.3, y: 1.45, w: 5.5, h: 4.4, barDir: "col", chartColors: ["9CC3EE", HEX.ours],
    showTitle: true, title: "F1 at both budgets", titleFontSize: 14, titleColor: HEX.label, titleFontFace: "+mn-lt",
    catAxisLabelColor: HEX.label, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 0.45, valGridLine: { style: "none" },
    catGridLine: { style: "none" }, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    dataLabelColor: HEX.label, dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.label,
    barGapWidthPct: 60,
  });
  card(s, 0.6, 6.0, 12.1, 0.75, HEX.tint, "qa-takeaway");
  txt(s, [{ text: "+0.099 F1 ", options: { bold: true, color: C.accent1 } }, { text: "over SeCom at 2K (+37%, p < 0.001)      " },
    { text: "+0.052 F1 ", options: { bold: true, color: C.accent1 } }, { text: "over turn-level hybrid at 4K (p = 0.02)" }],
    { x: 0.9, y: 6.0, w: 11.6, h: 0.75, fontSize: 18, valign: "middle" });
}

section(4, "Comparisons", "Against SeCom and turn-level retrieval, and what drives the gain",
  "Comparisons: head to head with SeCom, why segment memory loses, which questions gain, and the ablation.");
{
  const s = content("Head-to-head with SeCom (ICLR 2025)",
    "The full comparison on the test set. SeCom is re-implemented from its code: segment units, LLMLingua-2 "
    + "keys, BM25 or MPNet, whole units filling the budget. Its GPT-4 segmentation needs 19 thousand API "
    + "calls, so it gets ΔContext's topics cut at session starts instead. Giving SeCom our stronger retriever "
    + "barely helps. ΔContext-R wins every column and builds its memory with no LLM call.");
  const rows = [
    ["SeCom (faithful)", "segment · LLMLingua-2 · BM25", "0.641", "0.270", "0.782", "0.260", "1 per session"],
    ["SeCom (MPNet)", "segment · LLMLingua-2 · MPNet", "0.610", "0.253", "0.741", "–", "1 per session"],
    ["SeCom + our retriever", "segment · LLMLingua-2 · BM25+MiniLM", "0.669", "0.267", "0.819", "–", "1 per session"],
    ["Round-level (LongMemEval)", "exchange · raw · BM25+MiniLM", "0.814", "0.329", "0.891", "–", "0"],
    ["Turn-level hybrid", "turn · raw · BM25+MiniLM", "0.859", "0.351", "0.912", "0.258", "0"],
    ["ΔContext-R (ours)", "turn+topic · raw · BM25+MiniLM", "0.908", "0.369", "0.969", "0.310", "0"],
  ];
  const body = rows.map((r, i) => r.map((t, j) => ({ text: t, options: {
    ...(i === rows.length - 1 ? { bold: true, fill: { color: HEX.tint }, color: C.accent1 } : {}),
    ...(j >= 2 ? { align: "center" } : {}) } })));
  s.addTable([tableHead(["Memory", "Unit · keys · retriever", "Recall 2K", "F1 2K", "Recall 4K", "F1 4K", "LLM calls to build"])
    .map((c, j) => (j >= 2 ? { ...c, options: { ...c.options, align: "center" } } : c)), ...body],
    { ...tableOpts(0.6, 1.5, 12.1, [2.65, 3.6, 1.05, 0.9, 1.05, 0.9, 1.95], 15), rowH: 0.62 });
  txt(s, "237 answerable LongMemEval-S test questions; Qwen2.5-3B reader; – = not run. Significance tests in the paper.",
    { x: 0.6, y: 6.05, w: 12.1, h: 0.4, fontSize: 12, italic: true, color: C.accent6 });
}
{
  const s = content("Why segment memory loses: units outgrow the budget",
    "Why SeCom loses here. Each cell is dev recall when whole units are retrieved. Turns and exchanges reach "
    + "0.66 to 0.82; topics and sessions only 0.48 to 0.60, with every retriever. LLMLingua-2 denoising helps "
    + "dense retrievers, as SeCom reports, but does not close the gap. Even the finest units, exchanges, "
    + "reach only 0.814 on test, below ΔContext-R's 0.908.");
  s.addImage({ path: path.join(FIG, "memory_grid.png"), x: 0.5, y: 1.55, w: 8.2, h: 2.46,
    altText: "Heatmap of dev evidence recall by memory unit and retriever, raw versus LLMLingua-2 keys" });
  txt(s, "Dev evidence recall at 2,048 tokens when whole units are retrieved: (a) raw keys, (b) SeCom's LLMLingua-2 keys.",
    { x: 0.6, y: 4.1, w: 8.0, h: 0.6, fontSize: 12, italic: true, color: C.accent6 });
  const points = [["Recall falls as units grow", "with every retriever: turns 0.66–0.82, topics and sessions 0.48–0.60"],
    ["Sessions barely fit 2K", "~2,300 tokens each, mostly assistant text: one or two fit the budget"],
    ["Compression helps MPNet", "turns 0.67 → 0.75 with LLMLingua-2, but the gap remains"],
    ["Finer segments won't save it", "exchanges, the finest units, reach 0.814 on test vs 0.908"]];
  points.forEach(([h, b], i) => {
    const y = 1.55 + i * 1.25;
    badge(s, 9.0, y + 0.05, i + 1, C.accent2, 0.5, 16);
    txt(s, h, { x: 9.65, y, w: 3.15, h: 0.4, fontSize: 16, bold: true });
    txt(s, b, { x: 9.65, y: y + 0.42, w: 3.15, h: 0.8, fontSize: 14, color: C.text2 });
  });
  card(s, 0.6, 5.0, 8.0, 1.6, HEX.tint, "grid-takeaway");
  txt(s, "Under a small budget, the unit you retrieve matters more than how well you rank it.",
    { x: 0.9, y: 5.0, w: 7.4, h: 1.6, fontSize: 20, bold: true, valign: "middle", color: C.accent1 });
}
{
  const s = content("Gains by question type: multi-session and preference",
    "Recall by question type at 2K tokens. The big gains are on multi-session questions, where evidence is "
    + "spread over several sessions, and on preference questions. The price: when the evidence is an "
    + "assistant reply, reading the user first costs recall, 0.815 against 0.889.");
  const types = ["Single-session user", "Single-session assistant", "Preference", "Multi-session", "Knowledge update", "Temporal"];
  s.addChart(pres.charts.BAR, [
    { name: "SeCom (faithful)", labels: types, values: [0.938, 1.000, 0.567, 0.387, 0.655, 0.567] },
    { name: "Turn-level hybrid", labels: types, values: [0.922, 0.889, 0.678, 0.732, 0.977, 0.886] },
    { name: "ΔContext-R (ours)", labels: types, values: [0.984, 0.815, 1.000, 0.869, 0.981, 0.872] },
  ], {
    x: 0.5, y: 1.45, w: 8.7, h: 5.2, barDir: "col", chartColors: [HEX.secom, HEX.turn, HEX.ours],
    catAxisLabelColor: HEX.label, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelColor: HEX.label, valAxisLabelFontSize: 11,
    valAxisLabelFontFace: "+mn-lt", valAxisLabelFormatCode: "0.0", valGridLine: { color: HEX.grid, size: 0.75 },
    catGridLine: { style: "none" }, showLegend: true, legendPos: "t", legendFontSize: 13,
    legendFontFace: "+mn-lt", legendColor: HEX.label, barGapWidthPct: 70,
    showTitle: true, title: "Evidence recall by question type (2,048 tokens, test)", titleFontSize: 14,
    titleColor: HEX.label, titleFontFace: "+mn-lt",
  });
  stat(s, 9.6, 1.6, 3.2, "0.869", "multi-session recall (turn-level 0.732, SeCom 0.387)");
  stat(s, 9.6, 3.3, 3.2, "1.000", "preference recall (turn-level 0.678, SeCom 0.567)", C.accent1);
  stat(s, 9.6, 5.0, 3.2, "0.815", "assistant-evidence recall: the cost of reading the user (turn-level 0.889)", C.text2);
}
{
  const s = content("Honest ablation: what really drives the gain",
    "The honest part. We replaced one ingredient at a time, each control tuned on dev with its own grid. "
    + "Swapping our online topics for the true sessions changes nothing, and fixed 10-turn chunks are worse: "
    + "so the online topics are as good as the ground truth. But two rules with no topics at all, a global "
    + "bonus for user turns and cost-aware filling, match ΔContext-R. The gain is budget-aware reading of "
    + "user turns, not topic structure itself.");
  const rows = [["True sessions as topics", "−0.004", "+0.007"],
    ["10-turn chunks as topics", "+0.026", "+0.033 *"],
    ["Global user-turn prior (no topics)", "−0.014", "+0.017"],
    ["Cost-aware fill (no topics, no roles)", "−0.015", "+0.015"],
    ["Topic fusion without reading the user", "+0.045", "+0.063 *"]];
  const body = rows.map((r) => r.map((t, j) => ({ text: t, options: j > 0 ? { align: "center" } : {} })));
  s.addTable([tableHead(["Control (tuned on dev)", "Δ recall 2K", "Δ recall 4K"])
    .map((c, j) => (j > 0 ? { ...c, options: { ...c.options, align: "center" } } : c)), ...body],
    { ...tableOpts(0.6, 1.5, 7.6, [4.4, 1.6, 1.6], 15), rowH: 0.62 });
  txt(s, "Δ = ΔContext-R minus control, evidence recall on test. * significant (p < 0.001, Holm); others not significant.",
    { x: 0.6, y: 5.35, w: 7.6, h: 0.6, fontSize: 12, italic: true, color: C.accent6 });
  const take = [[C.accent1, "Budget-aware reading of user turns drives the gain"],
    [C.accent3, "Online topics are as good as true sessions, better than fixed chunks"],
    [C.accent2, "Topic structure alone adds no recall on LongMemEval"]];
  take.forEach(([col, t], i) => {
    const y = 1.5 + i * 1.55;
    card(s, 8.6, y, 4.1, 1.3, C.background2, `take-${i}`);
    badge(s, 8.85, y + 0.38, i + 1, col, 0.55, 18);
    txt(s, t, { x: 9.6, y, w: 2.95, h: 1.3, fontSize: 16, bold: true, valign: "middle" });
  });
}
{
  const s = content("Limitations",
    "Four limitations. One benchmark, with mostly single-topic sessions and evidence that is mostly user "
    + "statements. A 3B reader scored by token F1 rather than an LLM judge. SeCom's GPT-4 segmentation "
    + "replaced by an LLM-free equivalent, though the exchange-level result bounds finer segments. And the "
    + "user prior costs recall when the evidence is an assistant reply.");
  const lim = [["One benchmark", "LongMemEval sessions are mostly single-topic and its evidence is mostly user statements."],
    ["Small reader, lexical metric", "Qwen2.5-3B scored by token F1, not an LLM judge; it answers worse with 4K tokens."],
    ["SeCom without GPT-4 segments", "Replaced by an LLM-free equivalent; exchange-level retrieval bounds finer segments."],
    ["The user prior has a cost", "Questions answered by an assistant reply lose recall (0.815 vs 0.889)."]];
  lim.forEach(([h, b], i) => {
    const x = 0.6 + (i % 2) * 6.2, y = 1.6 + Math.floor(i / 2) * 2.55;
    card(s, x, y, 5.9, 2.25, C.background2, `lim-${i}`);
    badge(s, x + 0.3, y + 0.3, "!", C.accent4, 0.6, 22);
    txt(s, h, { x: x + 1.1, y: y + 0.25, w: 4.6, h: 0.7, fontSize: 20, bold: true, valign: "middle" });
    txt(s, b, { x: x + 0.3, y: y + 1.05, w: 5.3, h: 1.1, fontSize: 16, color: C.text2 });
  });
}

// ================================================================ conclusion, references, thanks
section(5, "Conclusion", "What we showed and what comes next", "Wrapping up.");
{
  const s = content("Conclusion and future work",
    "In one sentence: store the chat as topics online with no LLM, then spend a small budget on the user's "
    + "statements in the topics that matter. Against SeCom that lifts recall from 0.64 to 0.91 and answer F1 "
    + "by 37%, with zero LLM calls to build the memory. Next: test on benchmarks whose sessions mix topics, "
    + "such as LoCoMo, and learn the user prior and budget split from the question.");
  stat(s, 0.6, 1.5, 3.9, "0.64 → 0.91", "evidence recall vs SeCom at 2,048 tokens");
  stat(s, 4.75, 1.5, 3.9, "+37%", "answer F1 vs SeCom (0.270 → 0.369)", C.accent3);
  stat(s, 8.9, 1.5, 3.9, "0", "LLM calls to build the memory", C.accent2);
  card(s, 0.6, 3.55, 6.0, 3.05, C.background2, "novelty");
  txt(s, "Novelty", { x: 0.9, y: 3.75, w: 5.4, h: 0.5, fontSize: 20, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  const nov = ["Delayed-commitment topics: online, final, LLM-free", "Retrieval shaped by the budget: read user turns in relevant topics", "A controlled test of what drives topic-memory gains"];
  txt(s, nov.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < nov.length - 1 } })),
    { x: 0.9, y: 4.35, w: 5.5, h: 2.1, fontSize: 17, color: C.text2, paraSpaceAfter: 8 });
  card(s, 6.75, 3.55, 5.95, 3.05, HEX.tint, "future");
  txt(s, "Future work", { x: 7.05, y: 3.75, w: 5.4, h: 0.5, fontSize: 20, bold: true, color: C.accent3, fontFace: THEME.headFontFace });
  const fut = ["Benchmarks whose sessions mix several topics (e.g. LoCoMo)", "Learn the user prior and budget split from the question", "Larger readers and LLM-judged answers"];
  txt(s, fut.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < fut.length - 1 } })),
    { x: 7.05, y: 4.35, w: 5.45, h: 2.1, fontSize: 17, color: C.text2, paraSpaceAfter: 8 });
}
{
  const s = content("Key references",
    "The main references; the paper lists all 35.");
  const refs = [
    "Z. Pan et al., On memory construction and retrieval for personalized conversational agents (SeCom), ICLR 2025.",
    "D. Wu et al., LongMemEval: Benchmarking chat assistants on long-term interactive memory, ICLR 2025.",
    "Z. Pan et al., LLMLingua-2: Data distillation for efficient and faithful prompt compression, Findings of ACL 2024.",
    "W. Zhong et al., MemoryBank: Enhancing large language models with long-term memory, AAAI 2024.",
    "C. Packer et al., MemGPT: Towards LLMs as operating systems, arXiv:2310.08560, 2023.",
    "Q. Wang et al., Recursively summarizing enables long-term dialogue memory in LLMs, Neurocomputing 2025.",
    "J. Xu et al., Unsupervised mutual learning of discourse parsing and topic segmentation in dialogue, IEEE/ACM TASLP 2025.",
    "M. A. Hearst, TextTiling: Segmenting text into multi-paragraph subtopic passages, Computational Linguistics 1997.",
    "S. Robertson and H. Zaragoza, The probabilistic relevance framework: BM25 and beyond, FnTIR 2009.",
    "G. V. Cormack et al., Reciprocal rank fusion outperforms Condorcet and individual rank learning methods, SIGIR 2009.",
    "N. F. Liu et al., Lost in the middle: How language models use long contexts, TACL 2024.",
    "Qwen Team, Qwen2.5 technical report, arXiv:2412.15115, 2024.",
  ];
  txt(s, refs.map((t, j) => ({ text: t, options: { bullet: { type: "number" }, breakLine: j < refs.length - 1 } })),
    { x: 0.6, y: 1.45, w: 12.1, h: 5.4, fontSize: 16, color: C.text2, paraSpaceAfter: 6 });
}
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sectionName });
  s.addText("Thank you", { placeholder: "title" });
  s.addText("Questions?", { placeholder: "body" });
  txt(s, "Pravin Raj  ·  pravinraj2054@gmail.com", { x: 0.8, y: 4.7, w: 11, h: 0.5, fontSize: 20, color: C.background1 });
  txt(s, "Narendra R", { x: 0.8, y: 5.2, w: 11, h: 0.5, fontSize: 20, color: C.background1 });
  txt(s, "School of Artificial Intelligence, Amrita Vishwa Vidyapeetham", { x: 0.8, y: 5.75, w: 11, h: 0.45,
    fontSize: 16, color: C.background2 });
  s.addNotes("Thank the audience and invite questions. Likely questions: why not run GPT-4 segmentation "
    + "(19K API calls; exchange-level retrieval bounds what finer segments could reach), and why F1 drops at "
    + "4K (the 3B reader handles longer contexts worse; ΔContext-R degrades least).");
}

// ---------------------------------------------------------------- write + theme
(async () => {
  await pres.writeFile({ fileName: OUT });
  const applyThemePath = process.argv[2];
  if (applyThemePath) {
    const { applyTheme } = require(applyThemePath);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
