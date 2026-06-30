// ════════════════════════════════════════════════════════════════
// Sentiment Analysis — From Scratch
// Vanilla JS port of the React component. Same data + logic, plain
// DOM rendering instead of JSX. Inline onclick handlers are used in
// the generated HTML, so the relevant functions are exposed on
// window at the bottom of each section.
// ════════════════════════════════════════════════════════════════

// ─── small utils ───────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
function escapeJs(str) {
  return String(str).replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

// ─── DATA ────────────────────────────────────────────────────────────────
const BASE_LEXICON = {
  amazing:4,awesome:4,fantastic:4,excellent:4,outstanding:4,brilliant:4,superb:4,magnificent:4,wonderful:4,perfect:5,
  great:3,good:3,happy:3,love:3,best:3,beautiful:3,enjoy:3,glad:2,pleased:2,positive:2,nice:2,
  fine:1,okay:1,decent:1,acceptable:1,fair:1,alright:1,satisfactory:1,
  bad:-2,poor:-2,disappointing:-2,mediocre:-2,average:-1,lacking:-2,
  terrible:-4,horrible:-4,awful:-4,dreadful:-4,appalling:-4,atrocious:-4,disgusting:-3,
  hate:-3,worst:-4,fail:-3,failure:-3,useless:-3,pathetic:-3,broken:-3,
  slow:-2,late:-2,delayed:-2,unreliable:-2,difficult:-2,annoying:-2,frustrating:-3,
  fast:2,quick:2,efficient:2,reliable:2,helpful:2,friendly:2,professional:2,
  expensive:-1,cheap:1,affordable:2,overpriced:-2,underrated:-1,overrated:-2,
  delivery:-1,
};

const NEGATORS = new Set(["not","never","no","neither","nor","without","barely","hardly","scarcely","seldom"]);

const INTENSIFIERS = {very:1.5,really:1.4,absolutely:1.7,extremely:1.8,incredibly:1.6,truly:1.3,quite:1.2,pretty:1.2,somewhat:0.7,slightly:0.6,bit:0.8};

const SAMPLES = {
  pos: "The product is absolutely fantastic! The customer service was incredibly helpful and delivery was super fast. I love everything about it.",
  neg: "Terrible experience from start to finish. The product broke immediately, delivery was horribly late, and customer service was completely useless. I hate it.",
  mix: "The movie had a brilliant script and amazing visuals but the ending felt rushed and quite disappointing. Good overall but could have been perfect.",
};

const SIM_PIPELINES = {
  A: { name:"Lexicon-based pipeline", desc:"Walk through each stage of how a lexicon approach classifies sentiment, from raw text to final label.", steps:["Preprocess","Tokenize","Lookup","Score","Classify"] },
  B: { name:"ML (Naive Bayes) pipeline", desc:"See how a trained Naive Bayes classifier processes text through feature extraction and probability scoring.", steps:["Preprocess","Vectorize","Features","Bayes score","Classify"] },
  C: { name:"Transformer pipeline", desc:"Follow how a BERT-style transformer tokenizes, embeds, attends, and classifies sentiment end-to-end.", steps:["Tokenize","Embed","Attention","Pool","Classify"] },
};

const GRANULARITY = [
  { level:"Document-level", color:"#185FA5", bg:"#E6F1FB", bd:"#85B7EB",
    example:"\"I ordered this blender last month. Shipping was quick, setup was painless, and it blends smoothies perfectly. Couldn't be happier.\"",
    output:"→ One label for the whole text: Positive",
    note:"Treats the entire document as a single opinion. Fast and simple, but loses any nuance if the writer mentions multiple things." },
  { level:"Sentence-level", color:"#27500A", bg:"#EAF3DE", bd:"#97C459",
    example:"\"The hotel room was spotless. The breakfast was mediocre. Staff were incredibly kind.\"",
    output:"→ 3 separate labels: Positive · Negative · Positive",
    note:"Splits text into sentences first, scores each independently. Better resolution than document-level, still can't isolate which topic each sentence praises or criticizes." },
  { level:"Aspect-based (ABSA)", color:"#3C3489", bg:"#EEEDFE", bd:"#AFA9EC",
    example:"\"The camera is incredible but the battery life is disappointing and it's a bit expensive.\"",
    output:"→ camera: Positive · battery: Negative · price: Negative",
    note:"Identifies specific entities or features and attributes sentiment to each one individually. Most useful for product teams, hardest to build." },
];

const APPLICATIONS = [
  { icon:"⭐", title:"Product reviews", text:"Aggregating thousands of reviews into an at-a-glance rating, and surfacing which features people complain about most." },
  { icon:"📣", title:"Social media monitoring", text:"Tracking public mood toward a brand, product launch, or campaign in near real time across posts and comments." },
  { icon:"🎧", title:"Customer support triage", text:"Auto-flagging angry or urgent tickets so human agents respond to the most at-risk customers first." },
  { icon:"📈", title:"Market & competitor research", text:"Comparing sentiment toward your product against competitors' to spot strengths and weak points." },
  { icon:"🗳️", title:"Public opinion tracking", text:"Following sentiment shifts around policies, elections, or public figures over time." },
  { icon:"💬", title:"Employee feedback analysis", text:"Scanning anonymous survey responses to flag morale issues early, at a scale no one could read manually." },
];

const POLARITY_EXAMPLES = [
  { text:"This is the best purchase I've made all year.", pol:92, subj:85, note:"Strong opinion, clearly positive." },
  { text:"The screen is 6.1 inches and weighs 180 grams.", pol:50, subj:6, note:"Just a fact — no feeling expressed at all." },
  { text:"It's okay, does the job, nothing special.", pol:42, subj:55, note:"A mild opinion, leaning slightly negative." },
  { text:"Absolutely terrible, would not recommend to anyone.", pol:6, subj:90, note:"Strong opinion, clearly negative." },
];

const QUIZ = [
  { id:0, tag:"Basics", q:"What is the difference between polarity and subjectivity in sentiment analysis?", opts:["They are two names for the same concept","Polarity measures positive↔negative charge; subjectivity measures fact↔opinion","Polarity only applies to single words; subjectivity only applies to full documents","Subjectivity is a feature only transformer models can compute"], ans:1, exp:'Polarity asks "how positive or negative is this?" Subjectivity asks "is this a personal opinion or an objective fact?" "The phone weighs 180g" is objective with neutral polarity. "The phone is gorgeous" is subjective with positive polarity. Most sentiment systems implicitly assume the input is already subjective.' },
  { id:1, tag:"Basics", q:'A review says: "The camera is incredible but the battery life is disappointing." Why would document-level sentiment analysis struggle here?', opts:["Document-level analysis cannot process the word \"but\"","It produces a single overall label, blending two opposite opinions about two different features into one vague score","Document-level analysis only works on reviews longer than 50 words","It requires labeled training data for every possible product feature"], ans:1, exp:'Document-level sentiment collapses everything into one label — likely "Mixed" or a washed-out "slightly positive." It cannot tell you that the camera is loved and the battery is hated. That requires aspect-based sentiment analysis (ABSA), which attributes sentiment to each mentioned feature separately.' },
  { id:2, tag:"Lexicon", q:'Why does the sentence "The movie is not terrible" get misclassified by a basic lexicon model?', opts:["The word \"movie\" is not in the lexicon","It only sees \"terrible\" (negative score) without understanding \"not\" negates it","The sentence is too short for accurate analysis","Lexicon models only work on product reviews"], ans:1, exp:'Basic lexicon models sum word scores independently. "Terrible" scores −4, but "not" is assigned 0 — so the model reports negative sentiment, missing the negation entirely. Rule-based negation windows (flip polarity of next 3 words after a negator) partially fix this.' },
  { id:3, tag:"ML", q:'What is the "bag-of-words" assumption and what important linguistic information does it discard?', opts:["All words are in a bag and shuffled randomly to prevent bias","Text is represented as unordered word counts — discarding word order, grammar, and positional context","Words are weighted by their frequency in the corpus, discarding rare words","Each sentence is treated as an independent bag of characters, not words"], ans:1, exp:'"I love this, not hate it" and "I hate this, not love it" produce identical BoW vectors. Order, syntax, and grammar are completely lost. This is why Naive Bayes and vanilla TF-IDF SVMs struggle with negation and contrast.' },
  { id:4, tag:"Transformer", q:"What role does the [CLS] token play in BERT-based sentiment classification?", opts:["It classifies each individual word's sentiment independently","Its final hidden state aggregates full-sentence context and is fed into the classification head","It marks the boundary between two sentences in pairwise tasks only","It replaces unknown words that fall outside the vocabulary"], ans:1, exp:"[CLS] (classification) is prepended to every input. After 12 layers of self-attention, its hidden state h ∈ ℝ⁷⁶⁸ has attended to every other token — encoding the full sentence semantics. A linear layer + softmax maps this vector to class probabilities." },
  { id:5, tag:"General", q:'What is "aspect-based sentiment analysis" (ABSA) and how does it differ from document-level sentiment?', opts:["ABSA analyzes aspects of a user's writing style rather than their opinion","ABSA assigns sentiment to specific entities or aspects within a review, rather than one overall label for the document","ABSA uses a different tokenization strategy optimized for aspect words","ABSA is identical to document-level analysis but applied to shorter texts"], ans:1, exp:'Document-level: "This phone is great" → Positive. ABSA: "The camera is great but battery life is terrible" → {camera: Positive, battery: Negative}. ABSA is much harder — it requires span detection and sentiment attribution.' },
  { id:6, tag:"Lexicon", q:"What is the main advantage of a pre-built sentiment lexicon like VADER over a simple word-list approach?", opts:["VADER was trained on ImageNet and understands visual context","VADER includes rules for capitalization, punctuation (!!!), emoticons, and degree modifiers, not just word scores","VADER supports 120 languages natively through cross-lingual embeddings","VADER uses a neural network internally despite appearing rule-based"], ans:1, exp:"VADER goes beyond word lookup: \"GREAT!\" scores higher than \"great\" (capitalization + punctuation boost). \"very good\" scores higher than \"good\" (degree modifier). It also handles emoticons :) and social media slang." },
  { id:7, tag:"Transformer", q:"Why do transformers handle sarcasm better than lexicon or bag-of-words models?", opts:["Transformers are trained on sarcasm-specific datasets by default","Self-attention allows each word to incorporate the full sentence context — capitalization, surrounding irony, and tone all influence the representation","Transformers filter out sarcastic sentences as noise during training","Transformers classify sarcasm as a separate third class distinct from positive and negative"], ans:1, exp:'Attention mechanisms let the model learn that "Oh sure, absolutely PERFECT" in context of a complaint thread = negative despite positive words. The model sees capitalization, discourse markers, and the surrounding context simultaneously.' },
];

const EDGE_CASES = [
  { text:"This movie is not bad at all.", note:'Double negation — "not bad" is positive', lx:"Negative (misses negation)", ml:"Neutral (bag-of-words)", tr:"Positive (correct)" },
  { text:"Oh sure, the service was absolutely PERFECT.", note:"Sarcasm via capitalization and irony", lx:"Strongly Positive (wrong)", ml:"Positive (wrong)", tr:"Negative (correct with fine-tuning)" },
  { text:"The price is high but the quality is worth it.", note:"Contrastive sentence — two opposing opinions", lx:"Mixed", ml:"Neutral", tr:"Positive (net positive framing)" },
  { text:"I expected it to be terrible but it surprised me.", note:"Expectation violation — context shift", lx:'Negative (scores "terrible")', ml:"Negative", tr:"Positive (understands surprise reversal)" },
];

const APPROACHES = [
  { name:"Lexicon-based", color:"#185FA5", bg:"#E6F1FB", bd:"#85B7EB", pros:["No training data needed","Fast, interpretable","Domain-agnostic baseline"], cons:["Fixed vocabulary","Misses context","Struggles with sarcasm"], use:"Rapid prototyping, low-resource settings" },
  { name:"ML classifiers", color:"#27500A", bg:"#EAF3DE", bd:"#97C459", pros:["Learns from data","Adapts to domain","Handles unseen patterns"], cons:["Needs labeled data","Feature engineering required","BoW ignores word order"], use:"Production systems with labeled training sets" },
  { name:"Transformers", color:"#3C3489", bg:"#EEEDFE", bd:"#AFA9EC", pros:["State of the art","Context-aware","Handles negation, sarcasm"], cons:["Requires GPU","Expensive to fine-tune","Black-box"], use:"High-accuracy production, nuanced analysis" },
];

const NAV_ITEMS = [
  { id:"bas", icon:"📘", label:"Basics" },
  { id:"viz", icon:"📊", label:"Live Analyzer" },
  { id:"sim", icon:"⚡", label:"Simulator" },
  { id:"cmp", icon:"↔", label:"Comparison" },
  { id:"quiz", icon:"🧠", label:"Quiz" },
];

const HERO_META = {
  bas:  { pill:"Foundations",      sub:"NLP · Sentiment Analysis", title:"Sentiment Analysis, From Scratch", desc:"How a computer figures out if a sentence is happy, angry, or somewhere in between — starting with zero assumptions." },
  viz:  { pill:"Hands-On Practice", sub:"Live Analyzer",            title:"Try It Yourself",                  desc:"Type any sentence and watch a simple sentiment engine score it, word by word, in real time." },
  sim:  { pill:"Step By Step",      sub:"Pipeline Simulator",       title:"Inside The Engine",                desc:"Walk through exactly how three different kinds of sentiment models think, one step at a time." },
  cmp:  { pill:"Side By Side",      sub:"Model Comparison",         title:"Which Model Wins?",                desc:"Same sentence, three different opinions. See where simple models break and smarter ones don't." },
  quiz: { pill:"Check Yourself",    sub:"Concept Quiz",             title:"Test What You Know",               desc:"Questions covering the basics, the math, and the gotchas — with instant feedback on every answer." },
};

// ─── HELPERS ─────────────────────────────────────────────────────────────
function tokenizeSentence(text) {
  return text.toLowerCase().replace(/[^\w\s'-]/g, " ").split(/\s+/).filter(Boolean);
}

function analyzeTokens(text, lexicon = BASE_LEXICON) {
  const tokens = tokenizeSentence(text);
  const result = [];
  let negate = false, negateCount = 0, intensify = 1;
  tokens.forEach((raw) => {
    const score = lexicon[raw];
    let role = "neu", finalScore = 0;
    if (NEGATORS.has(raw)) { negate = true; negateCount = 3; role = "neg-mod"; }
    else if (INTENSIFIERS[raw] !== undefined) { intensify = INTENSIFIERS[raw]; role = "intensifier"; }
    else if (score !== undefined && score !== 0) {
      finalScore = score * intensify * (negate ? -1 : 1);
      role = finalScore > 0 ? "pos" : "neg";
      intensify = 1;
      if (negateCount > 0) { negateCount--; if (negateCount === 0) negate = false; }
    } else {
      intensify = 1;
      if (negateCount > 0) { negateCount--; if (negateCount === 0) negate = false; }
    }
    result.push({ word: raw, raw: score || 0, finalScore, role, negated: negate && score !== undefined && score !== 0 });
  });
  return result;
}

function getSentimentLabel(score, count) {
  if (count === 0) return { label: "Neutral", color: "#3C3489" };
  const avg = score / count;
  if (avg >= 2) return { label: "Strongly Positive", color: "#1a7a4a" };
  if (avg > 0.3) return { label: "Positive", color: "#1a7a4a" };
  if (avg <= -2) return { label: "Strongly Negative", color: "#a32d2d" };
  if (avg < -0.3) return { label: "Negative", color: "#a32d2d" };
  return { label: "Mixed / Neutral", color: "#854f0b" };
}

// ─── STATE ───────────────────────────────────────────────────────────────
const STATE = {
  tab: "bas",
  basics: { approach: "lex", exIdx: 0, granIdx: 0, appIdx: 0 },
  live: { text: SAMPLES.mix, lexSearch: "", lexFilter: "all", customLexicon: {} },
  sim: { pipeline: "A", simInput: "The product is absolutely fantastic but delivery was terrible", step: 0, simData: null },
  cmp: { cmpInput: "The service was okay but could definitely be better" },
};

// ─── APP SHELL ───────────────────────────────────────────────────────────
function initApp() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="sv-shell">
      <div class="sv-hero" id="heroBox"></div>
      <div class="sv-body">
        <aside class="sv-sidebar">
          <nav class="sv-navlist" id="navList"></nav>
          <div class="sv-navfoot">💡 New here? Start at Basics, then work down the list in order.</div>
        </aside>
        <main class="sv-main"><div class="sv-maininner" id="mainInner"></div></main>
      </div>
    </div>
  `;
  buildNav();
  buildPanels();
  updateHero();
  setActiveNav();
}

function buildNav() {
  document.getElementById("navList").innerHTML = NAV_ITEMS.map(item =>
    `<button class="sv-navitem" data-tab="${item.id}" onclick="switchTab('${item.id}')"><span class="ni-ic">${item.icon}</span>${escapeHtml(item.label)}</button>`
  ).join("");
}

function setActiveNav() {
  document.querySelectorAll(".sv-navitem").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === STATE.tab);
  });
}

function updateHero() {
  const h = HERO_META[STATE.tab];
  document.getElementById("heroBox").innerHTML = `
    <div class="sv-hero-pillrow">
      <span class="sv-hero-pill">${escapeHtml(h.pill)}</span>
      <span class="sv-hero-pillsub">${escapeHtml(h.sub)}</span>
    </div>
    <h1>${escapeHtml(h.title)}</h1>
    <p>${escapeHtml(h.desc)}</p>
  `;
}

function switchTab(id) {
  STATE.tab = id;
  updateHero();
  setActiveNav();
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.style.display = (p.dataset.tab === id) ? "" : "none";
  });
}
window.switchTab = switchTab;

function buildPanels() {
  document.getElementById("mainInner").innerHTML = `
    <section class="tab-panel" data-tab="bas" id="panel-bas"></section>
    <section class="tab-panel" data-tab="viz" id="panel-viz" style="display:none"></section>
    <section class="tab-panel" data-tab="sim" id="panel-sim" style="display:none"></section>
    <section class="tab-panel" data-tab="cmp" id="panel-cmp" style="display:none"></section>
    <section class="tab-panel" data-tab="quiz" id="panel-quiz" style="display:none"></section>
  `;
  mountBasics();
  mountLiveAnalyzer();
  mountSimulator();
  mountComparison();
  mountQuiz();
}

document.addEventListener("DOMContentLoaded", initApp);

// ─── TAB: BASICS ─────────────────────────────────────────────────────────
function approachData(key) {
  return { lex: APPROACHES[0], ml: APPROACHES[1], tr: APPROACHES[2] }[key];
}

function pipelineFlowSVG() {
  const branches = [
    { key:"lex", x:60,  label:"Lexicon",       color:"#185FA5", bg:"#E6F1FB" },
    { key:"ml",  x:295, label:"ML Classifier", color:"#27500A", bg:"#EAF3DE" },
    { key:"tr",  x:530, label:"Transformer",   color:"#3C3489", bg:"#EEEDFE" },
  ];
  const branchMarkup = branches.map(b => `
    <g class="flow-branch" onclick="selectApproach('${b.key}')" data-key="${b.key}">
      <rect id="branch-rect-${b.key}" x="${b.x}" y="224" width="130" height="40" rx="8" fill="${b.bg}" stroke="${b.color}" stroke-width="1.5"></rect>
      <text id="branch-text-${b.key}" x="${b.x + 65}" y="248" text-anchor="middle" fill="${b.color}" font-size="11.5" font-weight="700">${b.label}</text>
    </g>`).join("");
  return `
    <svg viewBox="0 0 720 340" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;">
      <defs>
        <marker id="saArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 Z" fill="#6c63ff"></path>
        </marker>
      </defs>
      <rect x="260" y="8" width="200" height="42" rx="10" fill="#0f1123"></rect>
      <text x="360" y="34" text-anchor="middle" fill="#fff" font-size="13" font-weight="700">Raw Text</text>
      <line x1="360" y1="50" x2="360" y2="72" stroke="#6c63ff" stroke-width="2" marker-end="url(#saArrow)"></line>
      <rect x="235" y="74" width="250" height="46" rx="10" fill="#EEEDFE" stroke="#AFA9EC" stroke-width="1.5"></rect>
      <text x="360" y="94" text-anchor="middle" fill="#3C3489" font-size="12.5" font-weight="700">Clean &amp; Tokenize</text>
      <text x="360" y="110" text-anchor="middle" fill="#6c63ff" font-size="10">lowercase → split into words</text>
      <line x1="360" y1="120" x2="360" y2="142" stroke="#6c63ff" stroke-width="2" marker-end="url(#saArrow)"></line>
      <rect x="210" y="144" width="300" height="42" rx="10" fill="#6c63ff"></rect>
      <text x="360" y="170" text-anchor="middle" fill="#fff" font-size="12.5" font-weight="700">Score With A Model — click one ↓</text>
      <line x1="360" y1="186" x2="360" y2="204" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="120" y1="204" x2="600" y2="204" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="120" y1="204" x2="120" y2="222" stroke="#6c63ff" stroke-width="1.5" marker-end="url(#saArrow)"></line>
      <line x1="360" y1="204" x2="360" y2="222" stroke="#6c63ff" stroke-width="1.5" marker-end="url(#saArrow)"></line>
      <line x1="600" y1="204" x2="600" y2="222" stroke="#6c63ff" stroke-width="1.5" marker-end="url(#saArrow)"></line>
      ${branchMarkup}
      <line x1="120" y1="264" x2="120" y2="280" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="360" y1="264" x2="360" y2="280" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="600" y1="264" x2="600" y2="280" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="120" y1="280" x2="600" y2="280" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="360" y1="280" x2="360" y2="296" stroke="#6c63ff" stroke-width="2" marker-end="url(#saArrow)"></line>
      <rect x="260" y="298" width="200" height="38" rx="10" fill="#0f1123"></rect>
      <text x="360" y="321" text-anchor="middle" fill="#fff" font-size="13" font-weight="700">Sentiment Label</text>
    </svg>`;
}

function mountBasics() {
  const panel = document.getElementById("panel-bas");
  panel.innerHTML = `
    <div class="sv-eyebrow">How It Works, At A Glance</div>
    <div class="sv-card">
      <div class="sv-card-title">From Text To Label</div>
      <p class="sv-desc">Every sentiment tool follows the same basic path. Click a box below to preview each modeling approach.</p>
      <div class="flow-wrap">${pipelineFlowSVG()}</div>
      <div class="sblock" id="approachInfo"></div>
      <div class="sv-insight i-bas">👉 You'll build the lexicon approach yourself in <strong>Live Analyzer</strong>, and step through all three in <strong>Simulator</strong>.</div>
    </div>

    <div class="sv-eyebrow">Two Things We Measure</div>
    <div class="sv-card">
      <div class="sv-card-title">Click A Sentence To Test It</div>
      <div class="word-grid" id="polChips"></div>
      <div class="sblock" id="polSentence" style="font-style:italic;font-size:13px;"></div>
      <div class="grid2" style="margin-bottom:0;">
        <div>
          <div class="sblock-lbl">Polarity</div>
          <div class="dial-track"><div class="dial-marker" id="polMarker"></div></div>
          <div class="dial-labels"><span>Negative</span><span>Neutral</span><span>Positive</span></div>
        </div>
        <div>
          <div class="sblock-lbl">Subjectivity</div>
          <div class="dial-track" style="background:linear-gradient(90deg, var(--color-background-secondary,#f4f3ee), var(--bas-bg));"><div class="dial-marker" id="subjMarker"></div></div>
          <div class="dial-labels"><span>Fact</span><span>—</span><span>Opinion</span></div>
        </div>
      </div>
      <div class="sv-insight i-bas" id="polNote"></div>
    </div>

    <div class="sv-eyebrow">How Much Detail Do You Want?</div>
    <div class="sv-card">
      <div class="sv-card-title">Pick A Zoom Level</div>
      <div class="sv-stabs" id="granTabs"></div>
      <div class="sblock" id="granPanel"></div>
    </div>

    <div class="sv-eyebrow">Why It Matters</div>
    <div class="sv-card">
      <div class="sv-card-title">Click To See Where It's Used</div>
      <div class="grid3c" id="appGrid"></div>
      <div class="sblock" id="appPanel" style="margin-top:10px;margin-bottom:0;"></div>
    </div>
  `;
  renderApproachBranches();
  updateApproachInfo();
  renderPolChips();
  updatePolarityExplorer();
  renderGranTabs();
  updateGranPanel();
  renderAppGrid();
  updateAppPanel();
}

function selectApproach(key) {
  STATE.basics.approach = key;
  renderApproachBranches();
  updateApproachInfo();
}
window.selectApproach = selectApproach;

function renderApproachBranches() {
  const colorMap = { lex:"#185FA5", ml:"#27500A", tr:"#3C3489" };
  const bgMap = { lex:"#E6F1FB", ml:"#EAF3DE", tr:"#EEEDFE" };
  ["lex","ml","tr"].forEach(k => {
    const rect = document.getElementById(`branch-rect-${k}`);
    const text = document.getElementById(`branch-text-${k}`);
    const isSel = STATE.basics.approach === k;
    rect.setAttribute("fill", isSel ? colorMap[k] : bgMap[k]);
    rect.setAttribute("stroke-width", isSel ? "2.5" : "1.5");
    text.setAttribute("fill", isSel ? "#fff" : colorMap[k]);
  });
}

function updateApproachInfo() {
  const a = approachData(STATE.basics.approach);
  const prosChips = a.pros.map(p => `<span class="chip c-pos" style="font-size:10px;">${escapeHtml(p)}</span>`).join("");
  const consChip = `<span class="chip c-neg" style="font-size:10px;">${escapeHtml(a.cons[0])}</span>`;
  document.getElementById("approachInfo").innerHTML = `
    <span class="sv-badge" style="background:${a.bg};color:${a.color};border:0.5px solid ${a.bd};">${escapeHtml(a.name)}</span>
    <div style="font-size:12.5px;color:var(--color-text-secondary,#5a5950);margin:4px 0 8px;">Best for: ${escapeHtml(a.use)}</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;">${prosChips}${consChip}</div>
  `;
}

function renderPolChips() {
  document.getElementById("polChips").innerHTML = POLARITY_EXAMPLES.map((e, i) => {
    const label = e.text.length > 30 ? e.text.slice(0, 30) + "…" : e.text;
    const active = i === STATE.basics.exIdx;
    return `<button class="chip ${active ? "c-bas" : "c-gray"}" style="font-weight:${active ? 600 : 500};" onclick="selectExample(${i})">${escapeHtml(label)}</button>`;
  }).join("");
}

function selectExample(i) {
  STATE.basics.exIdx = i;
  renderPolChips();
  updatePolarityExplorer();
}
window.selectExample = selectExample;

function updatePolarityExplorer() {
  const ex = POLARITY_EXAMPLES[STATE.basics.exIdx];
  document.getElementById("polSentence").textContent = `"${ex.text}"`;
  document.getElementById("polMarker").style.left = ex.pol + "%";
  document.getElementById("subjMarker").style.left = ex.subj + "%";
  document.getElementById("polNote").textContent = ex.note;
}

function renderGranTabs() {
  document.getElementById("granTabs").innerHTML = GRANULARITY.map((g, i) => {
    const active = i === STATE.basics.granIdx;
    const style = active ? `background:${g.bg};border-color:${g.bd};color:${g.color};font-weight:600;` : "";
    return `<button class="sv-stab" style="${style}" onclick="selectGran(${i})">${escapeHtml(g.level)}</button>`;
  }).join("");
}

function selectGran(i) {
  STATE.basics.granIdx = i;
  renderGranTabs();
  updateGranPanel();
}
window.selectGran = selectGran;

function updateGranPanel() {
  const g = GRANULARITY[STATE.basics.granIdx];
  const panel = document.getElementById("granPanel");
  panel.style.borderLeft = `3px solid ${g.color}`;
  panel.innerHTML = `
    <div style="font-size:12.5px;font-style:italic;color:var(--color-text-secondary,#5a5950);margin-bottom:6px;">${escapeHtml(g.example)}</div>
    <div style="font-size:12px;font-weight:600;color:${g.color};margin-bottom:6px;">${escapeHtml(g.output)}</div>
    <div style="font-size:11.5px;color:var(--color-text-tertiary,#8a8878);line-height:1.55;">${escapeHtml(g.note)}</div>
  `;
}

function renderAppGrid() {
  document.getElementById("appGrid").innerHTML = APPLICATIONS.map((a, i) => `
    <button class="pick-card${i === STATE.basics.appIdx ? " active" : ""}" onclick="selectApp(${i})">
      <div class="pc-ic">${a.icon}</div>
      <div class="pc-title">${escapeHtml(a.title)}</div>
    </button>`).join("");
}

function selectApp(i) {
  STATE.basics.appIdx = i;
  renderAppGrid();
  updateAppPanel();
}
window.selectApp = selectApp;

function updateAppPanel() {
  const a = APPLICATIONS[STATE.basics.appIdx];
  document.getElementById("appPanel").innerHTML = `
    <div class="sblock-lbl">${escapeHtml(a.title)}</div>
    <div style="font-size:12.5px;color:var(--color-text-secondary,#5a5950);line-height:1.6;">${escapeHtml(a.text)}</div>
  `;
}

// ─── TAB: LIVE ANALYZER ──────────────────────────────────────────────────
function mergedLexicon() {
  return Object.assign({}, BASE_LEXICON, STATE.live.customLexicon);
}

function mountLiveAnalyzer() {
  const panel = document.getElementById("panel-viz");
  panel.innerHTML = `
    <div class="sv-eyebrow">Type Something, See What Happens</div>
    <div class="sv-card">
      <div class="sv-badge b-blue">Live analysis</div>
      <div class="sv-card-title">Interactive Sentiment Analyzer</div>
      <p class="sv-desc">Type or paste any sentence below. Each word gets checked against a list of positive and negative words — hover a highlighted word to see its score.</p>
      <div class="analyzer-box">
        <textarea class="analyzer-input" id="laTextarea" placeholder="Type something here… e.g. 'The movie was absolutely fantastic but the ending felt rushed and disappointing.'"></textarea>
        <div class="analyzer-footer">
          <button class="sv-btn btn-pos" onclick="loadSample('pos')">Load positive sample</button>
          <button class="sv-btn btn-neg" onclick="loadSample('neg')">Load negative sample</button>
          <button class="sv-btn btn-sec" onclick="loadSample('mix')">Load mixed sample</button>
          <button class="sv-btn btn-sec" style="margin-left:auto;" onclick="clearLaText()">Clear</button>
        </div>
      </div>
      <div id="laAnalysisResult"></div>
    </div>

    <div class="sv-card">
      <div class="sv-badge b-bas">Build it yourself</div>
      <div class="sv-card-title">Build Your Own Lexicon</div>
      <p class="sv-desc">A lexicon is just a dictionary of words mapped to scores. Add your own words — slang, domain jargon, anything — and watch the analyzer above update instantly. You can even override a built-in word's score.</p>
      <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;align-items:center;">
        <input class="sv-input-sm" style="flex:1;min-width:160px;" id="laNewWord" placeholder="word, e.g. 'fire' or 'mid'">
        <select class="sv-select" id="laNewScore">
          ${[5,4,3,2,1,-1,-2,-3,-4,-5].map(s => `<option value="${s}" ${s===2?"selected":""}>${s>0?"+":""}${s}</option>`).join("")}
        </select>
        <button class="sv-btn btn-bas" onclick="addLexWord()">＋ Add / update word</button>
        <button class="sv-btn btn-sec" id="laResetBtn" onclick="resetLexicon()">↺ Reset custom words</button>
      </div>
      <div id="laCustomWords"></div>
      <div class="sv-insight i-bas">💡 This is exactly how real lexicon tools like VADER or domain-specific dictionaries get extended — someone manually decides a word's polarity and adds it. It's fast, but it doesn't scale to millions of words, which is why ML and transformer approaches exist.</div>
    </div>

    <div class="sv-card">
      <div class="sv-card-title">Lexicon word map <em>— all scored tokens</em></div>
      <p class="sv-desc">Every word currently in the lexicon (built-in + your additions), shown with its polarity score. Words marked ✦ are yours.</p>
      <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;">
        <input class="sv-select" id="laLexSearch" placeholder="Search word…" style="flex:1;min-width:140px;font-size:12px;padding:6px 10px;">
        <select class="sv-select" id="laLexFilter">
          <option value="all">All words</option>
          <option value="pos">Positive only</option>
          <option value="neg">Negative only</option>
          <option value="strong">Strong (|score|≥3)</option>
          <option value="custom">Your words only</option>
        </select>
      </div>
      <div class="word-grid" id="laLexMap"></div>
      <div style="font-size:11px;color:var(--color-text-tertiary,#8a8878);margin-top:4px;" id="laLexCount"></div>
    </div>

    <div class="sv-card">
      <div class="sv-card-title">Negation &amp; intensifier handling</div>
      <p class="sv-desc">Simple lexicon methods can be extended with rules: negations flip polarity, intensifiers multiply scores.</p>
      <div id="laNegPairs"></div>
      <div class="sv-insight i-info">💡 Rule-based negation (e.g. "not good" = negative) works for simple cases, but struggles with long-range dependencies. Transformers handle this natively through attention.</div>
    </div>
  `;

  const ta = document.getElementById("laTextarea");
  ta.value = STATE.live.text;
  ta.addEventListener("input", e => { STATE.live.text = e.target.value; updateLaAnalysis(); });

  document.getElementById("laLexSearch").addEventListener("input", e => { STATE.live.lexSearch = e.target.value; updateLaLexMap(); });
  document.getElementById("laLexFilter").addEventListener("change", e => { STATE.live.lexFilter = e.target.value; updateLaLexMap(); });

  updateLaAnalysis();
  updateLaCustomWords();
  updateLaLexMap();
  updateLaNegPairs();
}

function loadSample(key) {
  STATE.live.text = SAMPLES[key];
  document.getElementById("laTextarea").value = STATE.live.text;
  updateLaAnalysis();
}
window.loadSample = loadSample;

function clearLaText() {
  STATE.live.text = "";
  document.getElementById("laTextarea").value = "";
  updateLaAnalysis();
}
window.clearLaText = clearLaText;

function updateLaAnalysis() {
  const text = STATE.live.text;
  const container = document.getElementById("laAnalysisResult");
  if (!text.trim()) { container.innerHTML = ""; return; }
  const lex = mergedLexicon();
  const tokens = analyzeTokens(text, lex);
  const scored = tokens.filter(t => t.finalScore !== 0);
  const total = scored.reduce((a, t) => a + t.finalScore, 0);
  const sentiment = getSentimentLabel(total, scored.length);
  const posScore = scored.filter(t => t.finalScore > 0).reduce((a, t) => a + t.finalScore, 0);
  const negScore = Math.abs(scored.filter(t => t.finalScore < 0).reduce((a, t) => a + t.finalScore, 0));
  const norm = posScore + negScore || 1;
  const posP = Math.round(posScore / norm * 100);
  const negP = 100 - posP;

  const tokensHtml = tokens.map(t => {
    const cls = t.role === "pos" ? "tok-pos" : t.role === "neg" ? "tok-neg" : t.role === "intensifier" ? "tok-cmp" : "tok-neu";
    const title = t.finalScore !== 0 ? `score: ${t.finalScore > 0 ? "+" : ""}${t.finalScore.toFixed(1)}${t.negated ? " (negated)" : ""}` : t.role === "intensifier" ? "intensifier" : t.role === "neg-mod" ? "negator" : "";
    return `<span class="tok ${cls}" title="${escapeHtml(title)}">${escapeHtml(t.word)} </span>`;
  }).join("");

  container.innerHTML = `
    <div class="grid3" style="margin-bottom:14px;">
      <div class="metric"><div class="metric-lbl">Sentiment</div><div style="font-size:16px;font-weight:600;color:${sentiment.color};">${sentiment.label}</div></div>
      <div class="metric"><div class="metric-lbl">Compound score</div><div class="metric-val" style="color:${total >= 0 ? "var(--pos)" : "var(--neg)"};">${total >= 0 ? "+" : ""}${total.toFixed(1)}</div></div>
      <div class="metric"><div class="metric-lbl">Scored tokens</div><div class="metric-val">${scored.length} <span style="font-size:13px;font-weight:400;color:var(--color-text-tertiary,#8a8878);">/ ${tokens.length}</span></div></div>
    </div>
    <div class="sblock" style="margin-bottom:10px;">
      <div class="sblock-lbl">Token highlights — hover for score</div>
      <div class="token-wrap">${tokensHtml}</div>
    </div>
    <div class="sblock">
      <div class="sblock-lbl">Polarity balance</div>
      <div class="sbar-row" style="margin-bottom:6px;">
        <div class="sbar-label" style="color:var(--pos);">Positive</div>
        <div class="sbar-track"><div class="sbar-fill" style="width:${posP}%;background:var(--pos);"></div></div>
        <div class="sbar-val">${posP}%</div>
      </div>
      <div class="sbar-row">
        <div class="sbar-label" style="color:var(--neg);">Negative</div>
        <div class="sbar-track"><div class="sbar-fill" style="width:${negP}%;background:var(--neg);"></div></div>
        <div class="sbar-val">${negP}%</div>
      </div>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;">
      <span style="font-size:11px;color:var(--color-text-tertiary,#8a8878);">Legend:</span>
      <span class="chip c-pos">Positive word</span>
      <span class="chip c-neg">Negative word</span>
      <span class="chip c-cmp">Intensifier</span>
      <span class="chip c-gray">Neutral / stop word</span>
    </div>
  `;
}

function addLexWord() {
  const wInput = document.getElementById("laNewWord");
  const sSelect = document.getElementById("laNewScore");
  const w = wInput.value.trim().toLowerCase();
  if (!w) return;
  STATE.live.customLexicon[w] = Number(sSelect.value);
  wInput.value = "";
  updateLaCustomWords();
  updateLaLexMap();
  updateLaAnalysis();
  updateLaNegPairs();
}
window.addLexWord = addLexWord;

function removeLexWord(w) {
  delete STATE.live.customLexicon[w];
  updateLaCustomWords();
  updateLaLexMap();
  updateLaAnalysis();
  updateLaNegPairs();
}
window.removeLexWord = removeLexWord;

function resetLexicon() {
  STATE.live.customLexicon = {};
  updateLaCustomWords();
  updateLaLexMap();
  updateLaAnalysis();
  updateLaNegPairs();
}
window.resetLexicon = resetLexicon;

function updateLaCustomWords() {
  const keys = Object.keys(STATE.live.customLexicon);
  const container = document.getElementById("laCustomWords");
  document.getElementById("laResetBtn").disabled = keys.length === 0;
  if (keys.length === 0) {
    container.innerHTML = `<div style="font-size:11.5px;color:var(--color-text-tertiary,#8a8878);margin-bottom:8px;">No custom words yet — try adding "fire" as +3 or "mid" as −2, then type a sentence using it above.</div>`;
    return;
  }
  container.innerHTML = `<div class="word-grid">${keys.map(w => {
    const s = STATE.live.customLexicon[w];
    return `<span class="chip ${s > 0 ? "c-pos" : "c-neg"} c-mono c-removable" onclick="removeLexWord('${escapeJs(w)}')" title="Click to remove">✦ ${escapeHtml(w)} ${s > 0 ? "+" : ""}${s} ×</span>`;
  }).join("")}</div>`;
}

function updateLaLexMap() {
  const lex = mergedLexicon();
  const customKeys = Object.keys(STATE.live.customLexicon);
  let entries = Object.entries(lex).filter(([, s]) => s !== 0);
  const search = STATE.live.lexSearch.toLowerCase();
  const filter = STATE.live.lexFilter;
  if (search) entries = entries.filter(([w]) => w.includes(search));
  if (filter === "pos") entries = entries.filter(([, s]) => s > 0);
  if (filter === "neg") entries = entries.filter(([, s]) => s < 0);
  if (filter === "strong") entries = entries.filter(([, s]) => Math.abs(s) >= 3);
  if (filter === "custom") entries = entries.filter(([w]) => customKeys.includes(w));
  entries.sort((a, b) => b[1] - a[1]);

  document.getElementById("laLexMap").innerHTML = entries.map(([w, s]) => {
    const isCustom = customKeys.includes(w);
    const opacity = 0.5 + (Math.abs(s) / 5) * 0.5;
    const fontSize = 11 + Math.abs(s);
    const outline = isCustom ? "1.5px solid var(--bas)" : "none";
    return `<span class="chip ${s > 0 ? "c-pos" : "c-neg"} c-mono" style="opacity:${opacity};font-size:${fontSize}px;outline:${outline};outline-offset:1px;" title="${escapeHtml(w)}: ${s > 0 ? "+" : ""}${s}">${isCustom ? "✦ " : ""}${escapeHtml(w)} <span style="font-weight:700;">${s > 0 ? "+" : ""}${s}</span></span>`;
  }).join("");
  document.getElementById("laLexCount").textContent = `${entries.length} words shown. Larger = stronger score.`;
}

function updateLaNegPairs() {
  const lex = mergedLexicon();
  const pairs = [
    { orig:"The food is good.", neg:"The food is not good.", intensified:"The food is absolutely good." },
    { orig:"Fast delivery.", neg:"Not fast delivery.", intensified:"Incredibly fast delivery." },
    { orig:"Service was great.", neg:"Service was never great.", intensified:"Service was truly great." },
  ];
  document.getElementById("laNegPairs").innerHTML = pairs.map(p => {
    const o = analyzeTokens(p.orig, lex).reduce((a, t) => a + t.finalScore, 0);
    const n = analyzeTokens(p.neg, lex).reduce((a, t) => a + t.finalScore, 0);
    const it = analyzeTokens(p.intensified, lex).reduce((a, t) => a + t.finalScore, 0);
    const rows = [
      { text:p.orig, score:o, cls:"c-pos" },
      { text:p.neg, score:n, cls:"c-neg" },
      { text:p.intensified, score:it, cls:"c-cmp" },
    ];
    return `<div class="sblock" style="margin-bottom:8px;">${rows.map((row, j) => `
      <div style="display:grid;grid-template-columns:1fr 80px;gap:8px;align-items:center;margin-bottom:${j < 2 ? "5px" : "0"};">
        <span style="font-size:12.5px;color:var(--color-text-secondary,#5a5950);">${escapeHtml(row.text)}</span>
        <span class="chip ${row.cls} c-mono">${row.score >= 0 ? "+" : ""}${row.score.toFixed(1)}</span>
      </div>`).join("")}</div>`;
  }).join("");
}

// ─── TAB: SIMULATOR ──────────────────────────────────────────────────────
function mountSimulator() {
  const panel = document.getElementById("panel-sim");
  panel.innerHTML = `
    <div class="sv-eyebrow">Pick A Pipeline</div>
    <div class="sv-stabs" id="simPipelineTabs"></div>
    <div class="sv-card">
      <div class="sv-badge b-blue">Step-through simulator</div>
      <div class="sv-card-title" id="simPipeName"></div>
      <p class="sv-desc" id="simPipeDesc"></p>
      <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
        <input class="sv-input sv-mono" id="simInputField" style="font-size:12.5px;">
        <button class="sv-btn btn-blue" onclick="runSimulation()">Analyze</button>
      </div>
      <div class="sv-stepper" id="simStepper"></div>
    </div>
    <div id="simStepContent"></div>
    <div class="sim-nav">
      <button class="sv-btn btn-sec" id="simPrevBtn" onclick="simStepNav(-1)">← Previous</button>
      <span class="sim-nav-label" id="simStepLabel"></span>
      <button class="sv-btn btn-blue" id="simNextBtn" onclick="simStepNav(1)">Next →</button>
    </div>
  `;
  document.getElementById("simInputField").value = STATE.sim.simInput;
  document.getElementById("simInputField").addEventListener("input", e => {
    STATE.sim.simInput = e.target.value;
    STATE.sim.simData = null;
    STATE.sim.step = 0;
    renderSimStepperAndContent();
  });
  renderSimPipelineTabs();
  runSimulation();
}

function renderSimPipelineTabs() {
  const tabs = [["A","av","Lexicon pipeline"],["B","bv","ML pipeline"],["C","cv","Transformer pipeline"]];
  document.getElementById("simPipelineTabs").innerHTML = tabs.map(([p, cls, label]) =>
    `<button class="sv-stab${STATE.sim.pipeline === p ? " " + cls : ""}" onclick="switchSimPipeline('${p}')">${label}</button>`
  ).join("");
  document.getElementById("simPipeName").textContent = SIM_PIPELINES[STATE.sim.pipeline].name;
  document.getElementById("simPipeDesc").textContent = SIM_PIPELINES[STATE.sim.pipeline].desc;
}

function switchSimPipeline(p) {
  STATE.sim.pipeline = p;
  STATE.sim.step = 0;
  STATE.sim.simData = null;
  renderSimPipelineTabs();
  runSimulation();
}
window.switchSimPipeline = switchSimPipeline;

function runSimulation() {
  const text = STATE.sim.simInput;
  const tokens = analyzeTokens(text);
  const scored = tokens.filter(t => t.finalScore !== 0);
  const total = scored.reduce((a, t) => a + t.finalScore, 0);
  const sentiment = getSentimentLabel(total, scored.length);
  STATE.sim.simData = { text, tokens, scored, total, sentiment };
  STATE.sim.step = 0;
  renderSimStepperAndContent();
}
window.runSimulation = runSimulation;

function renderSimStepperAndContent() {
  const steps = SIM_PIPELINES[STATE.sim.pipeline].steps;
  document.getElementById("simStepper").innerHTML = steps.map((label, i) => {
    const cls = i === STATE.sim.step ? "active" : i < STATE.sim.step ? "done" : "";
    return `${i > 0 ? `<div class="sconn${i <= STATE.sim.step ? " done" : ""}"></div>` : ""}
      <div class="snode ${cls}" onclick="jumpSimStep(${i})">
        <div class="scircle">${i < STATE.sim.step ? "✓" : i + 1}</div>
        <div class="slabel">${escapeHtml(label)}</div>
      </div>`;
  }).join("");
  document.getElementById("simStepLabel").textContent = `Step ${STATE.sim.step + 1} of ${steps.length}`;
  document.getElementById("simPrevBtn").disabled = STATE.sim.step === 0;
  document.getElementById("simNextBtn").disabled = STATE.sim.step === steps.length - 1;
  renderSimStepContent();
}

function jumpSimStep(i) {
  STATE.sim.step = i;
  renderSimStepperAndContent();
}
window.jumpSimStep = jumpSimStep;

function simStepNav(delta) {
  const steps = SIM_PIPELINES[STATE.sim.pipeline].steps;
  STATE.sim.step = Math.max(0, Math.min(steps.length - 1, STATE.sim.step + delta));
  renderSimStepperAndContent();
}
window.simStepNav = simStepNav;

function renderSimStepContent() {
  const container = document.getElementById("simStepContent");
  if (!STATE.sim.simData) {
    container.innerHTML = `<div class="sv-card"><div class="sv-insight i-info">Enter a sentence above and click Analyze to start the walkthrough.</div></div>`;
    return;
  }
  const data = STATE.sim.simData;
  const pipeline = STATE.sim.pipeline;
  const step = STATE.sim.step;
  if (pipeline === "A") container.innerHTML = renderLexiconStep(step, data);
  else if (pipeline === "B") container.innerHTML = renderMlStep(step, data);
  else container.innerHTML = renderTransformerStep(step, data);
}

function renderLexiconStep(step, data) {
  const { text, tokens, scored, total, sentiment } = data;
  if (step === 0) {
    const clean = text.toLowerCase().replace(/[^\w\s]/g, " ").replace(/\s+/g, " ").trim();
    return `
      <div class="sv-card">
        <div class="sv-badge b-blue">Step 1 — Preprocessing</div>
        <p class="sv-desc">Lowercase, remove punctuation, collapse whitespace.</p>
        <div class="sblock"><div class="sblock-lbl">Input</div><div style="font-size:13px;margin-top:4px;">${escapeHtml(text)}</div></div>
        <div style="text-align:center;font-size:20px;color:var(--color-text-tertiary,#8a8878);margin:6px 0;">↓</div>
        <div class="sblock"><div class="sblock-lbl">After preprocessing</div><div style="font-size:13px;font-family:var(--mono);margin-top:4px;">${escapeHtml(clean)}</div></div>
        <div class="sv-insight i-info">💡 Preprocessing varies by model — transformers often skip this step entirely and pass raw text including punctuation, which carries semantic meaning.</div>
      </div>`;
  } else if (step === 1) {
    const chips = tokens.map((t, i) => `<span class="chip c-gray c-mono" style="font-size:12px;">${i + 1}. ${escapeHtml(t.word)}</span>`).join("");
    const uniqueCount = new Set(tokens.map(t => t.word)).size;
    return `
      <div class="sv-card">
        <div class="sv-badge b-blue">Step 2 — Tokenization</div>
        <p class="sv-desc">Split the cleaned text into individual word tokens by whitespace.</p>
        <div class="word-grid">${chips}</div>
        <div class="grid2" style="margin-top:8px;">
          <div class="metric"><div class="metric-lbl">Total tokens</div><div class="metric-val">${tokens.length}</div></div>
          <div class="metric"><div class="metric-lbl">Unique tokens</div><div class="metric-val">${uniqueCount}</div></div>
        </div>
        <div class="sv-insight i-info">💡 Lexicon models tokenize on whitespace. Transformers use subword tokenization (BPE/WordPiece) so "fantastic" might become ["fan","##tas","##tic"].</div>
      </div>`;
  } else if (step === 2) {
    const rows = tokens.map(t => `
      <tr>
        <td class="sv-mono">${escapeHtml(t.word)}</td>
        <td>${t.raw !== 0 ? '<span class="chip c-pos" style="font-size:10px;">yes</span>' : '<span class="chip c-gray" style="font-size:10px;">no</span>'}</td>
        <td class="sv-mono" style="color:${t.raw > 0 ? "var(--pos)" : t.raw < 0 ? "var(--neg)" : "var(--color-text-tertiary,#8a8878)"};">${t.raw !== 0 ? (t.raw > 0 ? "+" : "") + t.raw : "—"}</td>
        <td>${t.role === "intensifier" ? '<span class="chip c-cmp" style="font-size:10px;">intensifier</span>' : t.role === "neg-mod" ? '<span class="chip c-neu" style="font-size:10px;">negator</span>' : t.raw !== 0 ? `<span class="chip ${t.raw > 0 ? "c-pos" : "c-neg"}" style="font-size:10px;">sentiment</span>` : '<span class="chip c-gray" style="font-size:10px;">skip</span>'}</td>
      </tr>`).join("");
    return `
      <div class="sv-card">
        <div class="sv-badge b-blue">Step 3 — Lexicon lookup</div>
        <p class="sv-desc">Each token is looked up in the sentiment lexicon.</p>
        <div class="tbl-wrap"><div class="tbl-scroll">
          <table class="sv-table"><thead><tr><th>Token</th><th>In lexicon?</th><th>Raw score</th><th>Role</th></tr></thead><tbody>${rows}</tbody></table>
        </div></div>
        <div class="sv-insight i-info">💡 Out-of-vocabulary (OOV) words are simply ignored in lexicon methods. Transformers handle OOV via subword splits — no word is truly unseen.</div>
      </div>`;
  } else if (step === 3) {
    const posT = scored.filter(t => t.finalScore > 0);
    const negT = scored.filter(t => t.finalScore < 0);
    const posSum = posT.reduce((a, t) => a + t.finalScore, 0);
    const negSum = negT.reduce((a, t) => a + t.finalScore, 0);
    const posRows = posT.length
      ? posT.map(t => `<div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;"><span class="sv-mono">${escapeHtml(t.word)}</span><span style="color:var(--pos);font-weight:600;">+${t.finalScore.toFixed(1)}</span></div>`).join("") +
        `<div style="border-top:0.5px solid var(--color-border-tertiary,#dddbd2);margin-top:5px;padding-top:5px;font-weight:600;font-size:12px;color:var(--pos);">Subtotal: +${posSum.toFixed(1)}</div>`
      : `<span style="font-size:12px;color:var(--color-text-tertiary,#8a8878);">None</span>`;
    const negRows = negT.length
      ? negT.map(t => `<div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;"><span class="sv-mono">${escapeHtml(t.word)}</span><span style="color:var(--neg);font-weight:600;">${t.finalScore.toFixed(1)}</span></div>`).join("") +
        `<div style="border-top:0.5px solid var(--color-border-tertiary,#dddbd2);margin-top:5px;padding-top:5px;font-weight:600;font-size:12px;color:var(--neg);">Subtotal: ${negSum.toFixed(1)}</div>`
      : `<span style="font-size:12px;color:var(--color-text-tertiary,#8a8878);">None</span>`;
    return `
      <div class="sv-card">
        <div class="sv-badge b-blue">Step 4 — Score accumulation</div>
        <p class="sv-desc">Sum all token scores (with negation and intensifier adjustments) into a single compound score.</p>
        <div class="grid2">
          <div class="sblock"><div class="sblock-lbl">Positive contributions</div>${posRows}</div>
          <div class="sblock"><div class="sblock-lbl">Negative contributions</div>${negRows}</div>
        </div>
        <div class="sblock" style="text-align:center;">
          <div class="sblock-lbl">Compound score</div>
          <div style="font-family:var(--disp);font-size:32px;font-weight:600;color:${total >= 0 ? "var(--pos)" : "var(--neg)"};margin-top:4px;">${total >= 0 ? "+" : ""}${total.toFixed(2)}</div>
        </div>
        <div class="sv-insight i-info">💡 Some implementations normalize to [-1, +1] using tanh: score_norm = tanh(score / 10). This prevents extreme scores for long texts.</div>
      </div>`;
  } else {
    return `
      <div class="sv-card">
        <div class="sv-badge b-pos">Step 5 — Classification</div>
        <div style="text-align:center;padding:18px 0 14px;">
          <div style="font-family:var(--disp);font-size:26px;font-weight:600;color:${sentiment.color};">${sentiment.label}</div>
          <div style="font-size:12px;color:var(--color-text-tertiary,#8a8878);margin-top:5px;">for: <em>"${escapeHtml(text)}"</em></div>
        </div>
        <div class="grid3">
          <div class="metric"><div class="metric-lbl">Compound score</div><div class="metric-val" style="color:${total >= 0 ? "var(--pos)" : "var(--neg)"};">${total >= 0 ? "+" : ""}${total.toFixed(2)}</div></div>
          <div class="metric"><div class="metric-lbl">Positive tokens</div><div class="metric-val" style="color:var(--pos);">${scored.filter(t => t.finalScore > 0).length}</div></div>
          <div class="metric"><div class="metric-lbl">Negative tokens</div><div class="metric-val" style="color:var(--neg);">${scored.filter(t => t.finalScore < 0).length}</div></div>
        </div>
        <div class="sblock">
          <div class="sblock-lbl">Decision rule</div>
          <div class="tl-row"><div class="tl-dot" style="background:var(--pos);"></div><div class="tl-content">Score &gt; +1.0 → Positive</div></div>
          <div class="tl-row"><div class="tl-dot" style="background:var(--neg);"></div><div class="tl-content">Score &lt; -1.0 → Negative</div></div>
          <div class="tl-row"><div class="tl-dot" style="background:#534AB7;"></div><div class="tl-content">-1.0 ≤ Score ≤ +1.0 → Neutral / Mixed</div></div>
        </div>
        <div class="sv-insight i-pos">✓ Lexicon analysis complete. No training data required — but domain-specific slang, sarcasm, and context remain challenging.</div>
      </div>`;
  }
}

function renderMlStep(step, data) {
  const { text, tokens } = data;
  const words = tokens.map(t => t.word);
  const posWords = ["fantastic","great","love","amazing","excellent","good","fast","helpful"];
  const negWords = ["terrible","slow","poor","broken","hate","fail","useless","disappointing"];
  const posF = words.filter(w => posWords.includes(w)).length;
  const negF = words.filter(w => negWords.includes(w)).length;
  const stopwords = new Set(["the","a","an","is","was","and","or","but","for","in","on","at","to","of","it","i","he","she","they","we"]);

  if (step === 0) {
    const filtered = words.filter(w => !stopwords.has(w));
    return `
      <div class="sv-card">
        <div class="sv-badge b-blue">Step 1 — Preprocessing</div>
        <p class="sv-desc">For ML models, we clean text and optionally remove stopwords, keeping content-bearing words.</p>
        <div class="sblock"><div class="sblock-lbl">Raw input</div><div style="font-size:13px;margin-top:4px;">${escapeHtml(text)}</div></div>
        <div class="sblock" style="margin-top:8px;"><div class="sblock-lbl">After stopword removal</div>
          <div class="word-grid">${filtered.map(w => `<span class="chip c-gray c-mono">${escapeHtml(w)}</span>`).join("")}</div>
        </div>
        <div class="sv-insight i-info">💡 Stopwords ('the', 'a', 'is') carry little sentiment signal. Removing them reduces feature space and often improves accuracy.</div>
      </div>`;
  } else if (step === 1) {
    const rows = [...posWords, ...negWords].map(w => {
      const count = words.filter(x => x === w).length;
      return `<div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;align-items:center;">
        <span class="sv-mono">${escapeHtml(w)}</span>
        <div style="flex:1;margin:0 8px;"><div class="sbar-track"><div class="sbar-fill" style="width:${count * 100}%;background:${posWords.includes(w) ? "var(--pos)" : "var(--neg)"};"></div></div></div>
        <span class="sv-mono">${count}</span>
      </div>`;
    }).join("");
    return `
      <div class="sv-card">
        <div class="sv-badge b-blue">Step 2 — Bag-of-Words vectorization</div>
        <p class="sv-desc">The document is converted to a fixed-length vector where each dimension represents a vocabulary word and the value is its count.</p>
        <div class="sblock"><div class="sblock-lbl">Feature vector (selected dimensions)</div>${rows}</div>
        <div class="sv-insight i-info">💡 BoW ignores word order: "great service, not terrible" and "terrible service, not great" get identical vectors. This is a key limitation.</div>
      </div>`;
  } else if (step === 2) {
    return `
      <div class="sv-card">
        <div class="sv-badge b-blue">Step 3 — Feature extraction</div>
        <p class="sv-desc">Naive Bayes counts how many positive-class words and negative-class words appear in the document.</p>
        <div class="grid2">
          <div class="sblock"><div class="sblock-lbl">Positive-class features</div><div style="font-family:var(--disp);font-size:30px;font-weight:600;color:var(--pos);margin:6px 0;">${posF}</div><div style="font-size:11px;color:var(--color-text-tertiary,#8a8878);">${escapeHtml(words.filter(w => posWords.includes(w)).join(", ") || "none found")}</div></div>
          <div class="sblock"><div class="sblock-lbl">Negative-class features</div><div style="font-family:var(--disp);font-size:30px;font-weight:600;color:var(--neg);margin:6px 0;">${negF}</div><div style="font-size:11px;color:var(--color-text-tertiary,#8a8878);">${escapeHtml(words.filter(w => negWords.includes(w)).join(", ") || "none found")}</div></div>
        </div>
        <div class="sv-insight i-info">💡 In real Naive Bayes: P(Positive|doc) ∝ P(Positive) × ∏P(word|Positive). Laplace smoothing ensures zero-count words don't collapse the product.</div>
      </div>`;
  } else if (step === 3) {
    const posW_ = Math.min((posF / (posF + negF + 0.001)) * 100, 100);
    const negW_ = Math.min((negF / (posF + negF + 0.001)) * 100, 100);
    const posVal = (-2.3 + posF * 0.4).toFixed(2);
    const negVal = (-2.3 + negF * 0.4).toFixed(2);
    return `
      <div class="sv-card">
        <div class="sv-badge b-blue">Step 4 — Naive Bayes scoring</div>
        <p class="sv-desc">Compute log-probability for each class. The class with the higher log-probability wins.</p>
        <div class="sblock"><div class="sblock-lbl">Simulated log-probabilities</div>
          <div style="display:flex;align-items:center;gap:10px;margin:8px 0;"><span style="font-size:12px;font-weight:600;color:var(--pos);min-width:80px;">Positive</span><div class="sbar-track"><div class="sbar-fill" style="width:${posW_}%;background:var(--pos);"></div></div><span class="sv-mono" style="color:var(--pos);">${posVal}</span></div>
          <div style="display:flex;align-items:center;gap:10px;margin:8px 0;"><span style="font-size:12px;font-weight:600;color:var(--neg);min-width:80px;">Negative</span><div class="sbar-track"><div class="sbar-fill" style="width:${negW_}%;background:var(--neg);"></div></div><span class="sv-mono" style="color:var(--neg);">${negVal}</span></div>
        </div>
        <div class="sv-insight i-cmp">⚡ Log-space addition prevents numerical underflow — multiplying thousands of small probabilities would round to 0.0. log P(A×B) = log P(A) + log P(B).</div>
      </div>`;
  } else {
    const winner = posF >= negF ? "Positive" : "Negative";
    const winnerColor = posF >= negF ? "var(--pos)" : "var(--neg)";
    return `
      <div class="sv-card">
        <div class="sv-badge b-pos">Step 5 — Classification</div>
        <div style="text-align:center;padding:18px 0 14px;">
          <div style="font-family:var(--disp);font-size:26px;font-weight:600;color:${winnerColor};">${winner} Sentiment</div>
          <div style="font-size:12px;color:var(--color-text-tertiary,#8a8878);margin-top:5px;">Naive Bayes decision: class with highest log-probability</div>
        </div>
        <div class="grid2">
          <div class="metric"><div class="metric-lbl">Positive signal</div><div class="metric-val" style="color:var(--pos);">${posF} words</div></div>
          <div class="metric"><div class="metric-lbl">Negative signal</div><div class="metric-val" style="color:var(--neg);">${negF} words</div></div>
        </div>
        <div class="sv-insight i-info">💡 ML models like Naive Bayes, SVM, and logistic regression require labeled training data. They still struggle with sarcasm, context, and domain shift.</div>
      </div>`;
  }
}

function renderTransformerStep(step, data) {
  const { text, scored, total, sentiment } = data;
  const toks_ = text.split(/\s+/);
  const subwords = ["[CLS]", ...toks_.flatMap(w => w.length > 6 ? [w.slice(0, 4), "##" + w.slice(4) + "[subword]"] : [w]), "[SEP]"];

  if (step === 0) {
    const chips = subwords.map((t, i) => `<span class="chip ${i === 0 || i === subwords.length - 1 ? "c-neu" : t.startsWith("##") ? "c-cmp" : "c-gray"} c-mono">${escapeHtml(t)}</span>`).join("");
    return `
      <div class="sv-card">
        <div class="sv-badge b-neu">Step 1 — Subword tokenization</div>
        <p class="sv-desc">BERT uses WordPiece tokenization. Long words are split into subwords; every sequence starts with [CLS] and ends with [SEP].</p>
        <div class="word-grid">${chips}</div>
        <div class="grid2">
          <div class="metric"><div class="metric-lbl">Input words</div><div class="metric-val">${toks_.length}</div></div>
          <div class="metric"><div class="metric-lbl">Subword tokens</div><div class="metric-val">${subwords.length}</div></div>
        </div>
        <div class="sv-insight i-neu">💡 [CLS] is the classification token — its final hidden state is used as the sentence representation. No word is truly OOV.</div>
      </div>`;
  } else if (step === 1) {
    return `
      <div class="sv-card">
        <div class="sv-badge b-neu">Step 2 — Token embeddings</div>
        <p class="sv-desc">Each token is mapped to a 768-dimensional embedding vector (sum of token + position + segment embeddings).</p>
        <div class="sblock"><div class="sblock-lbl">Embedding components</div>
          <div class="tl-row"><div class="tl-dot" style="background:#534AB7;"></div><div class="tl-content"><strong>Token embedding</strong>: 768-dim lookup from 30,000-word vocabulary</div></div>
          <div class="tl-row"><div class="tl-dot" style="background:#0e7490;"></div><div class="tl-content"><strong>Position embedding</strong>: encodes token position (0…511)</div></div>
          <div class="tl-row"><div class="tl-dot" style="background:#854f0b;"></div><div class="tl-content"><strong>Segment embedding</strong>: marks sentence A vs sentence B (for pairs)</div></div>
        </div>
        <div class="sblock" style="margin-top:8px;"><div class="sblock-lbl">Input matrix shape</div><div style="font-family:var(--mono);font-size:13px;margin-top:4px;">[${subwords.length} tokens × 768 dims] = ${(subwords.length * 768).toLocaleString()} values</div></div>
        <div class="sv-insight i-neu">💡 Position embeddings are critical — without them, "good not bad" and "bad not good" would produce identical representations.</div>
      </div>`;
  } else if (step === 2) {
    return `
      <div class="sv-card">
        <div class="sv-badge b-neu">Step 3 — Multi-head self-attention</div>
        <p class="sv-desc">12 attention heads learn different relationships. Each token attends to every other token — enabling long-range dependency capture.</p>
        <div class="sblock"><div class="sblock-lbl">Attention formula</div><code style="display:block;padding:8px;font-size:11.5px;line-height:1.8;">Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V</code></div>
        <div class="grid3" style="margin-top:8px;">
          <div class="metric"><div class="metric-lbl">Attention heads</div><div class="metric-val">12</div></div>
          <div class="metric"><div class="metric-lbl">Transformer layers</div><div class="metric-val">12</div></div>
          <div class="metric"><div class="metric-lbl">Parameters</div><div class="metric-val" style="font-size:18px;">110M</div></div>
        </div>
        <div class="sv-insight i-neu">💡 Different heads learn different patterns: one head might track negation ("not" attending to "good"), another tracks subject-verb agreement.</div>
      </div>`;
  } else if (step === 3) {
    const posW = scored.length ? Math.max(30, Math.min(90, 50 + total * 5)) : 50;
    const negW = scored.length ? Math.max(10, Math.min(70, 50 - total * 5)) : 50;
    const posVal = scored.length ? ((50 + total * 5) > 50 ? "3.24" : "1.12") : "2.10";
    const negVal = scored.length ? ((50 - total * 5) > 50 ? "3.24" : "1.12") : "2.10";
    return `
      <div class="sv-card">
        <div class="sv-badge b-neu">Step 4 — [CLS] pooling</div>
        <p class="sv-desc">After 12 transformer layers, the [CLS] token's final hidden state serves as the sentence-level representation — a 768-dim vector encoding the full context.</p>
        <div class="sblock"><div class="sblock-lbl">[CLS] hidden state → classification head</div><div style="font-family:var(--mono);font-size:12px;color:var(--color-text-secondary,#5a5950);margin-top:4px;">[CLS] h⃗ ∈ ℝ⁷⁶⁸ → Dense(768→2) → softmax → [P_pos, P_neg]</div></div>
        <div class="sblock" style="margin-top:8px;"><div class="sblock-lbl">Simulated logits</div>
          <div class="sbar-row"><div class="sbar-label" style="color:var(--pos);">Positive</div><div class="sbar-track"><div class="sbar-fill" style="width:${posW}%;background:var(--pos);"></div></div><div class="sbar-val">${posVal}</div></div>
          <div class="sbar-row"><div class="sbar-label" style="color:var(--neg);">Negative</div><div class="sbar-track"><div class="sbar-fill" style="width:${negW}%;background:var(--neg);"></div></div><div class="sbar-val">${negVal}</div></div>
        </div>
        <div class="sv-insight i-neu">💡 For fine-tuned BERT, only the classification head weights are randomly initialized. The 12-layer transformer is pre-trained on 3.3B words.</div>
      </div>`;
  } else {
    return `
      <div class="sv-card">
        <div class="sv-badge b-pos">Step 5 — Softmax prediction</div>
        <div style="text-align:center;padding:18px 0 14px;">
          <div style="font-family:var(--disp);font-size:26px;font-weight:600;color:${sentiment.color};">${sentiment.label}</div>
          <div style="font-size:12px;color:var(--color-text-tertiary,#8a8878);margin-top:5px;">BERT-style prediction via softmax over logits</div>
        </div>
        <div class="sblock"><div class="sblock-lbl">Why transformers win on nuance</div>
          <div class="tl-row"><div class="tl-dot" style="background:#534AB7;"></div><div class="tl-content">Bidirectional context: "not bad" understood via full sentence attention</div></div>
          <div class="tl-row"><div class="tl-dot" style="background:#534AB7;"></div><div class="tl-content">Sarcasm cues: punctuation, capitalization, surrounding context all encoded</div></div>
          <div class="tl-row"><div class="tl-dot" style="background:#534AB7;"></div><div class="tl-content">Domain adaptation: fine-tune on domain-specific reviews in minutes</div></div>
        </div>
        <div class="sv-insight i-neu">💡 State-of-the-art: fine-tuned BERT achieves ~95% accuracy on SST-2 (Stanford Sentiment Treebank) vs ~83% for SVM and ~75% for VADER lexicon.</div>
      </div>`;
  }
}

// ─── TAB: COMPARISON ─────────────────────────────────────────────────────
function mountComparison() {
  const panel = document.getElementById("panel-cmp");
  panel.innerHTML = `
    <div class="sv-eyebrow">Same Sentence, Three Opinions</div>
    <div class="sv-card">
      <div class="sv-badge b-cmp">Model comparison</div>
      <div class="sv-card-title">Three Approaches <em>side by side</em></div>
      <p class="sv-desc">Type a sentence below and see how three different kinds of models score it — they don't always agree.</p>
      <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
        <input class="sv-input" id="cmpInputField">
      </div>
      <div class="grid3c" id="cmpResults" style="margin-bottom:10px;"></div>
      <div class="sv-insight i-info">💡 All three models see the same text but may disagree — especially on subtle, mixed, or domain-specific language. Transformer models tend to be most robust but require training data and compute.</div>
    </div>

    <div class="sv-card">
      <div class="sv-card-title">Approach overview <em>— when to use what</em></div>
      <div class="grid3c" id="cmpApproachOverview"></div>
    </div>

    <div class="sv-card">
      <div class="sv-card-title">Edge cases <em>— where models disagree</em></div>
      <p class="sv-desc">Click a sentence to load it into the comparison above.</p>
      <div id="cmpEdgeCases"></div>
    </div>
  `;
  const input = document.getElementById("cmpInputField");
  input.value = STATE.cmp.cmpInput;
  input.addEventListener("input", e => { STATE.cmp.cmpInput = e.target.value; updateCmpResults(); });
  renderCmpApproachOverview();
  renderCmpEdgeCases();
  updateCmpResults();
}

function updateCmpResults() {
  const text = STATE.cmp.cmpInput || "The service was okay but could definitely be better";
  const tokens = analyzeTokens(text);
  const scored = tokens.filter(t => t.finalScore !== 0);
  const total = scored.reduce((a, t) => a + t.finalScore, 0);
  const lx = getSentimentLabel(total, scored.length);
  const posW = tokens.filter(t => t.finalScore > 0).length;
  const negW = tokens.filter(t => t.finalScore < 0).length;
  const mlScore = posW - negW;
  const ml = getSentimentLabel(mlScore, posW + negW);
  const trScore = total * 0.9 - 0.2;
  const tr = getSentimentLabel(trScore, scored.length);
  const makeBarW = score => Math.max(0, Math.min(100, 50 + score * 10));

  const models = [
    { color:"#185FA5", badge:'<div class="sv-badge b-blue">Lexicon (VADER-style)</div>', sentiment:lx, score:total, extra:`${posW} pos tokens, ${negW} neg tokens`, scoreLabel:`Score: ${total >= 0 ? "+" : ""}${total.toFixed(2)}` },
    { color:"#3B6D11", badge:'<div class="sv-badge" style="background:#EAF3DE;color:#27500A;border:0.5px solid #97C459;">Naive Bayes (ML)</div>', sentiment:ml, score:mlScore, extra:"Feature matching on trained vocab", scoreLabel:`Log-odds: ${mlScore >= 0 ? "+" : ""}${mlScore.toFixed(2)}` },
    { color:"#534AB7", badge:'<div class="sv-badge b-neu">Transformer (BERT)</div>', sentiment:tr, score:trScore, extra:"Bidirectional context, fine-tuned", scoreLabel:`Confidence: ${Math.round(60 + Math.abs(trScore) * 5)}%` },
  ];

  document.getElementById("cmpResults").innerHTML = models.map(m => `
    <div class="cmp-card" style="border-top:2px solid ${m.color};">
      ${m.badge}
      <div style="font-family:var(--disp);font-size:20px;font-weight:600;color:${m.sentiment.color};margin:6px 0;">${m.sentiment.label}</div>
      <div style="font-size:11px;color:var(--color-text-tertiary,#8a8878);margin-bottom:8px;">${m.scoreLabel}</div>
      <div class="meter-track"><div class="meter-fill" style="width:${makeBarW(m.score)}%;background:${m.color};"></div></div>
      <div style="font-size:11px;color:var(--color-text-secondary,#5a5950);margin-top:8px;">${m.extra}</div>
    </div>`).join("");
}

function renderCmpApproachOverview() {
  document.getElementById("cmpApproachOverview").innerHTML = APPROACHES.map(a => `
    <div class="cmp-card">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
        <div style="width:32px;height:32px;border-radius:6px;background:${a.bg};border:0.5px solid ${a.bd};display:flex;align-items:center;justify-content:center;color:${a.color};font-size:16px;">◈</div>
        <div style="font-size:13px;font-weight:600;color:${a.color};">${escapeHtml(a.name)}</div>
      </div>
      <div style="font-size:11px;font-weight:600;color:var(--pos);margin-bottom:4px;text-transform:uppercase;letter-spacing:.3px;">Strengths</div>
      ${a.pros.map(p => `<div class="tl-row"><div class="tl-dot" style="background:var(--pos);"></div><div class="tl-content">${escapeHtml(p)}</div></div>`).join("")}
      <div style="font-size:11px;font-weight:600;color:var(--neg);margin:8px 0 4px;text-transform:uppercase;letter-spacing:.3px;">Limitations</div>
      ${a.cons.map(p => `<div class="tl-row"><div class="tl-dot" style="background:var(--neg);"></div><div class="tl-content">${escapeHtml(p)}</div></div>`).join("")}
      <div style="margin-top:8px;font-size:11px;background:${a.bg};color:${a.color};border:0.5px solid ${a.bd};border-radius:6px;padding:5px 8px;"><strong>Best for:</strong> ${escapeHtml(a.use)}</div>
    </div>`).join("");
}

function renderCmpEdgeCases() {
  document.getElementById("cmpEdgeCases").innerHTML = EDGE_CASES.map((e, i) => `
    <div class="sblock" style="margin-bottom:8px;cursor:pointer;" onclick="loadEdgeCase(${i})">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;">
        <div>
          <div style="font-size:13px;font-weight:500;margin-bottom:4px;">"${escapeHtml(e.text)}"</div>
          <div style="font-size:11.5px;color:var(--color-text-secondary,#5a5950);margin-bottom:6px;">${escapeHtml(e.note)}</div>
          <div style="display:flex;gap:5px;flex-wrap:wrap;">
            <span class="chip c-blue" style="font-size:10.5px;">Lexicon: ${escapeHtml(e.lx)}</span>
            <span class="chip" style="background:#EAF3DE;color:#27500A;border-color:#97C459;font-size:10.5px;">ML: ${escapeHtml(e.ml)}</span>
            <span class="chip c-neu" style="font-size:10.5px;">Transformer: ${escapeHtml(e.tr)}</span>
          </div>
        </div>
        <span style="font-size:11px;color:var(--color-text-tertiary,#8a8878);white-space:nowrap;margin-top:2px;">click to test →</span>
      </div>
    </div>`).join("");
}

function loadEdgeCase(i) {
  STATE.cmp.cmpInput = EDGE_CASES[i].text;
  document.getElementById("cmpInputField").value = STATE.cmp.cmpInput;
  updateCmpResults();
}
window.loadEdgeCase = loadEdgeCase;

// ─── TAB: QUIZ ───────────────────────────────────────────────────────────
function quizTagStyle(tag) {
  const map = {
    Basics: { bg:"#fbe9e0", color:"#9c4221", bd:"#e8a685" },
    Lexicon: { bg:"#E6F1FB", color:"#0C447C", bd:"#85B7EB" },
    ML: { bg:"#EAF3DE", color:"#27500A", bd:"#97C459" },
    Transformer: { bg:"#EEEDFE", color:"#3C3489", bd:"#AFA9EC" },
    General: { bg:"#faeeda", color:"#633806", bd:"#ef9f27" },
  };
  return map[tag] || map.General;
}

function mountQuiz() {
  const panel = document.getElementById("panel-quiz");
  panel.innerHTML = `
    <div class="sv-eyebrow">No Pressure, Just Practice</div>
    <div class="sv-card">
      <div class="sv-card-title">Concept quiz</div>
      <p class="sv-desc">Pick an answer for instant feedback and a short explanation — no scoring, just learning.</p>
      ${QUIZ.map((q, qi) => {
        const ts = quizTagStyle(q.tag);
        return `
        ${qi > 0 ? '<hr class="qsep">' : ""}
        <div data-qid="${q.id}">
          <div style="display:flex;gap:8px;margin-bottom:10px;align-items:flex-start;">
            <span class="sv-badge" style="background:${ts.bg};color:${ts.color};border:0.5px solid ${ts.bd};margin:2px 0 0;">${escapeHtml(q.tag)}</span>
            <div style="font-size:13.5px;font-weight:600;line-height:1.5;">${escapeHtml(q.q)}</div>
          </div>
          ${q.opts.map((opt, oi) => `<button class="qopt" id="qopt-${q.id}-${oi}" onclick="answerQuiz(${q.id},${oi},${q.ans})">${escapeHtml(opt)}</button>`).join("")}
          <div class="sv-insight" id="qfb-${q.id}" style="display:none;"></div>
        </div>`;
      }).join("")}
      <div style="text-align:center;margin-top:20px;">
        <button class="sv-btn btn-sec" onclick="resetQuiz()">↺ Reset quiz</button>
      </div>
    </div>
  `;
}

function answerQuiz(qid, oi, ansIdx) {
  const firstBtn = document.getElementById(`qopt-${qid}-0`);
  if (firstBtn.disabled) return;
  const q = QUIZ.find(x => x.id === qid);
  q.opts.forEach((opt, i) => {
    const btn = document.getElementById(`qopt-${qid}-${i}`);
    btn.disabled = true;
    if (i === ansIdx) btn.classList.add("correct");
    else if (i === oi) btn.classList.add("incorrect");
  });
  const fb = document.getElementById(`qfb-${qid}`);
  const correct = oi === ansIdx;
  fb.style.display = "block";
  fb.className = `sv-insight ${correct ? "i-pos" : "i-warn"}`;
  fb.innerHTML = `<strong>${correct ? "✓ Correct!" : "✗ Incorrect."}</strong> ${escapeHtml(q.exp)}`;
}
window.answerQuiz = answerQuiz;

function resetQuiz() {
  mountQuiz();
}
window.resetQuiz = resetQuiz;
