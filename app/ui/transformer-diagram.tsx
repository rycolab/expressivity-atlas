'use client';
import React, { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { MathJax } from 'better-react-mathjax';

// DAG definition (unchanged)
type NodeKey =
  | "embedding"
  | "query"
  | "key"
  | "value"
  | "dot"
  | "argmax"
  | "wsum"
  | "proj"
  | "addnorm1"
  | "ff_linear1"
  | "ff_gelu"
  | "ff_linear2"
  | "addnorm2"
  | "linear"

const DAG: Record<NodeKey, { parents: NodeKey[]; children: NodeKey[] }> = {
  embedding: { parents: [], children: ["query", "key", "value"] },
  query: { parents: ["embedding"], children: ["dot"] },
  key: { parents: ["embedding"], children: ["dot"] },
  value: { parents: ["embedding"], children: ["wsum"] },
  dot: { parents: ["query", "key"], children: ["argmax"] },
  argmax: { parents: ["dot"], children: ["wsum"] },
  wsum: { parents: ["argmax", "value"], children: ["proj"] },
  proj: { parents: ["wsum"], children: ["addnorm1"] },
  addnorm1: { parents: ["proj"], children: ["ff_linear1"] },
  ff_linear1: { parents: ["addnorm1"], children: ["ff_gelu"] },
  ff_gelu: { parents: ["ff_linear1"], children: ["ff_linear2"] },
  ff_linear2: { parents: ["ff_gelu"], children: ["addnorm2"] },
  addnorm2: { parents: ["ff_linear2"], children: ["linear"] },
  linear: { parents: ["addnorm2"], children: [] },
};

const nodeDisplay: Record<NodeKey, { label: string; color: string }> = {
  embedding: { label: "Embedding", color: "bg-blue-700" },
  query: { label: "Query projection", color: "bg-amber-600" },
  key: { label: "Key projection", color: "bg-amber-600" },
  value: { label: "Value projection", color: "bg-amber-600" },
  dot: { label: "Scaled Dot-Product", color: "bg-amber-600" },
  argmax: { label: "Argmax", color: "bg-amber-600" },
  wsum: { label: "Weighted sum", color: "bg-amber-600" },
  proj: { label: "Projection", color: "bg-amber-600" },
  addnorm1: { label: "Add & Norm", color: "bg-slate-600" },
  ff_linear1: { label: "Linear", color: "bg-lime-600" },
  ff_gelu: { label: "GELU", color: "bg-lime-600" },
  ff_linear2: { label: "Linear", color: "bg-lime-600" },
  addnorm2: { label: "Add & Norm", color: "bg-slate-600" },
  linear: { label: "Linear", color: "bg-rose-600" },
};

const nodeLogics: Record<NodeKey, React.ReactNode> = {
  embedding: (
    <MathJax>
      {`$$
        \\begin{align}
        E_{-11.5}(x) &= \\pi_a(x) \\\\
        E_{0.7}(x) &= \\pi_b(x) \\\\
        E_{0}(x) &= \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  query: (
    <MathJax>
      {`$$
        \\begin{align}
        Q_{-1.5}(x) &= E_{-11.5}(x) \\\\
        &= \\pi_a(x) \\\\
        Q_{-1.1}(x) &= E_{0.7}(x) \\lor E_{0}(x) \\\\
        &= \\pi_b(x) \\lor \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  key: (
    <MathJax>
      {`$$
        \\begin{align}
        K_{-0.3}(x) &= E_{-11.5}(x) \\\\
        &= \\pi_a(x) \\\\
        K_{-0.1}(x) &= E_{0.7}(x) \\lor E_{0}(x) \\\\
        &= \\pi_b(x) \\lor \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  value: (
    <MathJax>
      {`$$
        \\begin{align}
        V_{-9.7}(x) &= E_{-11.5}(x) \\\\
        &= \\pi_a(x) \\\\
        V_{2.3}(x) &= E_{0}(x) \\\\
        &= \\pi_c(x) \\\\
        V_{2.8}(x) &= E_{0.7}(x) \\\\
        &= \\pi_b(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  dot: (
    <MathJax>
      {`$$
        \\begin{align}
        D_{0.1}(x,y) &= Q_{-1.1}(x) \\land K_{-0.1}(y) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        D_{0.2}(x,y) &= Q_{-1.5}(x) \\land K_{-0.1}(y) \\\\
        &= \\pi_a(x) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        D_{0.3}(x,y) &= Q_{-1.1}(x) \\land K_{-0.3}(y) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(y) \\\\
        D_{0.5}(x,y) &= Q_{-1.5}(x) \\land K_{-0.3}(y) \\\\
        &= \\pi_a(x) \\land \\pi_a(y)
        \\end{align}
      $$`}
    </MathJax>
  ),
  argmax: (
    <MathJax>
      {`$$
        \\begin{align}
        \\text{argmax}_{0.5}(x, y) &= D_{0.5}(x,y)\\\\
        &= \\pi_a(x) \\land \\pi_a(y) \\\\
        \\text{argmax}_{0.3}(x, y) &= D_{0.3}(x,y) \\land \\lnot \\exists z\\leq x:D_{0.5}(x,z) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(y) \\land \\lnot \\exists z \\leq x:(\\pi_a(x) \\land \\pi_a(z)) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(y) \\\\
        \\text{argmax}_{0.2}(x, y) &= D_{0.2}(x,y) \\land \\lnot \\exists z\\leq x:(D_{0.5}(x,z)\\lor D_{0.3}(x,z)) \\\\
        &= \\pi_a(x) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        &\\quad \\land \\lnot \\exists z \\leq x:(\\pi_a(x) \\land \\pi_a(z) \\lor (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(z) ) \\\\
        &=\\false \\\\
        \\text{argmax}_{0.1}(x, y) &= D_{0.1}(x,y) \\land \\lnot \\exists z\\leq x:(D_{0.5}(x,z)\\lor D_{0.3}(x,z)\\lor D_{0.2}(x,z)) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        &\\quad \\land \\lnot \\exists z \\leq x:(\\pi_a(x) \\land \\pi_a(z) \\lor (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(z)  \\\\
        &\\quad\\quad\\quad\\quad\\quad\\quad \\lor \\pi_a(x) \\land (\\pi_b(z) \\lor \\pi_c(z)) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z)  \\\\
        \\text{argmax}(x, y) &= y\\leq x \\land (\\text{argmax}_{0.5}(x, y) \\lor \\text{argmax}_{0.3}(x, y) \\\\
        &\\quad\\quad\\quad\\quad\\quad \\lor \\text{argmax}_{0.2}(x, y) \\lor \\text{argmax}_{0.1}(x, y)) \\\\
        &= y\\leq x \\land (\\pi_a(y) \\lor (\\pi_b(x) \\lor \\pi_c(x)) \\land \\lnot \\exists z \\leq x:\\pi_a(z))
        \\end{align}
      $$`}
    </MathJax>
  ),
  wsum: (
    <MathJax>
      {`$$
        \\begin{align}
        S_{-9.7}(x)&=\\exists y\\leq x: (\\text{argmax}(x, y) \\land V_{-9.7}(y)) \\\\
        &= \\exists y\\leq x: (\\text{argmax}(x, y) \\land \\pi_a(y)) \\\\
        &= \\exists y\\leq x: \\pi_a(y) \\\\
        S_{2.3}(x)&=\\exists y\\leq x: (\\text{argmax}(x, y) \\land V_{2.3}(y)) \\\\
        &= \\exists y\\leq x: (\\text{argmax}(x, y) \\land \\pi_c(y)) \\\\
        &= \\exists y\\leq x: \\pi_c(y) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        S_{2.8}(x)&=\\exists y\\leq x: (\\text{argmax}(x, y) \\land V_{2.8}(y)) \\\\
        &= \\exists y\\leq x: (\\text{argmax}(x, y) \\land \\pi_b(y)) \\\\
        &= \\exists y\\leq x: \\pi_b(y) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  proj: (
    <MathJax>
      {`$$
        \\begin{align}
        A_{-20.8}(x)&=S_{-9.7}(x) \\\\
        &= \\exists y\\leq x: \\pi_a(y) \\\\
        A_{6.1}(x)&=S_{2.3}(x) \\\\
        &= \\exists y\\leq x: \\pi_c(y) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        A_{8.1}(x)&=S_{2.8}(x) \\\\
        &= \\exists y\\leq x: \\pi_b(y) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  addnorm1: (
    <MathJax>
      {`$$
        \\begin{align}
        N_{-32.3}(x) &= E_{-11.5}(x)\\land A_{-20.8}(x) \\\\
        &= \\pi_a(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        &= \\pi_a(x) \\\\
        N_{-20.8}(x) &= E_{0}(x) \\land A_{-20.8}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        N_{-20.1}(x) &= E_{0.7}(x) \\land A_{-20.8}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        N_{6.1}(x) &= E_{0}(x) \\land A_{6.1}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: \\pi_c(y) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        &= \\pi_c(x) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        N_{6.8}(x) &= E_{0.7}(x) \\land A_{6.1}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: \\pi_c(y) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        N_{8.1}(x) &= E_{0}(x) \\land A_{8.1}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: \\pi_b(y) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        N_{8.8}(x) &= E_{0.7}(x) \\land A_{8.1}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: \\pi_b(y) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        &= \\pi_b(x) \\land \\lnot \\exists z \\leq x:\\pi_a(z)
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_linear1: (
    <MathJax>
      {`$$
        \\begin{align}
        L_{[-16.0, 156.5, 0, 0]}(x) &= N_{-20.8}(x) \\lor N_{-20.1}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        L_{[-16.0, 234.8, 78.3, 0]}(x) &= N_{-32.3}(x) \\\\
        &= \\pi_a(x) \\\\
        L_{[ 0, -63.9, -16.0, -16.0]}(x) &= N_{8.1}(x) \\lor N_{8.8}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x) \\land \\exists y\\leq x: \\pi_b(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        L_{[ 0, -48.0, -16.0, 0]}(x) &= N_{6.1}(x) \\lor N_{6.8}(x) \\\\
        &= (\\pi_c(x) \\lor \\pi_b(x) \\land \\exists y\\leq x: \\pi_c(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_gelu: (
    <MathJax>
      {`$$
        \\begin{align}
        G_{[0, 0, 0, 0]}(x) &= L_{[ 0, -63.9, -16.0, -16.0]}(x) \\lor L_{[ 0, -48.0, -16.0, 0]}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\lnot \\exists z\\leq x: \\pi_a(z) \\\\
        &= \\lnot \\exists z\\leq x: \\pi_a(z) \\\\
        G_{[0, 167.7, 0, 0]}(x) &= L_{[-16.0, 156.5, 0, 0]}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        G_{[ 0, 234.8, 67.1, 0]}(x) &= L_{[-16.0, 234.8, 78.3, 0]}(x) \\\\
        &= \\pi_a(x) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_linear2: (
    <MathJax>
      {`$$
        \\begin{align}
        F_{-67.6}(x) &= G_{[ 0, 234.8, 67.1, 0]}(x) \\\\
        &= \\pi_a(x) \\\\
        F_{-45.1}(x) &= G_{[0, 167.7, 0, 0]}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        F_{0.7}(x) &= G_{[0, 0, 0, 0]}(x) \\\\
        &= \\lnot \\exists z\\leq x: \\pi_a(z) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  addnorm2: (
    <MathJax>
      {`$$
        \\begin{align}
        R_{-99.9}(x) &= N_{-32.3}(x)\\land F_{-67.6}(x)\\\\
        &= \\pi_a(x) \\\\
        R_{-66.6}(x) &= N_{-20.8}(x) \\land F_{-45.1}(x) \\lor N_{-20.1}(x)\\land F_{-45.1}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        R_{7.1}(x) &= N_{6.1}(x) \\land F_{0.7}(x) \\lor N_{6.8}(x)\\land F_{0.7}(x) \\\\
        &= (\\pi_c(x) \\lor \\pi_b(x) \\land \\exists y\\leq x: \\pi_c(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        R_{9.4}(x) &= N_{8.1}(x) \\land F_{0.7}(x) \\lor N_{8.8}(x)\\land F_{0.7}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x) \\land \\exists y\\leq x: \\pi_b(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z)
        \\end{align}
      $$`}
    </MathJax>
  ),
  linear: (
    <MathJax>
      {`$$
        \\begin{align}
        O_{-5.3}(x) &= R_{9.4}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x) \\land \\exists y\\leq x: \\pi_b(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        O_{-4.0}(x) &= R_{7.1}(x) \\\\
        &= (\\pi_c(x) \\lor \\pi_b(x) \\land \\exists y\\leq x: \\pi_c(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        O_{-5.3}(x) \\lor O_{-4.0}(x) &= \\lnot \\exists z \\leq x:\\pi_a(z) \\\\\
        O_{31.6}(x) &= R_{-66.6}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        O_{47.4}(x) &= R_{-99.9}(x) \\\\
        O_{31.6}(x) \\lor O_{47.4}(x) &= \\pi_a(x) \\lor (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        &= \\pi_a(x) \\lor \\exists y\\leq x: \\pi_a(y) \\\\
        &= \\exists y\\leq x: \\pi_a(y)
        \\end{align}
      $$`}
    </MathJax>
  ),
};


// displayOrder remains unchanged
const displayOrder: NodeKey[] = [
  "embedding",
  "query", "key", "value",
  "dot", "argmax", "wsum", "proj",
  "addnorm1",
  "ff_linear1", "ff_gelu", "ff_linear2",
  "addnorm2",
  "linear",
];

// *** Responsive TokenBox: font size and box size decrease on small screens
const TokenBox = ({ token }: { token: string }) => (
  <div
    className={`
      border rounded-xl shadow-sm bg-white text-center
      flex items-center justify-center
      transition-all
      w-7 h-7 text-[8px]
      xs:w-8 xs:h-8 xs:text-[9px]
      sm:w-10 sm:h-10 sm:text-[10px]
      md:w-12 md:h-12 md:text-[11px]
      lg:w-14 lg:h-14 lg:text-[12px]
      max-w-[2.75rem] max-h-[2.75rem]
      break-all
    `}
    style={{
      minWidth: '1.75rem',
      minHeight: '1.75rem',
    }}
  >
    {token}
  </div>
);


const TokenMatrix = ({ tokens }: { tokens: string[][] }) => (
  <div className="flex flex-col gap-[2px] my-2">
    {tokens.map((row, i) => (
      <div className="flex gap-[2px] justify-center" key={i}>
        {row.map((t, j) => (
          <TokenBox key={j} token={t || ""} />
        ))}
      </div>
    ))}
  </div>
);

// LayerBox and SubLayerBox unchanged except for font responsiveness if needed
const LayerBox = ({
  color,
  label,
  onPrev,
  onNext,
  children,
}: {
  color: string;
  label: string;
  onPrev?: () => void;
  onNext?: () => void;
  children?: React.ReactNode;
}) => (
  <div
    className={`relative rounded-2xl shadow-md p-3 sm:p-4 text-center w-full max-w-4xl mx-auto ${color} bg-opacity-90`}
  >
    {onPrev && (
      <button
        className="absolute left-0 top-1/2 -translate-y-1/2 ml-2 text-white text-xs bg-white/20 px-2 py-1 rounded hover:bg-white/30"
        onClick={onPrev}
        aria-label="Deactivate and hide this layer"
      >
        ▲
      </button>
    )}
    {onNext && (
      <button
        className="absolute right-0 top-1/2 -translate-y-1/2 mr-2 text-white text-xs bg-white/20 px-2 py-1 rounded hover:bg-white/30"
        onClick={onNext}
        aria-label="Activate and show this layer"
      >
        ▼
      </button>
    )}
    <div className="text-lg font-semibold text-white mb-2">{label}</div>
    {children && <div>{children}</div>}
  </div>
);

const SubLayerBox = ({
  color,
  label,
  onPrev,
  onNext,
  children,
}: {
  color: string;
  label: string;
  onPrev?: () => void;
  onNext?: () => void;
  children?: React.ReactNode;
}) => (
  <div
    className={`relative rounded-xl shadow p-2 flex flex-col items-center justify-center w-full ${color} bg-opacity-80`}
  >
    {onPrev && (
      <button
        className="absolute left-0 top-1/2 -translate-y-1/2 ml-2 text-white text-[10px] bg-white/20 px-2 py-1 rounded hover:bg-white/30"
        onClick={onPrev}
        aria-label="Deactivate and hide this sublayer"
      >
        ▲
      </button>
    )}
    {onNext && (
      <button
        className="absolute right-0 top-1/2 -translate-y-1/2 mr-2 text-white text-[10px] bg-white/20 px-2 py-1 rounded hover:bg-white/30"
        onClick={onNext}
        aria-label="Activate and show this sublayer"
      >
        ▼
      </button>
    )}
    <span className="text-xs font-semibold text-white mb-1">{label}</span>
    {children}
  </div>
);

const vocabulary: Record<string, number> = {
  b: 0.7,
  a: -11.5,
  c: 0,
};
const makeMatrix = (rows: number, cols: number, prefix: string) =>
  Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => prefix ? `${prefix}${i + 1}${j + 1}` : "")
  );

export default function TransformerDiagram() {
  const inputTokens = ["b", "a", "c", "a", "c"];

  const [nodeActive, setNodeActive] = useState<Record<NodeKey, boolean>>({
    embedding: false,
    query: false,
    key: false,
    value: false,
    dot: false,
    argmax: false,
    wsum: false,
    proj: false,
    addnorm1: false,
    ff_linear1: false,
    ff_gelu: false,
    ff_linear2: false,
    addnorm2: false,
    linear: false
  });

  const [latestActiveKey, setLatestActiveKey] = useState<NodeKey | null>(null);
  const logicRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const logicContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const activeKeys = displayOrder.filter(k => nodeActive[k]);
    if (activeKeys.length > 0) {
      setLatestActiveKey(activeKeys[activeKeys.length - 1]);
    }
  }, [nodeActive]);

  useEffect(() => {
    if (!latestActiveKey) return;
    const activeEl = logicRefs.current[latestActiveKey];
    const container = logicContainerRef.current;
    if (activeEl && container) {
      setTimeout(() => {
        const containerHeight = container.clientHeight;
        const itemOffsetTop = activeEl.offsetTop;
        const itemHeight = activeEl.offsetHeight;
        const offset = containerHeight * 0.4;
        container.scrollTo({
          top: itemOffsetTop - containerHeight / 2 + itemHeight / 2 + offset,
          behavior: "smooth",
        });
      }, 0);
    }
  }, [latestActiveKey]);

  const activateNodeAndAncestors = useCallback((node: NodeKey) => {
    const newState = { ...nodeActive };
    function activate(n: NodeKey) {
      DAG[n].parents.forEach(activate);
      newState[n] = true;
    }
    activate(node);
    setNodeActive(newState);
  }, [nodeActive]);

  const deactivateNodeAndDescendants = useCallback((node: NodeKey) => {
    const newState = { ...nodeActive };
    function deactivate(n: NodeKey) {
      newState[n] = false;
      DAG[n].children.forEach(deactivate);
    }
    deactivate(node);
    setNodeActive(newState);
  }, [nodeActive]);

  // (outputs definition unchanged)
  const outputs = useMemo(() => ({
    embedding: nodeActive.embedding
      ? [[...inputTokens.map((t) => vocabulary[t].toString())]]
      : [Array(5).fill("")],
    query: nodeActive.query
      ? [["-1.1", "-1.5", "-1.1", "-1.5", "-1.1"]]
      : [Array(5).fill("")],
    key: nodeActive.key
      ? [["-0.1", "-0.3", "-0.1", "-0.3", "-0.1"]]
      : [Array(5).fill("")],
    value: nodeActive.value
      ? [["2.8", "-9.7", "2.3", "-9.7", "2.3"]]
      : [Array(5).fill("")],
    dot: nodeActive.dot
      ? [
        ["0.1", "0.3", "0.1", "0.3", "0.1"],
        ["0.2", "0.5", "0.2", "0.5", "0.2"],
        ["0.1", "0.3", "0.1", "0.3", "0.1"],
        ["0.2", "0.5", "0.2", "0.5", "0.2"],
        ["0.1", "0.3", "0.1", "0.3", "0.1"],
      ]
      : makeMatrix(5, 5, ""),
    argmax: nodeActive.argmax
      ? [
        ["1", "0", "0", "0", "0"],
        ["0", "1", "0", "0", "0"],
        ["0", "1", "0", "0", "0"],
        ["0", "1", "0", "1", "0"],
        ["0", "1", "0", "1", "0"],
      ]
      : makeMatrix(5, 5, ""),
    wsum: nodeActive.wsum
      ? [["2.8", "-9.7", "-9.7", "-9.7", "-9.7"]]
      : [Array(5).fill("")],
    proj: nodeActive.proj
      ? [["8.1", "-20.8", "-20.8", "-20.8", "-20.8"]]
      : [Array(5).fill("")],
    addnorm1: nodeActive.addnorm1
      ? [["8.1", "-32.3", "-20.8", "-32.3", "-20.8"]]
      : [Array(5).fill("")],
    ff_linear1: nodeActive.ff_linear1
      ? [
        ["0", "-16.0", "-16.0", "-16.0", "-16.0"],
        ["-63.9", "234.8", "156.5", "234.8", "156.5"],
        ["-16.0", "78.3", "0", "78.3", "0"],
        ["-16.0", "0", "0", "0", "0"],
      ]
      : makeMatrix(4, 5, ""),
    ff_gelu: nodeActive.ff_gelu
      ? [
        ["0", "0", "0", "0", "0"],
        ["0", "234.8", "167.7", "234.8", "167.7"],
        ["0", "67.1", "0", "67.1", "0"],
        ["0", "0", "0", "0", "0"],
      ]
      : makeMatrix(4, 5, ""),
    ff_linear2: nodeActive.ff_linear2
      ? [["0.7", "-67.6", "-45.1", "-67.6", "-45.1"]]
      : [Array(5).fill("")],
    addnorm2: nodeActive.addnorm2
      ? [["9.4", "-99.9", "-66.6", "-99.9", "-66.6"]]
      : [Array(5).fill("")],
    linear: nodeActive.linear
      ? [["-5.3", "47.4", "31.6", "47.4", "31.6"]]
      : [Array(5).fill("")],
  }), [nodeActive, inputTokens]);

  // --- MAIN RETURN ---
  return (
    <div className="w-full min-h-[80vh] bg-gray-50 flex flex-col items-center justify-start">
    <div className="flex flex-col sm:flex-row w-fit max-w-full h-[80vh] overflow-hidden shadow">
      {/* ---- DIAGRAM COLUMN ---- */}
      <div className="flex-1 min-w-0 max-w-4xl overflow-y-auto px-1 xs:px-2 sm:px-4 md:px-8 py-3">
        <h1 className="text-2xl xs:text-3xl font-bold text-center mb-6 xs:mb-10">
          Example Transformer Decoder
        </h1>

        {/* Input tokens */}
        <TokenMatrix tokens={[inputTokens]} />

        {/* Embedding Layer */}
        <LayerBox
          color={nodeDisplay.embedding.color}
          label={nodeDisplay.embedding.label}
          onPrev={() => deactivateNodeAndDescendants("embedding")}
          onNext={() => activateNodeAndAncestors("embedding")}
        >
          <div className="text-xs text-white w-full flex justify-center mt-1">
            <table className="border-separate border-spacing-x-4 xs:border-spacing-x-6">
              <tbody>
                {[...new Set(inputTokens)].sort().map((token, idx) => (
                  <tr key={idx}>
                    <td className="text-right text-white text-xs xs:text-sm">{token}</td>
                    <td className="text-center text-white">→</td>
                    <td className="text-left text-white text-xs xs:text-sm">{vocabulary[token]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </LayerBox>
        <TokenMatrix tokens={outputs.embedding as string[][]} />

        {/* Decoder Layer (outer box) */}
        <div className="border-4 border-blue-400 bg-blue-50 rounded-3xl p-3 sm:p-8 w-full max-w-4xl mx-auto mb-6 xs:mb-8">
          <h2 className="text-xl xs:text-2xl font-bold text-center mb-5 xs:mb-7 text-blue-900 tracking-wide">
            Decoder Layer
          </h2>

          {/* Masked Self-Attention */}
          <div className="border border-amber-700 bg-amber-50 rounded-2xl p-2 sm:p-6 w-full max-w-4xl mb-4 sm:mb-6">
            <h3 className="text-lg xs:text-xl font-semibold text-center mb-2 xs:mb-4 text-amber-800">
              Masked Self-Attention
            </h3>
            {/* Responsive: stacks on small screens, side-by-side grid on >=sm */}
            <div
              className="
                flex flex-col
                sm:grid sm:grid-cols-3 sm:gap-x-2 sm:gap-y-4
                mt-2 w-full items-stretch
              "
            >
              {/* Query */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.query.color}
                  label={nodeDisplay.query.label}
                  onPrev={() => deactivateNodeAndDescendants("query")}
                  onNext={() => activateNodeAndAncestors("query")}
                />
                <TokenMatrix tokens={outputs.query as string[][]} />
              </div>
              {/* Key */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.key.color}
                  label={nodeDisplay.key.label}
                  onPrev={() => deactivateNodeAndDescendants("key")}
                  onNext={() => activateNodeAndAncestors("key")}
                />
                <TokenMatrix tokens={outputs.key as string[][]} />
              </div>
              {/* Value */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.value.color}
                  label={nodeDisplay.value.label}
                  onPrev={() => deactivateNodeAndDescendants("value")}
                  onNext={() => activateNodeAndAncestors("value")}
                />
                <TokenMatrix tokens={outputs.value as string[][]} />
              </div>
              {/* Dot */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.dot.color}
                  label={nodeDisplay.dot.label}
                  onPrev={() => deactivateNodeAndDescendants("dot")}
                  onNext={() => activateNodeAndAncestors("dot")}
                />
                <TokenMatrix tokens={outputs.dot as string[][]} />
              </div>
              {/* Argmax */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.argmax.color}
                  label={nodeDisplay.argmax.label}
                  onPrev={() => deactivateNodeAndDescendants("argmax")}
                  onNext={() => activateNodeAndAncestors("argmax")}
                />
                <TokenMatrix tokens={outputs.argmax as string[][]} />
              </div>
              {/* Weighted Sum */}
              <div className="flex flex-col items-center sm:col-span-3">
                <SubLayerBox
                  color={nodeDisplay.wsum.color}
                  label={nodeDisplay.wsum.label}
                  onPrev={() => deactivateNodeAndDescendants("wsum")}
                  onNext={() => activateNodeAndAncestors("wsum")}
                />
                <TokenMatrix tokens={outputs.wsum as string[][]} />
              </div>
              {/* Projection */}
              <div className="flex flex-col items-center sm:col-span-3">
                <SubLayerBox
                  color={nodeDisplay.proj.color}
                  label={nodeDisplay.proj.label}
                  onPrev={() => deactivateNodeAndDescendants("proj")}
                  onNext={() => activateNodeAndAncestors("proj")}
                />
                <TokenMatrix tokens={outputs.proj as string[][]} />
              </div>
            </div>
          </div>

          {/* Add & Norm 1 */}
          <LayerBox
            color={nodeDisplay.addnorm1.color}
            label={nodeDisplay.addnorm1.label}
            onPrev={() => deactivateNodeAndDescendants("addnorm1")}
            onNext={() => activateNodeAndAncestors("addnorm1")}
          />
          <TokenMatrix tokens={outputs.addnorm1 as string[][]} />

          {/* Feed Forward sublayers - vertical */}
          <div className="border border-lime-700 bg-lime-50 rounded-2xl p-2 sm:p-6 w-full max-w-4xl mb-4 sm:mb-6 mt-2 sm:mt-4">
            <h3 className="text-lg xs:text-xl font-semibold text-center mb-2 xs:mb-4 text-lime-800">
              Feed Forward
            </h3>
            <div className="flex flex-col gap-2 sm:gap-4 w-full items-center">
              <div className="flex flex-col items-center w-full">
                <SubLayerBox
                  color={nodeDisplay.ff_linear1.color}
                  label={nodeDisplay.ff_linear1.label}
                  onPrev={() => deactivateNodeAndDescendants("ff_linear1")}
                  onNext={() => activateNodeAndAncestors("ff_linear1")}
                />
                <TokenMatrix tokens={outputs.ff_linear1 as string[][]} />
              </div>
              <div className="flex flex-col items-center w-full">
                <SubLayerBox
                  color={nodeDisplay.ff_gelu.color}
                  label={nodeDisplay.ff_gelu.label}
                  onPrev={() => deactivateNodeAndDescendants("ff_gelu")}
                  onNext={() => activateNodeAndAncestors("ff_gelu")}
                />
                <TokenMatrix tokens={outputs.ff_gelu as string[][]} />
              </div>
              <div className="flex flex-col items-center w-full">
                <SubLayerBox
                  color={nodeDisplay.ff_linear2.color}
                  label={nodeDisplay.ff_linear2.label}
                  onPrev={() => deactivateNodeAndDescendants("ff_linear2")}
                  onNext={() => activateNodeAndAncestors("ff_linear2")}
                />
                <TokenMatrix tokens={outputs.ff_linear2 as string[][]} />
              </div>
            </div>
          </div>

          {/* Add & Norm 2 */}
          <LayerBox
            color={nodeDisplay.addnorm2.color}
            label={nodeDisplay.addnorm2.label}
            onPrev={() => deactivateNodeAndDescendants("addnorm2")}
            onNext={() => activateNodeAndAncestors("addnorm2")}
          />
          <TokenMatrix tokens={outputs.addnorm2 as string[][]} />
        </div>

        {/* Linear */}
        <LayerBox
          color={nodeDisplay.linear.color}
          label={nodeDisplay.linear.label}
          onPrev={() => deactivateNodeAndDescendants("linear")}
          onNext={() => activateNodeAndAncestors("linear")}
        />
        <TokenMatrix tokens={outputs.linear as string[][]} />
      </div>

      {/* --- LOGIC WALKTHROUGH SIDEBAR --- */}
<aside
  className={`
    w-[15rem] xs:w-[18rem] sm:w-[22rem] md:w-[28rem] lg:w-[42rem]
    max-w-full h-[80vh] sticky top-0
    overflow-y-auto overflow-x-auto
    border-l pl-0 pr-0
    pt-3 xs:pt-4 sm:pt-6 md:pt-8
    flex-shrink-0
    text-[12px] xs:text-[13px] sm:text-[14px] md:text-[15px] lg:text-[16px]
  `}
  ref={logicContainerRef}
>




{/* <h2 className="text-2xl font-bold text-center mb-6 tracking-tight">
  Equivalent Logical Expressions
</h2> */}
        <div className="h-[40vh]" />
        <div className="space-y-4 xs:space-y-6 text-left leading-relaxed">
          {displayOrder.filter(k => nodeActive[k]).map((key) => (
            <div
              key={key}
              ref={(el) => { logicRefs.current[key] = el }}
              className={`transition-all px-2 xs:px-6 py-2 xs:py-4 rounded-lg ${
                key === latestActiveKey
                  ? "bg-blue-50 shadow-md font-semibold"
                  : "text-gray-700"
              }`}
            >
              {/* @ts-ignore */}
              {nodeLogics[key]}
            </div>
          ))}
        </div>
        <div className="h-[40vh]" />
      </aside>
    </div>
    </div>
  );
}

// Place your nodeLogics, vocabulary, and makeMatrix functions as in your original file!
